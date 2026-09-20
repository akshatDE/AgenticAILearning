"""A connection to exactly one MCP server.

This module knows about sessions, transports and MCP result payloads. It
knows nothing about agents, loops, models or prompts — if reasoning ever
appears in here, it is in the wrong file.

One :class:`MCPClient` wraps one server. Coordinating several of them is
:mod:`agenticgpt.mcp.manager`'s job.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from .config import MCPServerConfig
from .registry import PromptSpec, ResourceSpec, ToolSpec


def result_to_text(result: Any) -> str:
    """Flatten an MCP call result into text the model can read.

    FastMCP 2.x returns a ``CallToolResult`` carrying ``.content``; older
    versions returned the content list directly. Both are handled.
    """
    blocks = getattr(result, "content", result)
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    return "\n".join(
        block.text for block in blocks if getattr(block, "text", None) is not None
    )


def _build_transport(config: MCPServerConfig):
    """Turn a config entry into something FastMCP can connect through."""
    if config.transport == "stdio":
        env = {**os.environ, **config.env} if config.env else None
        return PythonStdioTransport(script_path=str(config.script_path), env=env)
    # http — FastMCP infers the transport from the URL.
    return config.url


class MCPClient:
    """One MCP server: connect, discover what it offers, call into it.

    A client is deliberately **not** reused across event loops. It holds
    subprocess pipes bound to the loop that opened them, so a client created
    under one ``asyncio.run()`` breaks under the next. Streamlit calls
    ``asyncio.run()`` once per message, which is exactly that situation —
    build a fresh client per run instead of caching one at import time.

    Usage::

        async with MCPClient(config) as client:
            tools = await client.list_tools()
            text = await client.call_tool("calculator", {"expression": "2+2"})
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._client: Client | None = None
        self._stack: AsyncExitStack | None = None

    @property
    def name(self) -> str:
        return self.config.name

    # -- session lifecycle --------------------------------------------------
    async def connect(self) -> "MCPClient":
        """Open the session. For stdio servers this spawns the subprocess."""
        if self._client is not None:
            return self

        stack = AsyncExitStack()
        client = Client(_build_transport(self.config))
        await stack.enter_async_context(client)

        self._stack = stack
        self._client = client
        return self

    async def close(self) -> None:
        """Close the session and, for stdio servers, stop the subprocess."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._client = None

    async def __aenter__(self) -> "MCPClient":
        return await self.connect()

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    def _require_session(self) -> Client:
        if self._client is None:
            raise RuntimeError(
                f"MCP server {self.name!r} is not connected — "
                "use 'async with MCPClient(config)' or await connect() first"
            )
        return self._client

    # -- discovery ----------------------------------------------------------
    @staticmethod
    def _input_schema(tool: Any) -> dict[str, Any]:
        """Read a tool's JSON schema across FastMCP versions.

        FastMCP 2.x exposes ``inputSchema``; 4.x renamed it to
        ``input_schema`` and deprecated the old name.
        """
        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = getattr(tool, "inputSchema", None)
        return schema or {}

    async def list_tools(self) -> list[ToolSpec]:
        """Ask the server which tools it offers."""
        tools = await self._require_session().list_tools()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=self._input_schema(tool),
                server=self.name,
            )
            for tool in tools
        ]

    async def list_resources(self) -> list[ResourceSpec]:
        """Ask the server which resources it offers.

        Servers are free not to implement resources at all; that is reported
        as an error over the wire, and an empty list is the useful answer.
        """
        try:
            resources = await self._require_session().list_resources()
        except Exception:
            return []
        return [
            ResourceSpec(
                uri=str(resource.uri),
                name=getattr(resource, "name", "") or "",
                description=getattr(resource, "description", "") or "",
                mime_type=getattr(resource, "mimeType", None),
                server=self.name,
            )
            for resource in resources
        ]

    async def list_prompts(self) -> list[PromptSpec]:
        """Ask the server which prompt templates it offers."""
        try:
            prompts = await self._require_session().list_prompts()
        except Exception:
            return []
        return [
            PromptSpec(
                name=prompt.name,
                description=getattr(prompt, "description", "") or "",
                arguments=[
                    argument.model_dump()
                    if hasattr(argument, "model_dump")
                    else dict(argument)
                    for argument in (getattr(prompt, "arguments", None) or [])
                ],
                server=self.name,
            )
            for prompt in prompts
        ]

    # -- execution ----------------------------------------------------------
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run a tool on this server and return its result as text."""
        result = await self._require_session().call_tool(tool_name, arguments)
        return result_to_text(result)

    async def read_resource(self, uri: str) -> str:
        """Read a resource from this server and return it as text."""
        result = await self._require_session().read_resource(uri)
        return result_to_text(result)

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None):
        """Fetch a rendered prompt template from this server."""
        return await self._require_session().get_prompt(name, arguments or {})

    def __repr__(self) -> str:
        state = "connected" if self._client is not None else "disconnected"
        return f"<MCPClient {self.name!r} {state}>"
