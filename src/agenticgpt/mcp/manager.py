"""One front door to every connected MCP server.

The agent holds a manager and asks it for tools by name::

    result = await mcp_manager.call_tool(tool_name, arguments)

It never learns which server answered. Adding a second server is a change to
:mod:`agenticgpt.mcp.config` and nothing else.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from .client import MCPClient
from .config import MCPServerConfig, load_servers
from .registry import CapabilityRegistry, ToolSpec


class MCPManager:
    """Connects to the configured MCP servers and routes calls to them.

    Usage::

        async with MCPManager() as manager:
            tools = manager.tools()
            answer = await manager.call_tool("calculator", {"expression": "2+2"})

    The manager satisfies the small "toolbox" interface the agent loop expects
    — :meth:`tools` and :meth:`call_tool` — which is the whole of what the
    agent knows about MCP.
    """

    def __init__(
        self,
        servers: Sequence[MCPServerConfig] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        """
        Args:
            servers: Server configs. Defaults to
                :func:`agenticgpt.mcp.config.load_servers`.
            on_event: Called with a human-readable line for each notable step.
                Used by the CLI and the Streamlit trace; ``None`` stays silent.
        """
        self._configs = list(servers) if servers is not None else load_servers()
        self._on_event = on_event
        self._clients: dict[str, MCPClient] = {}
        self.registry = CapabilityRegistry()

    def _emit(self, message: str) -> None:
        if self._on_event is not None:
            self._on_event(message)

    # -- lifecycle ----------------------------------------------------------
    async def connect(self) -> "MCPManager":
        """Connect to every enabled server and discover what it offers."""
        for config in self._configs:
            if not config.enabled:
                continue
            client = MCPClient(config)
            try:
                await client.connect()
            except Exception as exc:  # one bad server must not sink the rest
                self._emit(f"MCP server {config.name!r} failed to start: {exc}")
                continue
            self._clients[config.name] = client
            await self._discover(client)

        for name, first, second in self.registry.conflicts:
            self._emit(f"name clash: {name!r} offered by both {first!r} and {second!r}")

        return self

    async def _discover(self, client: MCPClient) -> None:
        for spec in await client.list_tools():
            self.registry.add_tool(spec)
        for spec in await client.list_resources():
            self.registry.add_resource(spec)
        for spec in await client.list_prompts():
            self.registry.add_prompt(spec)

        owned = [t.name for t in self.registry.tool_specs() if t.server == client.name]
        self._emit(f"MCP server {client.name!r} offers: {', '.join(owned) or 'nothing'}")

    async def close(self) -> None:
        """Close every session."""
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception as exc:
                self._emit(f"MCP server {client.name!r} did not close cleanly: {exc}")
        self._clients.clear()
        self.registry.clear()

    async def __aenter__(self) -> "MCPManager":
        return await self.connect()

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    # -- capabilities -------------------------------------------------------
    def tools(self) -> list[ToolSpec]:
        """Every tool across every connected server."""
        return self.registry.tool_specs()

    def server_names(self) -> Iterable[str]:
        return self._clients.keys()

    def _client_for(self, server_name: str) -> MCPClient:
        client = self._clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP server {server_name!r} is not connected")
        return client

    # -- execution ----------------------------------------------------------
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name, wherever it lives.

        Returns the tool's output as text, including for failures: the agent
        loop feeds this straight back to the model, and a model that is told
        what went wrong can retry or explain, while a raised exception ends
        the run.
        """
        try:
            server_name = self.registry.server_for_tool(tool_name)
        except KeyError as exc:
            return f"Tool unavailable: {exc}"

        try:
            return await self._client_for(server_name).call_tool(tool_name, arguments)
        except Exception as exc:
            return f"Tool failed: {type(exc).__name__}: {exc}"

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI, wherever it lives."""
        try:
            server_name = self.registry.server_for_resource(uri)
        except KeyError as exc:
            return f"Resource unavailable: {exc}"
        return await self._client_for(server_name).read_resource(uri)

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None):
        """Fetch a prompt template by name, wherever it lives."""
        server_name = self.registry.server_for_prompt(name)
        return await self._client_for(server_name).get_prompt(name, arguments)

    def __repr__(self) -> str:
        return (
            f"<MCPManager servers={list(self._clients)} "
            f"tools={len(self.registry.tools)}>"
        )
