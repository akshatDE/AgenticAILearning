"""What the connected MCP servers can do, and which one does it.

The registry is a lookup table, nothing more. It holds no connections and
performs no I/O — :class:`~agenticgpt.mcp.manager.MCPManager` fills it during
discovery and reads it when routing a call.

It exists so the agent can ask for a tool by name without knowing, or caring,
which server owns it. That indirection is what makes a second MCP server a
configuration change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


class UnknownCapability(KeyError):
    """Raised when a name is not offered by any connected server."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool, and the server that owns it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server: str


@dataclass(frozen=True)
class ResourceSpec:
    """One resource, and the server that owns it."""

    uri: str
    name: str
    description: str
    mime_type: str | None
    server: str


@dataclass(frozen=True)
class PromptSpec:
    """One prompt template, and the server that owns it."""

    name: str
    description: str
    arguments: list[dict[str, Any]]
    server: str


@dataclass
class CapabilityRegistry:
    """Every capability discovered across every connected server.

    Names are assumed unique across servers. If two servers offer the same
    tool name the first one registered wins and the clash is recorded in
    :attr:`conflicts`, so the situation is visible rather than silent.
    """

    tools: dict[str, ToolSpec] = field(default_factory=dict)
    resources: dict[str, ResourceSpec] = field(default_factory=dict)
    prompts: dict[str, PromptSpec] = field(default_factory=dict)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)

    # -- registration -------------------------------------------------------
    def add_tool(self, spec: ToolSpec) -> None:
        existing = self.tools.get(spec.name)
        if existing is not None:
            self.conflicts.append((spec.name, existing.server, spec.server))
            return
        self.tools[spec.name] = spec

    def add_resource(self, spec: ResourceSpec) -> None:
        if spec.uri in self.resources:
            existing = self.resources[spec.uri]
            self.conflicts.append((spec.uri, existing.server, spec.server))
            return
        self.resources[spec.uri] = spec

    def add_prompt(self, spec: PromptSpec) -> None:
        if spec.name in self.prompts:
            existing = self.prompts[spec.name]
            self.conflicts.append((spec.name, existing.server, spec.server))
            return
        self.prompts[spec.name] = spec

    # -- lookup -------------------------------------------------------------
    def server_for_tool(self, tool_name: str) -> str:
        """Return the name of the server that owns ``tool_name``."""
        spec = self.tools.get(tool_name)
        if spec is None:
            raise UnknownCapability(
                f"no connected MCP server offers a tool named {tool_name!r}"
            )
        return spec.server

    def server_for_resource(self, uri: str) -> str:
        spec = self.resources.get(uri)
        if spec is None:
            raise UnknownCapability(f"no connected MCP server offers {uri!r}")
        return spec.server

    def server_for_prompt(self, prompt_name: str) -> str:
        spec = self.prompts.get(prompt_name)
        if spec is None:
            raise UnknownCapability(
                f"no connected MCP server offers a prompt named {prompt_name!r}"
            )
        return spec.server

    def tool_specs(self) -> list[ToolSpec]:
        """Every tool, in registration order."""
        return list(self.tools.values())

    def clear(self) -> None:
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        self.conflicts.clear()

    def __len__(self) -> int:
        return len(self.tools) + len(self.resources) + len(self.prompts)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.tools.values())
