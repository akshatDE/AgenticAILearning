"""Resources the workspace server exposes.

MCP servers offer three kinds of capability. Tools *do* things; **resources**
are things the client can *read* — a file, a schema, a config blob — addressed
by URI and fetched with ``read_resource``.

The workspace server offers none yet. The seam is here so adding one is a
module plus a line in :func:`register_resources`, exactly as tools work::

    @mcp.resource("workspace://notes")
    def notes() -> str:
        return (WORKSPACE_ROOT / "notes.md").read_text()

:class:`~agenticgpt.mcp.registry.CapabilityRegistry` already stores resources
and :class:`~agenticgpt.mcp.manager.MCPManager` already routes reads, so the
client side needs no changes when that day comes.
"""

from __future__ import annotations

from fastmcp import FastMCP


def register_resources(mcp: FastMCP) -> None:
    """Attach every resource to the server. None yet — see the module docstring."""


__all__ = ["register_resources"]
