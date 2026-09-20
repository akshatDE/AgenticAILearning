"""Prompt templates the workspace server exposes.

The third MCP capability. A **prompt** is a named, parameterised message
template the server owns and the client fetches — it lets a server ship the
wording that gets the best out of its own tools, instead of every client
inventing its own.

The workspace server offers none yet. Adding one::

    @mcp.prompt
    def summarise_search(query: str) -> str:
        return f"Search for {query}, then summarise the results in three bullets."

:class:`~agenticgpt.mcp.manager.MCPManager` already discovers and routes
prompts, so the client side needs no changes when that day comes.
"""

from __future__ import annotations

from fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Attach every prompt to the server. None yet — see the module docstring."""


__all__ = ["register_prompts"]
