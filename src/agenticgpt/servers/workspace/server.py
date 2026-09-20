"""The workspace MCP server.

An MCP server, not part of the agent. It is launched as a stdio subprocess by
:class:`~agenticgpt.mcp.client.MCPClient`, it inherits the parent's
environment (which is how API keys reach it), and it would work just as well
driven by Claude Desktop or any other MCP client.

Nothing in this package imports the agent, and the agent never imports this.
The only contract between them is MCP itself.

Run standalone with::

    python -m agenticgpt.servers.workspace.server
"""

from __future__ import annotations

import sys
from pathlib import Path

# Started as a bare script by the stdio transport (``python server.py``), the
# package root is not on sys.path and the absolute imports below fail. Put it
# there. A no-op when AgenticGPT is installed or run with -m.
if __package__ in (None, ""):  # pragma: no cover - only hit as a spawned script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastmcp import FastMCP  # noqa: E402

from agenticgpt.servers.workspace.prompts import register_prompts  # noqa: E402
from agenticgpt.servers.workspace.resources import register_resources  # noqa: E402
from agenticgpt.servers.workspace.tools import register_tools  # noqa: E402

INSTRUCTIONS = """\
Everyday capabilities for AgenticGPT: arithmetic, live currency conversion,
current weather, and web search.
"""


def build_server() -> FastMCP:
    """Create the server and attach every capability it offers."""
    mcp = FastMCP("agenticgpt-workspace", INSTRUCTIONS)
    register_tools(mcp)
    register_resources(mcp)
    register_prompts(mcp)
    return mcp


def main() -> None:
    """Serve over stdio. The FastMCP banner goes to stderr, so it cannot
    corrupt the JSON-RPC stream on stdout."""
    build_server().run()


if __name__ == "__main__":
    main()
