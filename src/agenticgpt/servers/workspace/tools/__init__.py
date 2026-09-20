"""Tools the workspace server exposes.

One module per tool. Each defines a ``register(mcp)`` that attaches its
function to the server, so adding a tool is: write the module, add one line
to :data:`TOOL_MODULES`.

The decorated function's **signature and docstring are the schema** the model
reads. FastMCP generates it from the type hints, which is why there is no
hand-written JSON anywhere in this package — that was version 1's job.
"""

from __future__ import annotations

from fastmcp import FastMCP

from . import calculator, currency, search, weather

TOOL_MODULES = (calculator, currency, search, weather)


def register_tools(mcp: FastMCP) -> None:
    """Attach every tool module to the server."""
    for module in TOOL_MODULES:
        module.register(mcp)


__all__ = ["TOOL_MODULES", "register_tools"]
