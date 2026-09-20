"""MCP client infrastructure.

AgenticGPT is an MCP *client*. This package is everything that entails:

- :mod:`~agenticgpt.mcp.config` — which servers exist and how to reach them
- :mod:`~agenticgpt.mcp.client` — one connection to one server
- :mod:`~agenticgpt.mcp.registry` — what was discovered, and who owns it
- :mod:`~agenticgpt.mcp.manager` — one front door over all of them

No module in here imports from :mod:`agenticgpt.agent`. The dependency runs
one way: the agent uses MCP, MCP knows nothing about agents.
"""

from .client import MCPClient, result_to_text
from .config import BUILTIN_SERVERS, MCPServerConfig, load_servers
from .manager import MCPManager
from .registry import (
    CapabilityRegistry,
    PromptSpec,
    ResourceSpec,
    ToolSpec,
    UnknownCapability,
)

__all__ = [
    "BUILTIN_SERVERS",
    "CapabilityRegistry",
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "PromptSpec",
    "ResourceSpec",
    "ToolSpec",
    "UnknownCapability",
    "load_servers",
    "result_to_text",
]
