"""Where MCP servers are declared.

One place that says which servers exist and how to reach them. Nothing else
in the project hardcodes a server path, a command, or a URL.

Today every server is a local Python script spoken to over stdio. The shape
below leaves room for HTTP/SSE servers without touching the client or the
manager: add a config with ``transport="http"`` and a ``url``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository-relative root of the bundled servers package.
SERVERS_ROOT = Path(__file__).resolve().parent.parent / "servers"

#: Set this to a JSON file to override the built-in server list.
CONFIG_ENV_VAR = "AGENTICGPT_MCP_CONFIG"


@dataclass(frozen=True)
class MCPServerConfig:
    """How to reach one MCP server.

    Args:
        name: Identifier used in logs and by the registry to say which server
            owns a capability.
        transport: ``"stdio"`` (spawn a local process) or ``"http"``.
        script_path: For stdio: the Python script to run as a subprocess.
        url: For http: the server endpoint.
        env: Extra environment variables for a spawned server. The subprocess
            already inherits the parent environment, so this is only for
            overrides — never put secrets here.
        enabled: Skipped by the manager when ``False``.
    """

    name: str
    transport: str = "stdio"
    script_path: Path | None = None
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.transport == "stdio" and self.script_path is None:
            raise ValueError(f"stdio server {self.name!r} needs a script_path")
        if self.transport == "http" and not self.url:
            raise ValueError(f"http server {self.name!r} needs a url")
        if self.transport not in ("stdio", "http"):
            raise ValueError(f"unknown transport {self.transport!r}")


#: The servers AgenticGPT ships with.
BUILTIN_SERVERS: tuple[MCPServerConfig, ...] = (
    MCPServerConfig(
        name="workspace",
        transport="stdio",
        script_path=SERVERS_ROOT / "workspace" / "server.py",
    ),
)


def _config_from_dict(raw: dict) -> MCPServerConfig:
    script_path = raw.get("script_path")
    return MCPServerConfig(
        name=raw["name"],
        transport=raw.get("transport", "stdio"),
        script_path=Path(script_path).expanduser() if script_path else None,
        url=raw.get("url"),
        env=raw.get("env", {}),
        enabled=raw.get("enabled", True),
    )


def load_servers(config_path: str | Path | None = None) -> list[MCPServerConfig]:
    """Return the MCP servers to connect to.

    Reads a JSON file when one is given or when ``AGENTICGPT_MCP_CONFIG``
    points at one; otherwise returns :data:`BUILTIN_SERVERS`.

    The JSON is a list of objects with the same keys as
    :class:`MCPServerConfig`::

        [{"name": "workspace", "transport": "stdio", "script_path": "..."}]
    """
    path = config_path or os.environ.get(CONFIG_ENV_VAR)
    if not path:
        return list(BUILTIN_SERVERS)

    raw = json.loads(Path(path).expanduser().read_text())
    return [_config_from_dict(entry) for entry in raw]
