"""Server details live in one place."""

import json

from agenticgpt.mcp.config import BUILTIN_SERVERS, load_servers


def test_the_workspace_server_script_exists():
    workspace = BUILTIN_SERVERS[0]

    assert workspace.name == "workspace"
    assert workspace.transport == "stdio"
    assert workspace.script_path.is_file()


def test_defaults_are_used_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("AGENTICGPT_MCP_CONFIG", raising=False)

    assert [s.name for s in load_servers()] == ["workspace"]


def test_a_json_file_can_replace_the_defaults(tmp_path):
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps([{"name": "remote", "transport": "http", "url": "http://x/mcp"}])
    )

    servers = load_servers(config)

    assert [s.name for s in servers] == ["remote"]
    assert servers[0].url == "http://x/mcp"
