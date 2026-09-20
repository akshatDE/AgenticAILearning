"""The registry is what lets the agent ask for a tool without naming a server."""

import pytest

from agenticgpt.mcp.registry import CapabilityRegistry, ToolSpec, UnknownCapability


def spec(name: str, server: str) -> ToolSpec:
    return ToolSpec(name=name, description="", input_schema={}, server=server)


def test_routes_a_tool_to_its_server():
    registry = CapabilityRegistry()
    registry.add_tool(spec("calculator", "workspace"))
    registry.add_tool(spec("read_file", "files"))

    assert registry.server_for_tool("calculator") == "workspace"
    assert registry.server_for_tool("read_file") == "files"


def test_unknown_tool_is_reported_not_guessed():
    registry = CapabilityRegistry()

    with pytest.raises(UnknownCapability):
        registry.server_for_tool("nope")


def test_name_clash_keeps_the_first_and_records_it():
    registry = CapabilityRegistry()
    registry.add_tool(spec("calculator", "workspace"))
    registry.add_tool(spec("calculator", "other"))

    assert registry.server_for_tool("calculator") == "workspace"
    assert registry.conflicts == [("calculator", "workspace", "other")]


def test_clear_empties_everything():
    registry = CapabilityRegistry()
    registry.add_tool(spec("calculator", "workspace"))
    registry.clear()

    assert len(registry) == 0
