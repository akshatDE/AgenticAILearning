"""The workspace MCP server, exercised in-process.

fastmcp's Client speaks to a FastMCP object directly, with no subprocess, so
these run in milliseconds.
"""

import asyncio

from fastmcp import Client

from agenticgpt.servers.workspace import build_server

EXPECTED_TOOLS = {"calculator", "convert_currency", "get_weather", "search_internet"}


def run(coro):
    return asyncio.run(coro)


async def _list_tool_names():
    async with Client(build_server()) as client:
        return {tool.name for tool in await client.list_tools()}


def test_every_tool_is_registered():
    assert run(_list_tool_names()) == EXPECTED_TOOLS


async def _schema_for(tool_name):
    async with Client(build_server()) as client:
        tools = await client.list_tools()
    return next(tool for tool in tools if tool.name == tool_name)


def schema_of(tool):
    """FastMCP 2.x calls it inputSchema; 4.x calls it input_schema."""
    return getattr(tool, "input_schema", None) or tool.inputSchema


def test_schemas_are_generated_from_the_signature():
    # No hand-written JSON anywhere: FastMCP reads the type hints. This is
    # the thing version 2 bought over version 1.
    tool = run(_schema_for("convert_currency"))

    assert set(schema_of(tool)["properties"]) == {
        "amount",
        "from_currency",
        "to_currency",
    }
    assert tool.description.startswith("Convert an amount between two currencies")


async def _call(tool_name, arguments):
    async with Client(build_server()) as client:
        result = await client.call_tool(tool_name, arguments)
    return result


def test_calculator_runs_over_mcp():
    result = run(_call("calculator", {"expression": "(145 * 32) + 78"}))

    assert result.content[0].text == "4718"


def test_missing_key_is_answered_not_raised(monkeypatch):
    # A tool that cannot work still has to return something the model can read.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = run(_call("search_internet", {"query": "anything"}))

    assert "TAVILY_API_KEY" in result.content[0].text
