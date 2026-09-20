"""Discovered MCP tools have to reach the provider in its own format."""

from agenticgpt.agent.llm import GroqClient, tools_to_schemas
from agenticgpt.mcp.registry import ToolSpec


def test_tool_spec_becomes_an_openai_function_definition():
    schema = {"type": "object", "properties": {"expression": {"type": "string"}}}
    spec = ToolSpec("calculator", "Adds things up.", schema, "workspace")

    assert tools_to_schemas([spec]) == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Adds things up.",
                "parameters": schema,
            },
        }
    ]


def test_no_tools_is_an_empty_list_not_an_error():
    assert tools_to_schemas([]) == []


def test_missing_groq_key_is_reported_to_the_model_not_raised(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    reply = GroqClient().chat([{"role": "user", "content": "hi"}])

    assert "GROQ_API_KEY" in reply["content"]
