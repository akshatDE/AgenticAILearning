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


def test_groq_stops_calling_after_the_request_limit(monkeypatch):
    from unittest import mock

    GroqClient._sent.clear()
    client = GroqClient(api_key="k", max_requests=2, window_seconds=60)
    ok = mock.Mock()
    ok.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    with mock.patch("agenticgpt.agent.llm.requests.post", return_value=ok) as post:
        replies = [client.chat([{"role": "user", "content": "hi"}]) for _ in range(3)]

    GroqClient._sent.clear()
    assert post.call_count == 2
    assert "request limit reached" in replies[2]["content"]
