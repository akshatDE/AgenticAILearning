"""The loop, driven by a scripted model and a fake toolbox.

No Ollama and no MCP server involved — which is the point of the loop knowing
only `.chat()` on one side and `.tools()` / `.call_tool()` on the other.
"""

import asyncio

from agenticgpt.agent.loop import AgentLoop, parse_arguments
from agenticgpt.mcp.registry import ToolSpec


class ScriptedLLM:
    """Returns prepared replies, one per turn, and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=(), model=None):
        self.calls.append({"messages": messages, "tools": list(tools), "model": model})
        return self.replies.pop(0) if self.replies else {"role": "assistant", "content": "done"}


class FakeToolbox:
    """Stands in for MCPManager."""

    def __init__(self, results=None):
        self.results = results or {}
        self.executed = []

    def tools(self):
        return [ToolSpec("calculator", "adds things up", {}, "workspace")]

    async def call_tool(self, tool_name, arguments):
        self.executed.append((tool_name, arguments))
        return self.results.get(tool_name, "ok")


def tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def run(coro):
    return asyncio.run(coro)


def test_answers_without_tools_when_none_are_requested():
    llm = ScriptedLLM([{"role": "assistant", "content": "42"}])
    toolbox = FakeToolbox()

    answer = run(AgentLoop(llm, toolbox).run("what is it?"))

    assert answer == "42"
    assert toolbox.executed == []


def test_executes_a_requested_tool_and_feeds_the_result_back():
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": "", "tool_calls": [tool_call("calculator", {"expression": "2+2"})]},
            {"role": "assistant", "content": "It is 4."},
        ]
    )
    toolbox = FakeToolbox({"calculator": "4"})

    answer = run(AgentLoop(llm, toolbox).run("2+2?"))

    assert answer == "It is 4."
    assert toolbox.executed == [("calculator", {"expression": "2+2"})]

    # The second request has to carry the tool result, or the model is
    # answering from nothing.
    second_request = llm.calls[1]["messages"]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["content"] == "4"
    assert second_request[-1]["tool_call_id"] == "call_1"


def test_json_string_arguments_are_parsed():
    # Groq returns a JSON string where Ollama returns a dict.
    llm = ScriptedLLM(
        [
            {"role": "assistant", "tool_calls": [tool_call("calculator", '{"expression": "2+2"}')]},
            {"role": "assistant", "content": "4"},
        ]
    )
    toolbox = FakeToolbox()

    run(AgentLoop(llm, toolbox).run("2+2?"))

    assert toolbox.executed == [("calculator", {"expression": "2+2"})]


def test_several_tool_calls_in_one_turn_all_run():
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("calculator", {"expression": "1+1"}, "a"),
                    tool_call("calculator", {"expression": "2+2"}, "b"),
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )
    toolbox = FakeToolbox()

    run(AgentLoop(llm, toolbox).run("two sums"))

    assert len(toolbox.executed) == 2


def test_stops_at_the_turn_limit():
    forever = {"role": "assistant", "tool_calls": [tool_call("calculator", {})]}
    llm = ScriptedLLM([forever] * 10)
    toolbox = FakeToolbox()

    answer = run(AgentLoop(llm, toolbox, max_turns=3).run("loop forever"))

    assert "Stopped after 3 turns" in answer
    assert len(toolbox.executed) == 3


def test_system_prompt_is_sent_when_given():
    llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])

    run(AgentLoop(llm, FakeToolbox(), system_prompt="Be terse.").run("hi"))

    assert llm.calls[0]["messages"][0] == {"role": "system", "content": "Be terse."}


def test_discovered_tools_are_handed_to_the_model():
    llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])

    run(AgentLoop(llm, FakeToolbox()).run("hi"))

    assert [tool.name for tool in llm.calls[0]["tools"]] == ["calculator"]


def test_trace_events_are_emitted():
    llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])
    steps = []

    run(AgentLoop(llm, FakeToolbox(), on_event=steps.append).run("hi"))

    assert any("calculator" in step for step in steps)


def test_parse_arguments_copes_with_junk():
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("not json") == {}
    assert parse_arguments("[1, 2]") == {}
    assert parse_arguments(None) == {}
