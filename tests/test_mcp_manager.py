"""The manager, over a real stdio subprocess.

Slower than the rest of the suite because it spawns `server.py`, but it is
the check that actually proves the restructured package still wires together:
config -> client -> discovery -> registry -> routed call.
"""

import asyncio

from agenticgpt.mcp import MCPManager


def run(coro):
    return asyncio.run(coro)


async def _discover():
    async with MCPManager() as manager:
        return [tool.name for tool in manager.tools()], list(manager.server_names())


def test_tools_are_discovered_from_the_configured_server():
    tool_names, servers = run(_discover())

    assert servers == ["workspace"]
    assert "calculator" in tool_names


async def _call(tool_name, arguments):
    async with MCPManager() as manager:
        return await manager.call_tool(tool_name, arguments)


def test_a_tool_is_routed_to_whichever_server_owns_it():
    assert run(_call("calculator", {"expression": "2 + 2"})) == "4"


def test_an_unknown_tool_is_reported_back_as_text():
    # The agent loop feeds this to the model, which can then say so or retry.
    # Raising here would end the run instead.
    result = run(_call("no_such_tool", {}))

    assert "unavailable" in result.lower()


async def _run_agent_end_to_end():
    from agenticgpt import run_agent

    class ScriptedLLM:
        """Stands in for Ollama, so the test needs no local model."""

        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools=(), model=None):
            self.turn += 1
            if self.turn == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "15 * 4"},
                            },
                        }
                    ],
                }
            return {"role": "assistant", "content": f"The answer is {messages[-1]['content']}."}

    return await run_agent("what is 15 times 4?", llm=ScriptedLLM())


def test_the_whole_stack_runs_with_a_scripted_model():
    assert run(_run_agent_end_to_end()) == "The answer is 60."
