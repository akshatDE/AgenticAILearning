"""
AgenticGPT agent loop.

Spawns server.py as an MCP stdio subprocess, discovers its tools, hands them
to a local Ollama model, and runs the tool-calling loop until the model
answers without requesting another tool.

Everything printed here is the trace the Streamlit frontend captures.

Run standalone with:  python app.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from pprint import pprint

import requests
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
REQUEST_TIMEOUT = 120  # local models on CPU are slow; 60s times out mid-answer

SERVER_PATH = Path(__file__).parent / "server.py"


def make_client() -> Client:
    """Build a fresh MCP client.

    Deliberately not a module-level singleton. Streamlit calls asyncio.run()
    once per message, which creates a new event loop each time, and a client
    holds subprocess pipes bound to the loop that opened them. Reusing one
    across loops works for the first message and fails on the second.
    """
    return Client(PythonStdioTransport(script_path=str(SERVER_PATH)))


# ---------------------------------------------------------------------------
# MCP tool schema -> Ollama tool schema
# ---------------------------------------------------------------------------
def convert_mcp_tools_to_ollama(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def ask_ai(messages: list, tools: list, model: str = DEFAULT_MODEL) -> dict:
    """Send the conversation and the available tools to Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "think": False,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("error", "")
        except Exception:
            detail = exc.response.text[:200] if exc.response is not None else ""
        return {
            "role": "assistant",
            "content": f"Ollama rejected the request: {detail or exc}",
        }
    except requests.RequestException as exc:
        return {"role": "assistant", "content": f"Could not reach Ollama: {exc}"}
    except (KeyError, ValueError):
        return {"role": "assistant", "content": "Ollama returned an unexpected response."}


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------
def result_to_text(result) -> str:
    """Flatten an MCP tool result into text the model can read.

    FastMCP 2.x returns a CallToolResult with .content; older versions
    returned the content list directly.
    """
    blocks = getattr(result, "content", result)
    if blocks is None:
        return ""
    return "\n".join(block.text for block in blocks if hasattr(block, "text"))


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
async def run_agent(
    user_message: str,
    max_turns: int = 5,
    model: str = DEFAULT_MODEL,
) -> str:
    messages = [{"role": "user", "content": user_message}]

    async with make_client() as client:

        # 1. Discover what the server offers.
        mcp_tools = await client.list_tools()

        print("\nMCP tools discovered:")
        for tool in mcp_tools:
            print(f"- {tool.name}")

        # 2. Translate for Ollama.
        llm_tools = convert_mcp_tools_to_ollama(mcp_tools)

        # 3. Loop until the model answers without asking for a tool.
        for turn in range(max_turns):

            response = ask_ai(messages, llm_tools, model=model)
            messages.append(response)

            print(f"\nAI response — turn {turn + 1}:")
            pprint(response)

            tool_calls = response.get("tool_calls")

            if not tool_calls:
                return response.get("content", "")

            for tool_call in tool_calls:
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = function["arguments"]

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                print(f"\nTool requested: {tool_name}")
                print("Arguments:")
                pprint(arguments)

                # 4. MCP runs the tool.
                try:
                    result = await client.call_tool(tool_name, arguments)
                    result_text = result_to_text(result)
                except Exception as exc:
                    result_text = f"Tool failed: {type(exc).__name__}: {exc}"

                print("MCP result:")
                pprint(result_text)

                # 5. Hand the result back to the model.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", tool_name),
                        "content": result_text,
                    }
                )

    return (
        f"Stopped after {max_turns} turns without reaching an answer. "
        "Try a narrower question."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    user_message = input("You: ")
    answer = await run_agent(user_message)
    print("\nAgent:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())