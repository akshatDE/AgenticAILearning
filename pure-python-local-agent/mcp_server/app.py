import asyncio
import json
from pprint import pprint

import requests
from fastmcp import Client


# ---------------------------------------------------------
# MCP CLIENT
# ---------------------------------------------------------

client = Client("server.py")


# ---------------------------------------------------------
# MCP TOOLS -> OLLAMA TOOL FORMAT
# ---------------------------------------------------------

def convert_mcp_tools_to_ollama(mcp_tools):
    """
    Convert tools discovered from the MCP server into the
    tool schema expected by Ollama.
    """

    ollama_tools = []

    for tool in mcp_tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )

    return ollama_tools


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

def ask_ai(messages: list, tools: list,model: str = "qwen3.5:9b") -> dict:
    """
    Send messages and available MCP tools to the local Ollama model.
    """

    try:
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "think": False,
            "stream": False,
        }

        response = requests.post(
            url,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        return response.json()["message"]

    except requests.RequestException as exc:
        return {
            "role": "assistant",
            "content": f"LLM request failed: {exc}",
        }


# ---------------------------------------------------------
# AGENT LOOP
# ---------------------------------------------------------

async def run_agent(
    user_message: str,
    max_turns: int = 5,
) -> str:

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # Connect to MCP server
    async with client:

        # ---------------------------------------------
        # 1. Discover tools from MCP server
        # ---------------------------------------------

        mcp_tools = await client.list_tools()

        print("\nMCP tools discovered:")

        for tool in mcp_tools:
            print(f"- {tool.name}")

        # ---------------------------------------------
        # 2. Convert them for Ollama
        # ---------------------------------------------

        llm_tools = convert_mcp_tools_to_ollama(mcp_tools)

        # ---------------------------------------------
        # 3. Agent loop
        # ---------------------------------------------

        for turn in range(max_turns):

            response = ask_ai(
                messages,
                llm_tools,
            )

            messages.append(response)

            print(f"\nAI response — turn {turn + 1}:")
            pprint(response)

            tool_calls = response.get("tool_calls")

            # -----------------------------------------
            # No tool requested -> final answer
            # -----------------------------------------

            if not tool_calls:
                return response.get("content", "")

            # -----------------------------------------
            # LLM requested one or more tools
            # -----------------------------------------

            for tool_call in tool_calls:

                function = tool_call["function"]

                tool_name = function["name"]
                arguments = function["arguments"]

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                print(f"\nTool requested: {tool_name}")
                print("Arguments:")
                pprint(arguments)

                # -------------------------------------
                # 4. MCP executes the tool
                # -------------------------------------

                result = await client.call_tool(
                    tool_name,
                    arguments,
                )

                print("MCP result:")
                pprint(result)

                # FastMCP returns MCP content objects.
                # Convert their text into something
                # Ollama can receive.
                result_text = "\n".join(
                    content.text
                    for content in result.content
                    if hasattr(content, "text")
                )

                # -------------------------------------
                # 5. Give tool result back to LLM
                # -------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get(
                            "id",
                            tool_name,
                        ),
                        "content": result_text,
                    }
                )

    return "Agent stopped because it reached the maximum number of turns."


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

async def main():

    user_message = input("You: ")

    response = await run_agent(user_message)

    print("\nAgent:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())