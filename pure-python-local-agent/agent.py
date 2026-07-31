import json
from pprint import pprint
from urllib import response
from tools import get_weather, calculator, convert_currency, TOOL_SCHEMAS, TOOLS_BY_NAME
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def ask_ai(message: list, model: str = "qwen3.5:9b") -> str:
    """Send a prompt to a local Ollama model and return its response.

    Submit the user's message to the Ollama chat API and return the
    generated response. If the request fails, an error message is
    returned instead.
    """
    try:
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": model,
            "messages": message,
            "tools": TOOL_SCHEMAS,
            "think": False,
            "stream": False,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["message"]

    except requests.RequestException as exc:
        return f"An error occurred: {exc}"


def ask_groq(message: list, model: str = "openai/gpt-oss-120b") -> str:
    """Send a prompt to the Groq cloud API and return its response.

    Submit the user's message to the Groq chat completions API and return
    the generated response. The API key is read from the GROQ_API_KEY
    environment variable. If the request fails, an error message is
    returned instead.
    """
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        }

        payload = {
            "model": model,
            "messages": message,
            "tools": TOOL_SCHEMAS,
            "stream": False,
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]

    except requests.RequestException as exc:
        return f"An error occurred: {exc}"


def run_agent(user_message: str, max_turns: int = 10) -> str:

    """

    Execute the complete agent workflow.

    The agent sends the user's query to the LLM, executes any requested tools,
    feeds the tool results back to the model, and repeats until a final answer
    is produced or the maximum number of turns is reached.

    Args:

        user_message: User's input query.

        max_turns: Maximum number of agent iterations.

    Returns:

        The final AI response as a string.

    """
    try:

        messages = [

            {

                "role": "user",

                "content": user_message,

            }

        ]

        for turn in range(max_turns):

            response = ask_groq(messages)
            messages.append(response)
            print(f"\nAI response — turn {turn + 1}:")

            pprint(response)

            if not response.get("tool_calls"):
                return response["content"]

            for tool_call in response["tool_calls"]:

                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                if isinstance(arguments, str):

                    arguments = json.loads(arguments)

                tool_function = TOOLS_BY_NAME.get(tool_name)

                if tool_function is None:

                    result = f"Error: Tool '{tool_name}' was not found."

                else:

                    result = tool_function(**arguments)

                print(f"\nTool: {tool_name}")

                print("Arguments:")

                pprint(arguments)

                print("Result:")

                pprint(result)

                messages.append(

                    {

                        "role": "tool",

                        "tool_call_id": tool_call.get("id", tool_name),

                        "content": str(result),

                    }

                )

        return "Agent stopped because it reached the maximum number of turns."
    except Exception as e:
        return f"An error occurred during agent execution: {e}"

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    final_response = run_agent(user_input)
    print("\nFinal AI Response:")
    pprint(final_response)

