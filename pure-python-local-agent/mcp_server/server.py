"""
AgenticGPT MCP server.

Exposes four tools over MCP. Launched as a stdio subprocess by app.py,
so it inherits the parent process's environment — that is how the API
keys reach it.

Required environment variables:
    WEATHER_API_KEY   weatherapi.com
    TAVILY_API_KEY    tavily.com

Run standalone with:  python server.py
"""

from __future__ import annotations

import ast
import operator
import os

import requests
from fastmcp import FastMCP

mcp = FastMCP(
    "AgenticGPT Testing Server",
    "A testing server for AgenticGPT tools.",
)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
@mcp.tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between two currencies at today's rate.

    Uses the Frankfurter API. Currency codes are three letters, e.g. USD, INR.
    """
    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={
                "base": from_currency.upper(),
                "symbols": to_currency.upper(),
            },
            timeout=10,
        )
        response.raise_for_status()
        rate = response.json()["rates"][to_currency.upper()]
        return f"{round(amount * rate, 2)} {to_currency.upper()}"
    except requests.exceptions.RequestException as exc:
        return f"Currency service unavailable: {exc}"
    except KeyError:
        return f"No rate available for {from_currency.upper()} -> {to_currency.upper()}"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
@mcp.tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Returns the location name, a description of conditions, and the
    temperature in Celsius.
    """
    api_key = "e3897ae36fa745dd94c34330263007"
    if not api_key:
        return {"error": "WEATHER_API_KEY is not set in the environment."}

    try:
        response = requests.get(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": api_key, "q": city},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "location": data["location"]["name"],
            "condition": data["current"]["condition"]["text"],
            "temperature_celsius": data["current"]["temp_c"],
        }
    except requests.exceptions.RequestException as exc:
        return {"error": f"Weather service unavailable: {exc}"}
    except (KeyError, ValueError):
        return {"error": f"Weather service returned an unexpected response for {city}."}


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
@mcp.tool
def search_internet(query: str, count: int = 5) -> str:
    """Search the web and return the top results as a numbered list.

    Each result shows its title, a short snippet, and the source URL.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "TAVILY_API_KEY is not set in the environment."

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": count},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()["results"]

        if not results:
            return f"No search results found for {query}"

        return "\n\n".join(
            f"{position}. {item['title']}\n"
            f"{item.get('content', '')}\n"
            f"{item['url']}"
            for position, item in enumerate(results, start=1)
        )
    except requests.exceptions.RequestException as exc:
        return f"Search service unavailable: {exc}"
    except (KeyError, ValueError):
        return f"Search returned an unexpected response for {query}"


# ---------------------------------------------------------------------------
# Calculator
#
# The expression arrives from the model, which is in turn influenced by user
# input and by web pages fetched through search_internet. eval() would make
# that path arbitrary code execution, so the expression is parsed and walked
# instead: only numbers and arithmetic operators are permitted.
# ---------------------------------------------------------------------------
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPONENT = 1_000  # keeps 9**9**9 from freezing the server


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if op is operator.pow and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent is too large")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        return op(_evaluate(node.operand))

    raise ValueError("unsupported expression")


@mcp.tool
def calculator(expression: str) -> str:
    """Work out an arithmetic expression, e.g. (145 * 32) + 78.

    Supports + - * / // % ** and parentheses. Numbers only — no variables,
    function calls, or names.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except ZeroDivisionError:
        return "Cannot divide by zero."
    except (SyntaxError, ValueError) as exc:
        return f"Could not work that out: {exc}"
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the model
        return f"Could not work that out: {exc}"

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


if __name__ == "__main__":
    mcp.run()