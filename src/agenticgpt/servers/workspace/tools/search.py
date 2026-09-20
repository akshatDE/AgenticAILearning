"""Web search, via the Tavily API.

Needs ``TAVILY_API_KEY``, inherited from the parent process.

Whatever this returns reaches the model as text, and the model may hand parts
of it to another tool. That is the reason ``calculator`` refuses to use
``eval()``.
"""

from __future__ import annotations

import os

import requests
from fastmcp import FastMCP

TAVILY_URL = "https://api.tavily.com/search"


def register(mcp: FastMCP) -> None:
    """Attach this tool to the server."""

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
                TAVILY_URL,
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
