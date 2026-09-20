"""Live exchange rates, via the Frankfurter API (free, no key)."""

from __future__ import annotations

import requests
from fastmcp import FastMCP

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"


def register(mcp: FastMCP) -> None:
    """Attach this tool to the server."""

    @mcp.tool
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert an amount between two currencies at today's rate.

        Currency codes are three letters, e.g. USD, INR.
        """
        try:
            response = requests.get(
                FRANKFURTER_URL,
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
        except (KeyError, ValueError):
            return (
                f"No rate available for {from_currency.upper()} -> "
                f"{to_currency.upper()}"
            )
