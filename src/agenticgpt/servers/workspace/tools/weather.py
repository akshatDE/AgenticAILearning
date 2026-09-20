"""Current conditions, via WeatherAPI.

Needs ``WEATHER_API_KEY``. The server is spawned as a child process, so it
inherits that from whoever started the agent — nothing is hardcoded here.
"""

from __future__ import annotations

import os

import requests
from fastmcp import FastMCP

WEATHER_URL = "http://api.weatherapi.com/v1/current.json"


def register(mcp: FastMCP) -> None:
    """Attach this tool to the server."""

    @mcp.tool
    def get_weather(city: str) -> dict:
        """Get the current weather for a city.

        Returns the location name, a description of conditions, and the
        temperature in Celsius.
        """
        api_key = os.environ.get("WEATHER_API_KEY", "")
        if not api_key:
            return {"error": "WEATHER_API_KEY is not set in the environment."}

        try:
            response = requests.get(
                WEATHER_URL,
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
            return {
                "error": f"Weather service returned an unexpected response for {city}."
            }
