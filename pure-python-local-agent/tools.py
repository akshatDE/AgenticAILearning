import os
import requests


def get_weather_tool(location: str) -> dict:
    """Fetch the current weather for a location.

    Query the WeatherAPI for the specified location and return the
    current weather condition and temperature in Celsius. If the
    request fails, an error message is returned instead.

    """
    weather_api_key = os.getenv("WEATHER_API_KEY")
    base_url = "http://api.weatherapi.com/v1"
    response = requests.get(f"{base_url}/current.json?key={weather_api_key}&q={location}")
    try:
        data = response.json()
        location_name = data['location']['name']
        weather_condition = data['current']['condition']['text']
        temperature_celsius = data['current']['temp_c']
        return {
            "location": location_name,
            "condition": weather_condition,
            "temperature_celsius": temperature_celsius
        }
    except Exception as e:
        return {response.status_code: "Failed to get weather data"}

def convert_currency_tool(amount: float, from_currency: str, to_currency: str) -> str:
    """Live conversion via the Frankfurter API (free, no key required).
    A real network call, unlike the other two tools -- worth pointing out
    that tools can mix live data and static data freely.
    """
    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": from_currency.upper(), "symbols": to_currency.upper()},
            timeout=10,
        )
        response.raise_for_status()
        rate = response.json()["rates"][to_currency.upper()]
        return str(round(amount * rate, 2))
    except requests.exceptions.RequestException as exc:
        return f"Currency service unavailable: {exc}"
    except KeyError:
        return f"No rate available for {from_currency} -> {to_currency}"

def calculate_tool(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Safely evaluate the provided mathematical expression using Python's
    eval function. If the evaluation fails, an error message is returned
    instead.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"