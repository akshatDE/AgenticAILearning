from fastmcp import FastMCP
import requests
import os

mcp = FastMCP("AgenticGPT Testing Server", "A testing server for AgenticGPT tools.")

@mcp.tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
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
    
@mcp.tool
def get_weather(city: str) -> dict:
    """Fetch the current weather for a location.

    Query the WeatherAPI for the specified location and return the
    current weather condition and temperature in Celsius. If the
    request fails, an error message is returned instead.

    """
    try:
        
        base_url = "http://api.weatherapi.com/v1"
        response = requests.get(f"{base_url}/current.json?key=e3897ae36fa745dd94c34330263007&q={city}")
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
    
@mcp.tool
def search_internet(query: str, count: int = 5) -> str:
    """Search the web using the Tavily Search API.

    Send the query to Tavily and return the top results as a numbered list
    of titles, snippets and links. The API key is read from the
    TAVILY_API_KEY environment variable. If the request fails, an error
    message is returned instead.
    """
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {os.environ.get('TAVILY_API_KEY', '')}",
            },
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
    except KeyError:
        return f"Search returned an unexpected response for {query}"

@mcp.tool
def calculator(expression: str) -> str:
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
    
if __name__ == "__main__":
    mcp.run()