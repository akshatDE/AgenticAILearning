import requests


def ask_ai(message: str, model: str = "qwen3.5:9b") -> str:
    """Send a prompt to a local Ollama model and return its response.

    Submit the user's message to the Ollama chat API and return the
    generated response. If the request fails, an error message is
    returned instead.
    """
    try:
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "think": False,
            "stream": False,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["message"]["content"]

    except requests.RequestException as exc:
        return f"An error occurred: {exc}"