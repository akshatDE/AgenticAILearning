"""Talking to a model provider.

Everything provider-shaped lives here: the endpoint, the payload, the timeout,
and the translation from an MCP tool description into whatever JSON the
provider expects. The agent loop calls :meth:`LLMClient.chat` and reads
``content`` and ``tool_calls`` — nothing more.

Adding a provider means adding a class here. The loop does not change.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Protocol, Sequence, runtime_checkable

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"

#: Local models on CPU are slow; 60s times out mid-answer.
REQUEST_TIMEOUT = 120

#: Groq requests allowed per window. One question can cost several (one per
#: agent turn), so this caps total spend, not the number of questions.
GROQ_MAX_REQUESTS = 20
GROQ_WINDOW_SECONDS = 60


@runtime_checkable
class LLMClient(Protocol):
    """What the agent loop needs from a model provider."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Any] = (),
        model: str | None = None,
    ) -> dict[str, Any]:
        """Return one assistant message: ``content`` and maybe ``tool_calls``."""
        ...


def _tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Describe one tool the way OpenAI-style APIs expect.

    Accepts an :class:`~agenticgpt.mcp.registry.ToolSpec` or anything with the
    same three attributes, which keeps this function usable without importing
    the MCP layer.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "parameters": getattr(tool, "input_schema", None) or {},
        },
    }


def tools_to_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Translate discovered tools into provider tool definitions.

    Both providers below speak the OpenAI tool format, so one translation
    covers them. A provider that does not would override this.
    """
    return [_tool_to_openai_schema(tool) for tool in tools]


class OllamaClient:
    """A local model served by Ollama.

    Ollama is stateless per call: the full message list and every tool
    definition are re-sent on every turn. That is visible in the trace and is
    worth understanding before reaching for a framework that hides it.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_CHAT_URL,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Any] = (),
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "tools": tools_to_schemas(tools),
            "think": False,
            "stream": False,
        }

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]
        except requests.HTTPError as exc:
            return {"role": "assistant", "content": _http_detail("Ollama", exc)}
        except requests.RequestException as exc:
            return {"role": "assistant", "content": f"Could not reach Ollama: {exc}"}
        except (KeyError, ValueError):
            return {
                "role": "assistant",
                "content": "Ollama returned an unexpected response.",
            }


class GroqClient:
    """Groq's hosted OpenAI-compatible endpoint.

    Kept because a 9B local model cannot be hosted on free infrastructure,
    which is what made a live demo possible. It also exposes a difference a
    framework would have hidden: Groq returns tool arguments as a JSON
    *string* while Ollama returns a dict. The loop handles both.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    #: Shared by every instance: Streamlit builds a new client per message, so
    #: a per-instance counter would reset on each one and limit nothing.
    _sent: deque[float] = deque()
    _lock = threading.Lock()

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = REQUEST_TIMEOUT,
        max_requests: int = GROQ_MAX_REQUESTS,
        window_seconds: float = GROQ_WINDOW_SECONDS,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._api_key = api_key
        self.timeout = timeout
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def _acquire(self) -> float:
        """Record a request, or return how many seconds until one is allowed."""
        now = time.monotonic()
        with self._lock:
            sent = GroqClient._sent
            while sent and now - sent[0] >= self.window_seconds:
                sent.popleft()
            if len(sent) >= self.max_requests:
                return self.window_seconds - (now - sent[0])
            sent.append(now)
            return 0.0

    @property
    def api_key(self) -> str:
        """Read the key at call time so a late ``load_dotenv()`` still works."""
        return self._api_key or os.environ.get("GROQ_API_KEY", "")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Any] = (),
        model: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {
                "role": "assistant",
                "content": "GROQ_API_KEY is not set in the environment.",
            }

        wait = self._acquire()
        if wait:
            return {
                "role": "assistant",
                "content": (
                    f"Groq request limit reached ({self.max_requests} per "
                    f"{self.window_seconds:g}s). Try again in {wait:.0f}s."
                ),
            }

        payload = {
            "model": model or self.model,
            "messages": messages,
            "tools": tools_to_schemas(tools),
            "stream": False,
        }

        try:
            response = requests.post(
                self.URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]
        except requests.HTTPError as exc:
            return {"role": "assistant", "content": _http_detail("Groq", exc)}
        except requests.RequestException as exc:
            return {"role": "assistant", "content": f"Could not reach Groq: {exc}"}
        except (KeyError, IndexError, ValueError):
            return {
                "role": "assistant",
                "content": "Groq returned an unexpected response.",
            }


def _http_detail(provider: str, exc: requests.HTTPError) -> str:
    """Pull the useful part out of an error response body."""
    detail = ""
    if exc.response is not None:
        try:
            detail = exc.response.json().get("error", "")
            if isinstance(detail, dict):
                detail = detail.get("message", "")
        except ValueError:
            detail = exc.response.text[:200]
    return f"{provider} rejected the request: {detail or exc}"
