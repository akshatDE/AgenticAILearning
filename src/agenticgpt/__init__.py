"""AgenticGPT — a tool-calling AI agent in pure Python.

No LangChain, LangGraph, CrewAI or AutoGen. The loop, the message list and the
tool plumbing are written out rather than imported, because the point is to
see them.

Four layers, each ignorant of the next:

.. code-block:: text

    apps/                 user interfaces (Streamlit, CLI)
        ↓
    agenticgpt.agent      reasoning: the loop, messages, the model
        ↓
    agenticgpt.mcp        MCP client: config, connections, registry, manager
        ↓
    agenticgpt.servers    MCP servers: tools, resources, prompts

The usual entry point is :func:`run_agent`, which wires all of it together::

    import asyncio
    from agenticgpt import run_agent

    print(asyncio.run(run_agent("What is 15% of 4,320?")))

For more control, build the pieces yourself::

    async with MCPManager() as manager:
        agent = AgentLoop(llm=OllamaClient(), toolbox=manager)
        answer = await agent.run("Convert 500 USD to INR")
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    AgentLoop,
    Conversation,
    GroqClient,
    LLMClient,
    OllamaClient,
)
from .mcp import MCPClient, MCPManager, MCPServerConfig, ToolSpec, load_servers

__version__ = "0.3.0"

__all__ = [
    "AgentLoop",
    "Conversation",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "GroqClient",
    "LLMClient",
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "OllamaClient",
    "ToolSpec",
    "__version__",
    "discover_tools",
    "load_environment",
    "load_servers",
    "main",
    "run_agent",
]

#: Repository root, when running from a checkout rather than an installed wheel.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Load API keys from a ``.env`` file into the environment.

    Called by the apps at start-up. Keys are read from the environment at the
    point of use, never baked into source, and the MCP server subprocess
    inherits them from whoever spawned it.

    Silently does nothing if ``python-dotenv`` is not installed or no file is
    found — the environment may well already be populated, as it is on
    Streamlit Cloud.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return

    path = Path(dotenv_path) if dotenv_path else REPO_ROOT / ".env"
    if not path.exists():
        return

    for name, value in dotenv_values(path).items():
        if value is not None and not os.environ.get(name):
            os.environ[name] = value


async def run_agent(
    user_message: str,
    model: str | None = None,
    history: list[dict[str, Any]] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    system_prompt: str | None = None,
    servers: Sequence[MCPServerConfig] | None = None,
    llm: LLMClient | None = None,
    on_event: Callable[[str], None] | None = None,
) -> str:
    """Answer one question, with MCP tools available.

    Connects to the configured MCP servers, discovers what they offer, runs
    the agent loop, and closes the sessions again.

    Args:
        user_message: The question.
        model: Model tag, e.g. ``"qwen3:8b"``. Defaults to the LLM client's own.
        history: Earlier messages, for multi-turn conversations.
        max_turns: How many times the model may be asked before giving up.
        system_prompt: Optional instruction placed ahead of the user turn.
        servers: MCP servers to connect to. Defaults to the configured ones.
        llm: Model provider. Defaults to a local Ollama client.
        on_event: Called with a human-readable line per step, for a trace.

    Returns:
        The final answer, or an explanation of why the loop stopped.

    A fresh MCP connection is opened per call rather than cached at import
    time. A client holds subprocess pipes bound to the event loop that opened
    them, and Streamlit runs ``asyncio.run()`` once per message — a cached
    client works for the first message and fails on the second.
    """
    async with MCPManager(servers=servers, on_event=on_event) as manager:
        agent = AgentLoop(
            llm=llm or OllamaClient(model=model or DEFAULT_MODEL),
            toolbox=manager,
            system_prompt=system_prompt,
            max_turns=max_turns,
            on_event=on_event,
        )
        return await agent.run(user_message, history=history, model=model)


async def discover_tools(
    servers: Sequence[MCPServerConfig] | None = None,
) -> list[ToolSpec]:
    """Connect to the configured MCP servers and report what they offer.

    For interfaces that want to show the available tools without running the
    agent. The list is discovered, never hardcoded — that is the whole point
    of version 2.
    """
    async with MCPManager(servers=servers) as manager:
        return manager.tools()


def main() -> None:
    """Entry point for the ``agenticgpt`` command: one question, one answer."""
    load_environment()
    model = os.environ.get("AGENTICGPT_MODEL") or DEFAULT_MODEL

    try:
        user_message = input("You: ")
    except (EOFError, KeyboardInterrupt):
        return

    if not user_message.strip():
        return

    answer = asyncio.run(run_agent(user_message, model=model, on_event=print))
    print("\nAgent:")
    print(answer)
