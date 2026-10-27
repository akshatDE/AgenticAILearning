"""Agent reasoning and runtime.

- :mod:`~agenticgpt.agent.loop` — choose, execute, observe, repeat
- :mod:`~agenticgpt.agent.messages` — the message list, the agent's memory
- :mod:`~agenticgpt.agent.llm` — provider-specific request/response handling

Nothing in here imports :mod:`agenticgpt.mcp`. The loop receives a toolbox and
uses it; it does not know that tools arrive over MCP, or from where.
"""

from .llm import DEFAULT_MODEL, GroqClient, LLMClient, OllamaClient, tools_to_schemas
from .loop import DEFAULT_MAX_TURNS, DEFAULT_SYSTEM_PROMPT, AgentLoop, Toolbox, parse_arguments
from .messages import (
    Conversation,
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)

__all__ = [
    "AgentLoop",
    "Conversation",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MODEL",
    "GroqClient",
    "LLMClient",
    "Message",
    "OllamaClient",
    "Toolbox",
    "assistant_message",
    "parse_arguments",
    "system_message",
    "tool_message",
    "tools_to_schemas",
    "user_message",
]
