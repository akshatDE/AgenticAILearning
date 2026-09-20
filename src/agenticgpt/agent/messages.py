"""The message list — an agent's entire memory.

There is no hidden state anywhere in this project. What the model knows is
exactly what is in this list, and it is rebuilt and re-sent on every turn.

Keeping it here rather than inside the loop means the loop reads as
"decide, execute, observe" instead of dictionary-shuffling.
"""

from __future__ import annotations

from typing import Any, Iterator

Message = dict[str, Any]


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: str, tool_calls: list[dict] | None = None) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def tool_message(content: str, tool_call_id: str, name: str | None = None) -> Message:
    """Build the reply to one tool call.

    ``tool_call_id`` ties the result back to the request. Providers differ
    here — Ollama also accepts a ``tool_name`` key, Groq rejects it and
    requires ``tool_call_id`` — which is why the id is not optional.
    """
    message: Message = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    if name:
        message["name"] = name
    return message


class Conversation:
    """An ordered message list with typed ways to append to it.

    Usage::

        conversation = Conversation(system_prompt="You are terse.")
        conversation.add_user("What is 2 + 2?")
        conversation.add_assistant(reply)
        conversation.add_tool_result("4", tool_call_id="call_1", name="calculator")
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        history: list[Message] | None = None,
    ) -> None:
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(system_message(system_prompt))
        if history:
            self._messages.extend(history)

    # -- appending ----------------------------------------------------------
    def add_user(self, content: str) -> Message:
        return self._append(user_message(content))

    def add_assistant(self, message: Message) -> Message:
        """Append a provider reply as-is.

        The raw dict is kept rather than rebuilt: it carries the provider's
        own ``tool_calls`` structure, and the next request has to echo that
        back unchanged or the ids stop matching.
        """
        return self._append(message)

    def add_tool_result(
        self,
        content: str,
        tool_call_id: str,
        name: str | None = None,
    ) -> Message:
        return self._append(tool_message(content, tool_call_id, name))

    def _append(self, message: Message) -> Message:
        self._messages.append(message)
        return message

    # -- reading ------------------------------------------------------------
    def to_list(self) -> list[Message]:
        """A copy of the messages, ready to send to a provider."""
        return list(self._messages)

    @property
    def last(self) -> Message | None:
        return self._messages[-1] if self._messages else None

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __repr__(self) -> str:
        roles = ", ".join(message["role"] for message in self._messages)
        return f"<Conversation [{roles}]>"
