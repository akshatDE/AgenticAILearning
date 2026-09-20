"""The message list is the agent's whole memory, so its shape matters."""

from agenticgpt.agent.messages import Conversation, tool_message


def test_system_prompt_comes_first():
    conversation = Conversation(system_prompt="Be terse.")
    conversation.add_user("hello")

    roles = [message["role"] for message in conversation]
    assert roles == ["system", "user"]


def test_no_system_message_when_no_prompt():
    conversation = Conversation()
    conversation.add_user("hello")

    assert [message["role"] for message in conversation] == ["user"]


def test_assistant_reply_is_stored_verbatim():
    # The provider's own tool_calls structure has to survive the round trip,
    # or the ids stop matching on the next request.
    reply = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
    conversation = Conversation()
    conversation.add_assistant(reply)

    assert conversation.last == reply


def test_tool_result_carries_the_call_id():
    message = tool_message("4", tool_call_id="call_1", name="calculator")

    assert message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "4",
        "name": "calculator",
    }


def test_to_list_is_a_copy():
    conversation = Conversation()
    conversation.add_user("hello")

    snapshot = conversation.to_list()
    conversation.add_user("again")

    assert len(snapshot) == 1
    assert len(conversation) == 2


def test_history_is_continued():
    conversation = Conversation(history=[{"role": "user", "content": "earlier"}])
    conversation.add_user("now")

    assert [m["content"] for m in conversation] == ["earlier", "now"]
