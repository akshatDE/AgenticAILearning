"""MCP server implementations.

Servers live here, deliberately apart from the client infrastructure in
:mod:`agenticgpt.mcp`. The agent is an MCP *client*; these are the *servers*
it talks to. Keeping them in separate packages is what stops the two from
growing into each other.

A second server is a new subpackage plus one entry in
:mod:`agenticgpt.mcp.config` — no change to the agent, the manager, or the UI.
"""
