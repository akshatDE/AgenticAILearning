"""The workspace server: everyday tools for AgenticGPT.

- :mod:`~agenticgpt.servers.workspace.server` — builds and runs the server
- ``tools/`` — one module per tool
- ``resources/`` — readable things (none yet)
- ``prompts/`` — server-owned prompt templates (none yet)

``build_server`` is exported for tests, which exercise the server in-process
rather than over a subprocess.
"""

from .server import INSTRUCTIONS, build_server

__all__ = ["INSTRUCTIONS", "build_server"]
