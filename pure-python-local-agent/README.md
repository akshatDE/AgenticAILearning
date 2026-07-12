# Pure Python Agentic AI

A simple Agentic AI project built using **pure Python**, without frameworks such as LangChain, LangGraph, CrewAI, or AutoGen.

The goal of this project is to understand how an AI agent works internally before using agent frameworks.

The agent can:

- Understand a user request
- Decide whether a tool is required
- Select and call the correct Python function
- Read the tool result
- Continue calling tools when needed
- Return a final answer

---

## Arch Daigram
![python_agent_arch](python_agent_arch.png)

---
## Agent Workflow

```text
User Question
     ↓
Local LLM receives available tool schemas
     ↓
LLM selects a tool and generates its arguments
     ↓
Python executes the tool
     ↓
Tool result is added to conversation memory
     ↓
LLM returns an answer or calls another tool
```

The main agent cycle is:

```text
Choose → Execute → Observe → Repeat
```

---

## Local Model

The project will use **Qwen3.5 9B**, running locally on my Mac.

The model can be served through a local OpenAI-compatible endpoint using tools such as Ollama.

Example configuration:

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

model = "qwen3.5:9b"
```

This keeps the project local and avoids sending conversations or tool results to an external LLM provider.

---

## Available Tools

The Streamlit version currently includes:

| Tool | Purpose |
|---|---|
| `get_weather` | Returns sample weather data |
| `calculator` | Evaluates basic arithmetic |
| `convert_currency` | Converts currencies using a live API |

Tools are regular Python functions.

```python
def get_weather(city: str) -> str:
    data = SAMPLE_WEATHER.get(city.lower())

    if data is None:
        return f"No weather data for {city!r}."

    return f"{city.title()}: {data['celsius']}C, {data['conditions']}"
```

Each function also has a JSON schema that tells the model:

- Tool name
- Tool purpose
- Required arguments
- Argument data types

---

## Project Structure

```text
pure-python-agent/
│
├── agent.py
├── streamlit_app.py
├── tools.py
├── .env
├── pyproject.toml
└── README.md
```

### `agent.py`

Contains the core agent loop and terminal chat interface.

### `streamlit_app.py`

Provides the Streamlit chat interface and stores conversation history using `st.session_state`.

### `tools.py`

Contains the Python tool functions, tool schemas, and tool registry.

---

## Installation

```bash
git clone https://github.com/your-username/pure-python-agent.git
cd pure-python-agent
```

Install the dependencies:

```bash
uv add openai streamlit requests python-dotenv
```

Install and start Ollama, then download the local model:

```bash
ollama pull qwen3.5:9b
```

Run the model:

```bash
ollama serve
```

---

## Run the Terminal Agent

```bash
uv run python agent.py
```

---

## Run the Streamlit App

```bash
uv run streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal.

---

## Example Prompts

```text
What is the weather in Tokyo?
```

```text
Calculate (250 * 12) / 5.
```

```text
Convert 100 USD to INR.
```

```text
Convert 500 USD to INR and divide the result by 4.
```

The last example may require multiple tool calls.

---

## Core Components

The project contains five main parts:

1. **Brain** — Qwen3.5 9B running locally
2. **Tools** — Python functions that perform actions
3. **Tool schemas** — descriptions of tools provided to the model
4. **Memory** — the conversation and tool results stored in a message list
5. **Agent loop** — repeatedly connects the model and tools until the task is complete

---

## Why This Is Agentic AI

A normal chatbot follows:

```text
User Question → Model Answer
```

This project follows:

```text
User Question
     ↓
Model selects an action
     ↓
Python executes the action
     ↓
Model observes the result
     ↓
Model decides the next step
     ↓
Final Answer
```

The model is not only generating text. It is deciding which actions should be performed to complete the task.

---

## Current Limitations

- Weather data is static
- Conversation memory is not persistent
- Tool execution is synchronous
- Tool arguments need stronger validation
- The calculator uses a restricted version of `eval()`
- No logging, retry handling, or token tracking
- Tool-calling quality depends on the local model

---

## Future Improvements

- Add Pydantic tool validation
- Replace `eval()` with a safe expression parser
- Add SQLite or PostgreSQL conversation memory
- Add logging and tool execution tracking
- Add file-reading and SQL tools
- Build a Data Engineering agent
- Add code review and pipeline validation tools

---

## Learning Objective

This project is designed to understand what agent frameworks do internally:

```text
Local Model + Tools + Memory + Agent Loop
```

The focus is not on building the most advanced agent. The focus is on learning the core mechanics using simple Python code and a locally running model.

## Author

**Akshat Sharma**

Learning and building projects in:

- Data Engineering
- Python
- Local LLMs
- Agentic AI
