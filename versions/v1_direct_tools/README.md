# Version 1 — Direct Tool Calling

The original AgenticGPT. Archived, unchanged in architecture, still runnable.

```text
User
 ↓
Pure Python Agent Loop
 ↓
LLM  (Ollama locally, or Groq in the cloud)
 ↓
Tool Decision
 ↓
Local Tool Registry   (TOOLS_BY_NAME in tools.py)
 ↓
Python Function
```

![V1 architecture](../../docs/architecture/v1-direct-tools.png)

## What this version demonstrates

The model never "calls" anything. It returns a **name and a JSON blob**. Everything
between that and a Python function running is code you write yourself:

- `TOOL_SCHEMAS` — the hand-written JSON the model reads to learn what exists
- `TOOLS_BY_NAME` — a dict mapping the returned name back to the function
- `run_agent()` — send → check for `tool_calls` → execute → append result → repeat

Two backends run against the same loop, which surfaces a detail a framework
would have hidden: **Ollama returns tool arguments as a dict, Groq returns them
as a JSON string**, and Ollama accepts a `tool_name` key on tool results while
Groq requires `tool_call_id`.

## Files

| File | Role |
|---|---|
| `agent.py` | the loop, plus `ask_ai` (Ollama) and `ask_groq` (Groq) |
| `tools.py` | the four tool functions, `TOOL_SCHEMAS`, `TOOLS_BY_NAME` |
| `app.py` | Streamlit UI with a backend switcher |

## Running it

```bash
cd versions/v1_direct_tools
pip install -r requirements.txt
```

Set keys in the repository-root `.env` (see `.env.example`), then either:

```bash
python agent.py
```

```bash
streamlit run app.py
```

`agent.py` uses the Groq backend by default, so `GROQ_API_KEY` must be set.
For the Ollama path, run `ollama serve` and swap `ask_groq` for `ask_ai`
inside `run_agent()`, or use the sidebar switcher in `app.py`.

## Known rough edges — deliberately left in place

- `calculator` uses raw `eval()`. Version 2 replaces it with an AST walker.
- Tool schemas are hand-written and drift from the functions they describe.
- No conversation memory: every call starts from an empty message list.

The only change made during the repository restructure was replacing a
hardcoded WeatherAPI key with `os.environ.get("WEATHER_API_KEY")`. The
architecture is untouched.
