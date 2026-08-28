"""
AgenticGPT — ChatGPT-style Streamlit frontend for the MCP-powered agent.

    Streamlit  ->  run_agent()  ->  Ollama (Qwen)
                        |
                        +-------->  MCP client -> server.py -> tools

run_agent() prints its MCP discovery / tool-calling trace to stdout.
That stdout is captured per turn and shown in a collapsed "Agent trace".
"""

from __future__ import annotations

import asyncio
import html
import inspect
import io
import os
import time
from contextlib import redirect_stdout

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Secrets -> environment, so the spawned MCP server subprocess inherits them.
# ---------------------------------------------------------------------------
for _name in ("TAVILY_API_KEY", "WEATHER_API_KEY"):
    try:
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = st.secrets[_name]
    except Exception:
        pass

from mcp_server.app import run_agent  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
STREAM_DELAY = 0.012  # seconds per token when replaying the answer

st.set_page_config(
    page_title="AgenticGPT",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — ChatGPT dark palette:
#   canvas #212121 · sidebar #181818 · user bubble #303030
#   primary text #ececec · secondary #b4b4b4 · accent #10a37f
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --bg:#212121; --bg-side:#181818; --bubble:#303030;
      --fg:#ececec; --fg-dim:#b4b4b4; --fg-faint:#8f8f8f;
      --line:#2f2f2f; --accent:#10a37f;
      --font: "Söhne", ui-sans-serif, -apple-system, "Segoe UI", Helvetica,
              "Helvetica Neue", Arial, "Apple Color Emoji", sans-serif;
    }

    .stApp { background: var(--bg); }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="css"], .stMarkdown, .stButton > button { font-family: var(--font); }

    .block-container {
      max-width: 46rem;
      padding-top: 2.2rem;
      padding-bottom: 7rem;
    }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
      background: var(--bg-side);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--fg); }
    [data-testid="stSidebar"] hr { border-color: var(--line); margin: .9rem 0; }
    .side-label {
      font-size: .72rem; font-weight: 600; letter-spacing: .06em;
      text-transform: uppercase; color: var(--fg-faint); margin: .2rem 0 .5rem;
    }

    /* ---------- brand ---------- */
    .brand {
      display:flex; align-items:center; gap:.6rem;
      font-size:1.05rem; font-weight:600; color:var(--fg); padding:.1rem 0 .1rem;
    }
    .mark {
      width:26px; height:26px; border-radius:50%;
      background:var(--accent); color:#fff;
      display:inline-flex; align-items:center; justify-content:center;
      font-size:.8rem; font-weight:700; flex:0 0 auto;
    }

    /* ---------- empty state ---------- */
    .greeting {
      text-align:center; margin: 16vh 0 .4rem;
      font-size:1.75rem; font-weight:600; color:var(--fg); letter-spacing:-.01em;
    }
    .subgreeting {
      text-align:center; color:var(--fg-faint);
      font-size:.9rem; margin-bottom:1.8rem;
    }

    /* ---------- messages ---------- */
    .user-row { display:flex; justify-content:flex-end; margin:1.4rem 0 .2rem; }
    .user-bubble {
      background:var(--bubble); color:var(--fg);
      padding:.65rem 1.05rem; border-radius:1.35rem;
      max-width:75%; line-height:1.6; font-size:.97rem;
      white-space:pre-wrap; overflow-wrap:anywhere;
    }
    .assistant-head { margin:1.5rem 0 .35rem; }
    .assistant-body p, .assistant-body li { color:var(--fg); line-height:1.72; font-size:.97rem; }

    /* ---------- meta row under an answer ---------- */
    .meta { color:var(--fg-faint); font-size:.75rem; margin:.2rem 0 .1rem; }

    /* ---------- example prompt cards ---------- */
    .stButton > button {
      background:transparent; border:1px solid #3f3f3f; border-radius:.85rem;
      color:var(--fg-dim); font-size:.86rem; font-weight:400;
      padding:.7rem .95rem; text-align:left; line-height:1.4;
      transition: background .12s ease, border-color .12s ease;
    }
    .stButton > button:hover { background:#2a2a2a; border-color:#565656; color:var(--fg); }
    .stButton > button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
    [data-testid="stSidebar"] .stButton > button { border-radius:.6rem; text-align:center; }

    /* ---------- chat input ---------- */
    [data-testid="stChatInput"] {
      background:var(--bubble) !important;
      border:1px solid #444 !important;
      border-radius:1.65rem !important;
    }
    [data-testid="stChatInput"] textarea { background:transparent !important; color:var(--fg) !important; }
    [data-testid="stChatInput"] textarea::placeholder { color:#8a8a8a !important; }
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background:var(--bg); }
    [data-testid="stBottomBlockContainer"] { max-width:46rem; }

    /* ---------- trace ---------- */
    [data-testid="stExpander"] {
      background:#1c1c1c; border:1px solid var(--line) !important; border-radius:.7rem;
    }
    [data-testid="stExpander"] summary p { font-size:.8rem !important; color:var(--fg-faint) !important; }

    /* ---------- status ---------- */
    .pill {
      display:inline-flex; align-items:center; gap:.45rem;
      padding:.28rem .7rem; border-radius:999px; font-size:.78rem; font-weight:500;
      border:1px solid var(--line); background:#202020; color:var(--fg-dim);
    }
    .dot { width:.45rem; height:.45rem; border-radius:50%; flex:0 0 auto; }
    .dot.ok  { background:var(--accent); box-shadow:0 0 6px var(--accent); }
    .dot.bad { background:#ef4444;       box-shadow:0 0 6px #ef4444; }

    /* ---------- tools ---------- */
    .tool {
      display:flex; gap:.5rem; align-items:baseline;
      font-size:.83rem; color:var(--fg-dim); padding:.32rem 0;
    }
    .tool b { color:var(--fg); font-weight:500; }
    .tool span.desc { color:var(--fg-faint); font-size:.78rem; }

    @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
    @media (max-width: 640px) {
      .greeting { margin-top:12vh; font-size:1.45rem; }
      .user-bubble { max-width:88%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TOOLS_UI = [
    ("🌤️", "get_weather", "current conditions for a city"),
    ("💱", "convert_currency", "live FX rates"),
    ("🧮", "calculator", "arithmetic expressions"),
    ("🔎", "search_internet", "web search via Tavily"),
]

EXAMPLES = [
    "What's the weather in Berlin right now?",
    "Convert 250 USD to INR",
    "What is (145 * 32) + 78?",
    "Search for the latest AI news",
]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def ollama_models() -> tuple[bool, list[str]]:
    """Return (reachable, installed model tags)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        return True, [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return False, []


_ACCEPTS_MODEL = "model" in inspect.signature(run_agent).parameters


def run_agent_with_trace(prompt: str, model: str) -> tuple[str, str, float]:
    """Run the async agent from sync Streamlit, capturing its stdout trace."""
    buffer = io.StringIO()
    start = time.perf_counter()
    kwargs = {"model": model} if _ACCEPTS_MODEL else {}
    try:
        with redirect_stdout(buffer):
            answer = asyncio.run(run_agent(prompt, **kwargs))
    except Exception as exc:
        answer = f"The agent stopped before answering.\n\n`{type(exc).__name__}: {exc}`"
    return answer, buffer.getvalue(), time.perf_counter() - start


def stream_tokens(text: str):
    """Replay a finished answer word by word so it lands like a live stream."""
    for token in text.split(" "):
        yield token + " "
        time.sleep(STREAM_DELAY)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_user(text: str) -> None:
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def assistant_shell():
    """Draw the assistant avatar and return an empty slot for the answer body."""
    st.markdown(
        '<div class="assistant-head"><span class="mark">A</span></div>',
        unsafe_allow_html=True,
    )
    return st.container()


def render_meta(trace: str, elapsed: float, answer: str, key: str) -> None:
    left, right = st.columns([1, 1])
    with left:
        if trace.strip():
            with st.expander(f"Agent trace · {elapsed:.1f}s"):
                st.code(trace, language="text")
        else:
            st.markdown(f'<div class="meta">{elapsed:.1f}s</div>', unsafe_allow_html=True)
    with right:
        if hasattr(st, "popover"):
            with st.popover("Copy answer", use_container_width=False):
                st.code(answer, language="markdown")


def render_assistant(text: str, trace: str, elapsed: float, key: str) -> None:
    body = assistant_shell()
    with body:
        st.markdown(text)
    render_meta(trace, elapsed, text, key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
online, installed = ollama_models()

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="mark">A</span>AgenticGPT</div>',
        unsafe_allow_html=True,
    )
    st.caption("Local agent over MCP tools")

    if st.button("New chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    st.divider()

    st.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="pill"><span class="dot {"ok" if online else "bad"}"></span>'
        f'Ollama {"connected" if online else "not running"}</span>',
        unsafe_allow_html=True,
    )

    if not online:
        st.error("Start Ollama with `ollama serve`, then reload.")
        model = DEFAULT_MODEL
    elif not installed:
        st.warning("No models installed. Pull one with `ollama pull qwen3:8b`.")
        model = DEFAULT_MODEL
    else:
        default_index = installed.index(DEFAULT_MODEL) if DEFAULT_MODEL in installed else 0
        model = st.selectbox("Model", installed, index=default_index, label_visibility="collapsed")
        if not _ACCEPTS_MODEL:
            st.caption("Picker is display-only until `run_agent` accepts a `model` argument.")

    st.divider()

    st.markdown('<div class="side-label">Tools</div>', unsafe_allow_html=True)
    for icon, name, desc in TOOLS_UI:
        st.markdown(
            f'<div class="tool">{icon}<b>{name}</b><span class="desc">{desc}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Discovered from the MCP server at run time.")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
st.session_state.setdefault("chat", [])

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.chat:
    st.markdown('<div class="greeting">What can I help with?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subgreeting">Runs locally on Ollama. Tools execute through MCP.</div>',
        unsafe_allow_html=True,
    )

    rows = [EXAMPLES[:2], EXAMPLES[2:]]
    for row in rows:
        cols = st.columns(2, gap="small")
        for col, example in zip(cols, row):
            if col.button(example, use_container_width=True, key=f"ex_{example}"):
                st.session_state.chat.append({"role": "user", "content": example})
                st.rerun()

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.chat):
    if msg["role"] == "user":
        render_user(msg["content"])
    else:
        render_assistant(msg["content"], msg.get("trace", ""), msg.get("elapsed", 0.0), f"h{i}")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Message AgenticGPT"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

# ---------------------------------------------------------------------------
# Answer a pending user turn (last message has no reply yet)
# ---------------------------------------------------------------------------
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    pending = st.session_state.chat[-1]["content"]

    body = assistant_shell()
    with body:
        with st.spinner("Working through it…"):
            answer, trace, elapsed = run_agent_with_trace(pending, model)
        st.write_stream(stream_tokens(answer))

    render_meta(trace, elapsed, answer, "live")

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "trace": trace, "elapsed": elapsed}
    )