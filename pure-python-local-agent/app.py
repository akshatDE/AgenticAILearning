"""AgenticGPT — ChatGPT-style Streamlit frontend for the pure-python local agent.

Imports run_agent from agent.py as-is (no changes to agent.py / tools.py).
The agent prints its tool-calling trace to stdout, so we capture stdout
during the run and surface it in the UI as an expandable "agent trace".
"""

import io
import time
from contextlib import redirect_stdout

import requests
import streamlit as st

from agent import run_agent

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5:9b"

st.set_page_config(
    page_title="AgenticGPT",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    /* ---- overall canvas: ChatGPT dark ---- */
    .stApp { background: #212121; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 48rem; padding-top: 1.5rem; position: relative; z-index: 1; }

    /* ---- AgenticGPT branding watermark (background layer) ---- */
    .brand-watermark {
        position: fixed; top: 0; right: 0; bottom: 0; left: 0;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        gap: 1.1rem;
        z-index: 0; pointer-events: none;
        opacity: .09;
    }
    /* keep the watermark centred over the chat column: when the sidebar is
       expanded, start the watermark box after it (sidebar default = 300px) */
    [data-testid="stSidebar"][aria-expanded="true"] ~ div .brand-watermark {
        left: 300px;
    }
    .brand-watermark .wm-logo {
        width: 130px; height: 130px; border-radius: 50%;
        background: #10a37f; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 64px; font-weight: 800;
    }
    .brand-watermark .wm-name {
        font-size: 3.4rem; font-weight: 800; letter-spacing: .04em;
        color: #ececec;
    }
    .brand-watermark .wm-powered {
        display: flex; align-items: center; gap: 1.4rem;
        font-size: 1.5rem; font-weight: 700; color: #ececec;
    }
    .brand-watermark .wm-powered .x { font-weight: 400; color: #9a9a9a; }

    /* ---- visible powered-by badges ---- */
    .powered-row {
        display: flex; justify-content: center; align-items: center;
        gap: .8rem; margin: -0.6rem 0 1.6rem;
    }
    .powered-badge {
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .35rem .9rem; border-radius: 999px;
        border: 1px solid #3a3a3a; background: rgba(48,48,48,.6);
        font-size: .85rem; font-weight: 600; color: #d4d4d4;
    }
    .qwen-text {
        background: linear-gradient(90deg, #7c6bff, #b06bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #2f2f2f;
    }
    [data-testid="stSidebar"] * { color: #ececec; }

    /* ---- brand ---- */
    .brand {
        display: flex; align-items: center; gap: .6rem;
        font-size: 1.25rem; font-weight: 700; color: #ececec;
        padding: .25rem 0 .5rem;
    }
    .brand .logo {
        width: 30px; height: 30px; border-radius: 50%;
        background: #10a37f; color: #fff;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1rem; font-weight: 700;
    }

    /* ---- empty-state greeting ---- */
    .greeting {
        text-align: center;
        margin: 22vh 0 2rem;
        font-size: 1.9rem;
        font-weight: 600;
        color: #ececec;
    }

    /* ---- chat messages ---- */
    .user-row { display: flex; justify-content: flex-end; margin: .9rem 0; }
    .user-bubble {
        background: #303030; color: #ececec;
        padding: .7rem 1.15rem; border-radius: 1.3rem;
        max-width: 78%; line-height: 1.55; font-size: .98rem;
        word-wrap: break-word;
    }
    .assistant-logo {
        width: 28px; height: 28px; border-radius: 50%;
        background: #10a37f; color: #fff;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: .85rem; font-weight: 700;
        margin-top: .2rem;
    }

    /* ---- pill buttons (suggestions, sidebar) ---- */
    .stButton > button {
        background: transparent;
        border: 1px solid #4a4a4a;
        border-radius: 999px;
        color: #c9c9c9;
        font-size: .87rem;
        padding: .45rem 1rem;
    }
    .stButton > button:hover {
        background: #2f2f2f; color: #ececec; border-color: #5a5a5a;
    }

    /* ---- chat input: rounded ChatGPT-style composer ---- */
    [data-testid="stChatInput"] {
        background: #303030 !important;
        border: 1px solid #424242 !important;
        border-radius: 1.6rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #ececec !important;
    }
    [data-testid="stBottom"] > div { background: #212121; }
    [data-testid="stBottomBlockContainer"] { background: #212121; max-width: 48rem; }

    /* ---- expander (agent trace) ---- */
    [data-testid="stExpander"] {
        background: #262626;
        border: 1px solid #3a3a3a !important;
        border-radius: .8rem;
    }

    /* ---- status pill ---- */
    .status-pill {
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .3rem .8rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600;
        border: 1px solid #3a3a3a; background: #212121;
    }
    .dot { width: .5rem; height: .5rem; border-radius: 50%; }
    .dot.ok { background: #10a37f; box-shadow: 0 0 7px #10a37f; }
    .dot.bad { background: #ef4444; box-shadow: 0 0 7px #ef4444; }

    .tool-line {
        font-size: .85rem; color: #b4b4b4;
        padding: .35rem 0; border-bottom: 1px solid #262626;
    }
    .tool-line b { color: #ececec; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# simple geometric mark evoking Qwen's angular spark logo (inline SVG, no assets)
QWEN_MARK = """<svg width="{s}" height="{s}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="qg{i}" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#7c6bff"/><stop offset="1" stop-color="#b06bff"/>
</linearGradient></defs>
<path d="M50 4 L62 32 L92 32 L70 52 L80 84 L50 66 L20 84 L30 52 L8 32 L38 32 Z"
      fill="url(#qg{i})" opacity="0.95"/>
</svg>"""

# background watermark: AgenticGPT brand + Ollama x Qwen, fixed behind the chat
st.markdown(
    f"""
    <div class="brand-watermark">
        <div class="wm-logo">A</div>
        <div class="wm-name">AgenticGPT</div>
        <div class="wm-powered">
            <span>🦙 Ollama</span>
            <span class="x">×</span>
            <span style="display:inline-flex;align-items:center;gap:.5rem;">
                {QWEN_MARK.format(s=34, i="wm")} Qwen
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- helpers
TOOLS_UI = [
    ("🌤️", "get_weather", "live weather for any city"),
    ("🧮", "calculator", "arithmetic expressions"),
    ("💱", "convert_currency", "live FX rates"),
]

EXAMPLES = [
    "What's the weather in Berlin right now?",
    "Convert 250 USD to INR",
    "What is (145 * 32) + 78?",
]


@st.cache_data(ttl=10)
def ollama_status() -> tuple[bool, list[str]]:
    """Return (reachable, installed model names)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except requests.RequestException:
        return False, []


def run_agent_with_trace(prompt: str) -> tuple[str, str, float]:
    """Call run_agent unchanged, capturing its printed trace from stdout."""
    buffer = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buffer):
        answer = run_agent(prompt)
    elapsed = time.perf_counter() - start
    return answer, buffer.getvalue(), elapsed


def render_user(text: str) -> None:
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistant(text: str, trace: str = "", elapsed: float = 0.0) -> None:
    logo_col, body_col = st.columns([1, 14], gap="small")
    with logo_col:
        st.markdown('<div class="assistant-logo">A</div>', unsafe_allow_html=True)
    with body_col:
        st.markdown(text)
        if trace.strip():
            with st.expander(f"Agent trace · {elapsed:.1f}s"):
                st.code(trace, language="text")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="logo">A</span>AgenticGPT</div>',
        unsafe_allow_html=True,
    )
    st.caption("Pure-Python agent · runs 100% locally")
    st.markdown(
        f"""
        <div class="powered-row" style="justify-content:flex-start;margin:.2rem 0 .8rem;">
            <span class="powered-badge">🦙 Ollama</span>
            <span class="powered-badge">{QWEN_MARK.format(s=14, i="sb")}
                <span class="qwen-text">Qwen</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    st.divider()

    online, models = ollama_status()
    if online:
        st.markdown(
            '<span class="status-pill"><span class="dot ok"></span>Ollama online</span>',
            unsafe_allow_html=True,
        )
        if MODEL_NAME not in models:
            st.warning(f"Model `{MODEL_NAME}` not found. Run `ollama pull {MODEL_NAME}`.")
    else:
        st.markdown(
            '<span class="status-pill"><span class="dot bad"></span>Ollama offline</span>',
            unsafe_allow_html=True,
        )
        st.error("Start Ollama first: `ollama serve`")

    st.caption(f"Model: `{MODEL_NAME}`")

    st.divider()
    st.markdown("**Tools**")
    for icon, name, desc in TOOLS_UI:
        st.markdown(
            f'<div class="tool-line">{icon} <b>{name}</b> — {desc}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- main
if "chat" not in st.session_state:
    st.session_state.chat = []

# empty state: centered greeting + suggestion pills, like ChatGPT's home
if not st.session_state.chat:
    st.markdown('<div class="greeting">What can I help with?</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="powered-row">
            <span class="powered-badge">🦙 Ollama</span>
            <span style="color:#7a7a7a;">×</span>
            <span class="powered-badge">{QWEN_MARK.format(s=16, i="pb")}
                <span class="qwen-text">Qwen&nbsp;3.5</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

# replay history
for msg in st.session_state.chat:
    if msg["role"] == "user":
        render_user(msg["content"])
    else:
        render_assistant(msg["content"], msg.get("trace", ""), msg.get("elapsed", 0.0))

# input — either typed or from an example button
prompt = st.chat_input("Message AgenticGPT")
if prompt is None:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    render_user(prompt)

    with st.spinner("Thinking…"):
        answer, trace, elapsed = run_agent_with_trace(prompt)

    render_assistant(answer, trace, elapsed)
    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "trace": trace, "elapsed": elapsed}
    )
    st.rerun()
