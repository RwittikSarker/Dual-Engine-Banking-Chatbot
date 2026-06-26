"""
app.py
------
Streamlit front-end for the Dual-Engine Banking Chatbot.

Launch:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bug fix: always resolve paths relative to repo root, not CWD
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.utils import load_config

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Banking Assistant | Dual-Engine Chatbot",
    page_icon="🏦",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #1a3a6e, #1f5fb5);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 8px 0 8px 60px;
        box-shadow: 0 2px 8px rgba(31, 95, 181, 0.25);
        line-height: 1.55;
    }
    .bot-msg {
        background: #1a1a2e;
        color: #e8eaf6;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 60px 8px 0;
        border-left: 4px solid #1f5fb5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        line-height: 1.55;
    }
    .msg-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 4px;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 12px 0;
    }
    .metric-card {
        flex: 1;
        background: #0f1535;
        border: 1px solid #1e2d5e;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #ccd6f6;
    }
    .metric-badge {
        font-size: 11px;
        border-radius: 999px;
        padding: 2px 8px;
        display: inline-block;
        margin-top: 4px;
        font-weight: 600;
    }
    .badge-high   { background: #1a4731; color: #4ade80; }
    .badge-medium { background: #3d2f00; color: #fbbf24; }
    .badge-low    { background: #3d0f0f; color: #f87171; }

    /* Escalation */
    .escalate-yes {
        background: #1a1a2e;
        border-left: 5px solid #fbbf24;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .escalate-no {
        background: #1a1a2e;
        border-left: 5px solid #4ade80;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }

    /* Input area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #1e2d5e;
        background: #0f1535;
        color: #e8eaf6;
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea:focus {
        border-color: #1f5fb5;
        box-shadow: 0 0 0 3px rgba(31,95,181,0.15);
    }

    /* Chat container scroll */
    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load pipeline (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Loading models… this may take a minute on first run.")
def load_pipeline():
    from src.predict import DualEnginePipeline
    # Bug fix: use absolute path relative to repo root
    cfg = load_config(str(_REPO_ROOT / "configs" / "config.yaml"))
    return DualEnginePipeline(cfg)


# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, result}
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "show_warning" not in st.session_state:
    st.session_state.show_warning = False

# Callback handlers
def handle_submit():
    q = st.session_state.input_text.strip()
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.pending_query = q
        st.session_state.show_warning = False
    else:
        st.session_state.show_warning = True
    st.session_state.input_text = ""

def handle_clear():
    st.session_state.messages = []
    st.session_state.input_text = ""
    st.session_state.pending_query = ""
    st.session_state.show_warning = False

def select_example(ex_text):
    st.session_state.input_text = ex_text

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("🏦 Dual-Engine Banking Chatbot")
st.markdown(
    "Powered by **DistilBERT** (intent classification) + "
    "**MiniLM + FAISS + Flan-T5** (retrieval-augmented generation)."
)
st.divider()

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────

if st.session_state.messages:
    with st.container():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">'
                    f'<div class="msg-label">You</div>'
                    f'{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                result = msg.get("result", {})
                st.markdown(
                    f'<div class="bot-msg">'
                    f'<div class="msg-label">Banking Assistant</div>'
                    f'{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Metrics row
                if result:
                    conf = result.get("confidence_score", 0)
                    sim = result.get("retrieval_similarity", 0)
                    intent = result.get("detected_intent", "unknown").replace("_", " ").title()

                    def _badge(val, high=0.8, med=0.6):
                        if val >= high:
                            return "high", "badge-high"
                        elif val >= med:
                            return "medium", "badge-medium"
                        return "low", "badge-low"

                    conf_label, conf_cls = _badge(conf)
                    sim_label, sim_cls = _badge(sim, 0.8, 0.5)

                    st.markdown(
                        f"""
                        <div class="metric-row">
                          <div class="metric-card">
                            <div class="metric-label">🎯 Intent</div>
                            <div class="metric-value" style="font-size:14px;">{intent}</div>
                          </div>
                          <div class="metric-card">
                            <div class="metric-label">🧠 Confidence</div>
                            <div class="metric-value">{conf:.1%}</div>
                            <span class="metric-badge {conf_cls}">{conf_label}</span>
                          </div>
                          <div class="metric-card">
                            <div class="metric-label">🔍 Similarity</div>
                            <div class="metric-value">{sim:.1%}</div>
                            <span class="metric-badge {sim_cls}">{sim_label}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    esc = result.get("escalation_recommendation", "No")
                    if esc == "Yes":
                        reasons = ", ".join(result.get("escalation_reasons", []))
                        st.markdown(
                            f'<div class="escalate-yes">⚠️ <strong>Escalation Recommended</strong>'
                            f'<br><small>{reasons}</small></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="escalate-no">✅ <strong>No Escalation Needed</strong>'
                            " — System is confident in this response.</div>",
                            unsafe_allow_html=True,
                        )

                    # Show raw Flan-T5 output in expander for academic transparency
                    rag_answer = result.get("rag_answer", "")
                    if rag_answer:
                        with st.expander("🔬 RAG Engine (Flan-T5) raw output"):
                            st.caption("This is the raw generation from Flan-T5 small (80M params), shown for academic transparency. The final answer above uses intent-matched templates for reliability.")
                            st.write(rag_answer)


    st.divider()

# ──────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────

# Warn user if they tried to submit empty input
if st.session_state.show_warning:
    st.warning("Please enter a query before submitting.")
    st.session_state.show_warning = False

# Bug fix: bind text_area to session_state key so example buttons pre-fill it
query = st.text_area(
    "📝 Your banking question:",
    placeholder="e.g. My card was charged twice for the same transaction.",
    height=100,
    key="input_text",   # bound to st.session_state.input_text
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    st.button("🔍 Get Answer", type="primary", use_container_width=True, on_click=handle_submit)
with col_clear:
    st.button("🗑️ Clear Chat", use_container_width=False, on_click=handle_clear)

# ──────────────────────────────────────────────
# Pipeline & output
# ──────────────────────────────────────────────

if st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = ""  # Clear it immediately to avoid processing twice on next rerun

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        st.error(
            f"**Models not found:** {exc}\n\n"
            "Please run the following commands first:\n"
            "```bash\n"
            "python src/data_download.py\n"
            "python src/train.py --config configs/config.yaml\n"
            "python src/retrieval.py --build\n"
            "```"
        )
        st.stop()

    with st.spinner("🤖 Analysing your query…"):
        result = pipeline.predict(query_to_process)

    # Add bot reply to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["generated_answer"],
        "result": result,
    })
    st.rerun()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This chatbot uses two engines:

        **Engine 1 — Intent Classifier**
        - Model: `distilbert-base-uncased`
        - Fine-tuned on Banking77 (77 intents)

        **Engine 2 — RAG Generator**
        - Embeddings: `all-MiniLM-L6-v2`
        - Vector DB: FAISS (inner-product)
        - Generator: `flan-t5-small`

        **Escalation Engine**
        - Flags low-confidence queries
        - Flags out-of-domain queries
        - Flags weak generated answers
        """
    )
    st.divider()

    st.header("💬 Example Queries")
    examples = [
        "My card was charged twice",
        "How do I cancel a direct debit?",
        "What is my credit limit?",
        "I need to update my PIN",
        "I received a suspicious email",
        "My account is frozen",
        "How do I get a refund?",
        "I want to dispute a transaction",
    ]
    for ex in examples:
        st.button(ex, key=f"example_{ex}", on_click=select_example, args=(ex,))

    st.divider()
    st.caption("Dual-Engine Banking Chatbot | SICIP @ BRAC University")
