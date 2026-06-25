"""
app.py
------
Streamlit front-end for the Dual-Engine Banking Chatbot.

Launch:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.utils import load_config

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Banking Assistant",
    page_icon="🏦",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    body { font-family: 'Segoe UI', sans-serif; }
    .main { background: #f0f4f8; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .escalate-yes {
        background: #1a1a2e;
        border-left: 5px solid #ffc107;
        padding: 12px 18px;
        border-radius: 6px;
    }
    .escalate-no {
        background: #1a1a2e;
        border-left: 5px solid #28a745;
        padding: 12px 18px;
        border-radius: 6px;
    }
    .answer-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #0d6efd;
        margin-top: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load pipeline (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models… this may take a minute on first run.")
def load_pipeline():
    from src.predict import DualEnginePipeline
    cfg = load_config("configs/config.yaml")
    return DualEnginePipeline(cfg)


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
# Input
# ──────────────────────────────────────────────

query = st.text_area(
    "📝 Enter your banking question:",
    placeholder="e.g. My card was charged twice for the same transaction.",
    height=100,
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    submitted = st.button("🔍 Get Answer", type="primary", use_container_width=True)
with col_clear:
    clear = st.button("Clear", use_container_width=False)

if clear:
    st.rerun()

# ──────────────────────────────────────────────
# Pipeline & output
# ──────────────────────────────────────────────

if submitted:
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        try:
            pipeline = load_pipeline()
        except FileNotFoundError as exc:
            st.error(
                f"Models not found: {exc}\n\n"
                "Please run the following commands first:\n"
                "```\npython src/data_download.py\n"
                "python src/train.py --config configs/config.yaml\n"
                "python src/retrieval.py --build\n```"
            )
            st.stop()

        with st.spinner("Analysing your query…"):
            result = pipeline.predict(query)

        st.divider()

        # ── Generated Answer ─────────────────────────
        st.subheader("💬 Generated Answer")
        st.markdown(
            f'<div class="answer-box">{result["generated_answer"]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Metrics ──────────────────────────────────
        st.subheader("📊 Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🎯 Detected Intent",
                value=result["detected_intent"].replace("_", " ").title(),
            )

        with col2:
            conf = result["confidence_score"]
            delta_color = "normal" if conf >= 0.60 else "inverse"
            st.metric(
                label="🧠 Classifier Confidence",
                value=f"{conf:.1%}",
                delta=f"{'High' if conf >= 0.80 else 'Medium' if conf >= 0.60 else 'Low'}",
                delta_color=delta_color,
            )

        with col3:
            sim = result["retrieval_similarity"]
            st.metric(
                label="🔍 Retrieval Similarity",
                value=f"{sim:.1%}",
                delta=f"{'High' if sim >= 0.80 else 'Medium' if sim >= 0.50 else 'Low'}",
                delta_color="normal" if sim >= 0.50 else "inverse",
            )

        st.divider()

        # ── Escalation ───────────────────────────────
        esc = result["escalation_recommendation"]
        if esc == "Yes":
            st.markdown(
                f"""
                <div class="escalate-yes">
                <strong>⚠️ Escalation Recommended</strong><br>
                This query has been flagged for human review.<br>
                <em>Reasons: {', '.join(result.get('escalation_reasons', []))}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="escalate-no">
                <strong>✅ No Escalation Needed</strong><br>
                The system is confident in this response.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Raw JSON ─────────────────────────────────
        with st.expander("🔧 Raw JSON Output"):
            st.json(result)

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
        - Vector DB: FAISS
        - Generator: `flan-t5-small`

        **Escalation Engine**
        - Flags low-confidence queries
        - Flags out-of-domain queries
        """
    )
    st.divider()
    st.header("📋 Example Queries")
    examples = [
        "My card was charged twice",
        "How do I cancel a direct debit?",
        "What is my credit limit?",
        "I need to update my PIN",
        "I received a suspicious email",
        "My account is frozen",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query"] = ex
