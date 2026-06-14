"""
app.py — Streamlit Frontend for Agentic GraphRAG

A clean web UI for testing the RAG system interactively.
Connects to the FastAPI backend at http://localhost:8000.

Run:
  streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import os
import requests
import streamlit as st
import time

# ── Configuration ──────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic GraphRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    expand_queries = st.toggle("Query Expansion", value=True,
                               help="Generate alternative query phrasings")
    max_retries = st.slider("Max Retries", 1, 5, 3,
                            help="Self-correction retry limit")

    st.markdown("---")
    st.markdown("## 📊 System")

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: {health['status']}")
        st.info(f"LLM: {health['provider']}")
        st.info(f"Qdrant: {health['qdrant']}")
    except Exception:
        st.error("❌ API offline — start with `python api.py`")

    # Stats
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        st.metric("Qdrant Vectors", stats["qdrant_count"])
        st.metric("KG Nodes", stats["kg_nodes"])
        st.metric("KG Edges", stats["kg_edges"])
        st.metric("Cache Hit Rate", stats["cache_stats"]["hit_rate"])
    except Exception:
        pass

    if st.button("🗑️ Clear Cache"):
        try:
            requests.post(f"{API_URL}/clear-cache", timeout=3)
            st.success("Cache cleared!")
        except Exception:
            st.error("Failed to clear cache")

    st.markdown("---")
    st.markdown(
        "**Stack:** LangChain · LangGraph · Qdrant · Cross-Encoder · RAGAS"
    )


# ── Main Content ──────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🧠 Agentic GraphRAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Self-Correcting · Multi-Strategy · Knowledge-Graph-Augmented RAG'
    '</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Query input
query = st.text_area(
    "Ask a question about the research papers:",
    placeholder="e.g., What is the relationship between BERT and the Transformer architecture?",
    height=100,
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    submit = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    demo_query = st.selectbox(
        "Demo queries:",
        [
            "",
            "What is the relationship between BERT and the Transformer?",
            "How does multi-head attention work?",
            "What problem does Bahdanau attention solve?",
            "How many parameters does BERT-Large have?",
            "What is the boiling point of liquid nitrogen?",
        ],
        label_visibility="collapsed",
    )
with col3:
    if demo_query:
        query = demo_query
        submit = True

# ── Process Query ──────────────────────────────────────────────────────────────

if submit and query:
    with st.spinner("🧠 Thinking..."):
        start = time.time()
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": query,
                    "expand_queries": expand_queries,
                    "max_retries": max_retries,
                },
                timeout=600,
            ).json()

            elapsed = time.time() - start

            # ── Answer Display ────────────────────────────────────────────────
            if response.get("verified"):
                st.success("✅ **VERIFIED** — Answer is grounded in source documents")
            else:
                st.warning("⚠️ **UNVERIFIED** — Answer may not be fully supported")

            st.markdown(f"### Answer\n{response['answer']}")

            # ── Metrics Row ───────────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Strategy", response["strategy_used"])
            m2.metric("Confidence", f"{response['confidence']:.2f}")
            m3.metric("Chunks", response["num_chunks"])
            m4.metric("Time", f"{response['elapsed_sec']:.1f}s")
            m5.metric("Cached", "Yes" if response["cached"] else "No")

            # ── Expanded Queries ──────────────────────────────────────────────
            if response.get("expanded_queries"):
                with st.expander("🔀 Expanded Queries"):
                    for eq in response["expanded_queries"]:
                        st.markdown(f"- {eq}")

            # ── Context Chunks ────────────────────────────────────────────────
            if response.get("context_preview"):
                with st.expander("📄 Retrieved Context"):
                    for i, chunk in enumerate(response["context_preview"], 1):
                        st.markdown(f"**[{i}]** {chunk}")

            # ── Strategy Flow ─────────────────────────────────────────────────
            if response.get("strategies_tried"):
                with st.expander("🔄 Strategy Flow"):
                    st.markdown(
                        " → ".join(
                            f"**{s}**" for s in response["strategies_tried"]
                        )
                    )

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to API. "
                "Start the backend with: `python api.py`"
            )
        except Exception as exc:
            st.error(f"Error: {exc}")

elif submit and not query:
    st.warning("Please enter a query.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>"
    "Agentic GraphRAG v2.0 — "
    "Built with LangChain · LangGraph · Qdrant · FastAPI · Streamlit"
    "</small></center>",
    unsafe_allow_html=True,
)
