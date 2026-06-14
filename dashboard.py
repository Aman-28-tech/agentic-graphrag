"""
dashboard.py — Production GraphRAG Evaluation Dashboard

Streamlit dashboard with:
  • Production readiness scorecard
  • RAGAS radar chart
  • Per-query latency breakdown
  • Strategy distribution
  • Verification rate donut
  • Embedded PyVis knowledge graph
"""

import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fusion GraphRAG Evaluation Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Fusion Agentic GraphRAG — Evaluation Dashboard")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(filepath="eval_results.json"):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


df = load_data()

if df.empty:
    st.warning(
        "No evaluation data found. Run `python evaluate.py` to generate `eval_results.json`."
    )
    st.stop()


# ── Scorecard ──────────────────────────────────────────────────────────────────
st.markdown("### 🏆 Production Readiness Scorecard")

cols = st.columns(6)

avg_latency = df["elapsed_sec"].mean()
cols[0].metric("Avg Latency", f"{avg_latency:.1f}s", delta=f"{'🟢' if avg_latency < 30 else '🔴'}")

avg_confidence = df["confidence"].mean()
cols[1].metric("Avg Confidence", f"{avg_confidence:.2f}", delta=f"{'🟢' if avg_confidence >= 0.7 else '🔴'}")

avg_khr = df["keyword_hit_rate"].mean()
cols[2].metric("Keyword Hit Rate", f"{avg_khr:.0%}", delta=f"{'🟢' if avg_khr >= 0.6 else '🔴'}")

verified_rate = df["verified"].mean() if "verified" in df.columns else 0
cols[3].metric("Verified Rate", f"{verified_rate:.0%}", delta=f"{'🟢' if verified_rate >= 0.7 else '🔴'}")

pass_rate = ((df["keyword_hit_rate"] >= 0.5) & (df["verified"] | (df["confidence"] >= 0.6))).mean() if not df.empty else 0
cols[4].metric("Pass Rate", f"{pass_rate:.0%}", delta=f"{'🟢' if pass_rate >= 0.8 else '🔴'}")

total_queries = len(df)
cols[5].metric("Total Queries", str(total_queries))

st.markdown("---")


# ── Row 1: RAGAS + Verification ────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### 🚀 RAGAS Metrics")
    ragas_cols = ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision"]
    has_ragas = all(c in df.columns for c in ragas_cols) and df[ragas_cols[0]].notnull().any()

    if has_ragas:
        means = [df[c].mean() for c in ragas_cols]
        labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision"]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=means + [means[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#636EFA"),
            name="RAGAS",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("RAGAS metrics not computed. Run evaluation without `--no-ragas` to populate.")

with col2:
    st.markdown("### ✅ Verification Rate")
    if "verified" in df.columns:
        v_count = df["verified"].sum()
        u_count = len(df) - v_count
        fig = go.Figure(go.Pie(
            labels=["Verified", "Unverified"],
            values=[v_count, u_count],
            hole=0.6,
            marker=dict(colors=["#00CC96", "#EF553B"]),
        ))
        fig.update_layout(
            showlegend=True,
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("### 🔀 Strategy Distribution")
    if "strategy_used" in df.columns:
        strat_counts = df["strategy_used"].value_counts()
        fig = go.Figure(go.Pie(
            labels=strat_counts.index.tolist(),
            values=strat_counts.values.tolist(),
            hole=0.5,
        ))
        fig.update_layout(
            showlegend=True,
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Row 2: Latency + Confidence per query ──────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.markdown("### ⏱️ Per-Query Latency")
    df_sorted = df.sort_values("elapsed_sec", ascending=True).reset_index(drop=True)
    colors = ["#00CC96" if t < 30 else "#FFA15A" if t < 60 else "#EF553B" for t in df_sorted["elapsed_sec"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["elapsed_sec"],
        y=[q[:40] + "…" if len(q) > 40 else q for q in df_sorted["query"]],
        orientation="h",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        xaxis_title="Seconds",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(df) * 40),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col5:
    st.markdown("### 📊 Confidence Distribution")
    fig = px.histogram(
        df, x="confidence", nbins=10,
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Row 3: Query-level table ──────────────────────────────────────────────────
st.markdown("### 🔍 Query-level Results")
display_cols = [c for c in ["query", "strategy_used", "verified", "confidence", "keyword_hit_rate", "elapsed_sec"] if c in df.columns]
st.dataframe(
    df[display_cols].style.format({
        "confidence": "{:.2f}",
        "keyword_hit_rate": "{:.0%}",
        "elapsed_sec": "{:.1f}s",
    }).highlight_max(axis=0, subset=["confidence", "keyword_hit_rate"]),
    use_container_width=True,
)


# ── Row 4: Precision/Recall proxy ─────────────────────────────────────────────
st.markdown("### 🛠️ Information Retrieval (Keyword Hit Rate as Recall Proxy)")
st.info("Note: True Precision@K / Recall@K require ground-truth relevance labels. Keyword hit rate serves as a lightweight recall proxy.")
if "keyword_hit_rate" in df.columns:
    fig = px.bar(
        df, x="query", y="keyword_hit_rate",
        color="keyword_hit_rate",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 1],
        height=350,
        margin=dict(l=10, r=10, t=10, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Row 5: Knowledge Graph ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🕸️ Knowledge Graph Visualization")

graph_path = os.path.join(os.path.dirname(__file__), "graph_visualization.html")
if os.path.exists(graph_path):
    with open(graph_path, "r") as f:
        html_data = f.read()
    components.html(html_data, height=800, scrolling=True)
else:
    st.info(
        "Graph visualization not found. Run `rag.kg.export_visual()` to generate it."
    )
    if st.button("🔄 Generate Graph Visualization"):
        st.info("Please run the following in your Python environment:\n\n"
                "```python\nfrom graphrag_pipeline import AgenticGraphRAG\n"
                "rag = AgenticGraphRAG(skip_indexing=True)\n"
                "rag.kg.export_visual()\n```")
