"""
api.py — FastAPI Backend for Agentic GraphRAG

Endpoints:
  POST /query           → Run a RAG query and return structured result
  GET  /health          → Health check
  GET  /stats           → System statistics (cache, Qdrant, KG)
  POST /clear-cache     → Clear the retrieval + answer cache

Usage:
  python api.py
  # or
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import config

app = FastAPI(
    title="Agentic GraphRAG API",
    description=(
        "Self-Correcting Multi-Strategy RAG with Knowledge Graph Integration. "
        "Supports GPT-4o, Claude 3.5 Sonnet, and local Qwen models."
    ),
    version="2.0.0",
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (lazy init)
_rag = None


def _get_rag():
    """Lazy-initialize the RAG system on first request."""
    global _rag
    if _rag is None:
        from graphrag_pipeline import AgenticGraphRAG
        _rag = AgenticGraphRAG(skip_indexing=True)
    return _rag


# ── Request / Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to answer", min_length=3)
    expand_queries: bool = Field(True, description="Enable query expansion")
    max_retries: Optional[int] = Field(None, description="Override max retries")


class QueryResponse(BaseModel):
    query:            str
    answer:           str
    strategy_used:    str
    strategies_tried: list[str]
    verified:         bool
    confidence:       float
    num_chunks:       int
    elapsed_sec:      float
    expanded_queries: list[str]
    cached:           bool
    context_preview:  list[str]   # first 100 chars of each chunk


class HealthResponse(BaseModel):
    status:   str
    provider: str
    qdrant:   str
    uptime:   float


class StatsResponse(BaseModel):
    cache_stats:  dict
    qdrant_count: int
    kg_nodes:     int
    kg_edges:     int


# ── Endpoints ──────────────────────────────────────────────────────────────────

_start_time = time.time()


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    provider_str = "Groq (llama-3.3-70b-versatile)" if config.LLM_PROVIDER == "local" else config.LLM_PROVIDER
    return HealthResponse(
        status="healthy",
        provider=provider_str,
        qdrant=config.QDRANT_MODE,
        uptime=round(time.time() - _start_time, 1),
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Run a RAG query through the full agentic pipeline.

    The pipeline: route → expand → retrieve → re-rank → generate → verify → (retry)
    """
    try:
        rag    = _get_rag()
        result = rag.query(
            user_query=req.query,
            expand_queries=req.expand_queries,
            max_retries=req.max_retries,
        )

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            strategy_used=result.strategy_used,
            strategies_tried=result.strategies_tried,
            verified=result.verified,
            confidence=result.confidence,
            num_chunks=len(result.context_chunks),
            elapsed_sec=round(result.elapsed_sec, 3),
            expanded_queries=result.expanded_queries,
            cached=result.cached,
            context_preview=[c[:100] + "…" for c in result.context_chunks[:5]],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return system statistics."""
    rag = _get_rag()
    return StatsResponse(
        cache_stats=rag.cache.stats,
        qdrant_count=rag.vs.client.get_collection(
            config.QDRANT_COLLECTION
        ).points_count,
        kg_nodes=rag.kg.graph.number_of_nodes(),
        kg_edges=rag.kg.graph.number_of_edges(),
    )


@app.post("/clear-cache")
async def clear_cache():
    """Clear the retrieval and answer cache."""
    rag = _get_rag()
    rag.cache.clear()
    return {"status": "cache cleared"}


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
