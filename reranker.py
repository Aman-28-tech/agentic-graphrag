"""
reranker.py — Cross-Encoder Re-ranker

Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to re-score and
re-order retrieval results. Cross-encoders are much more accurate than
bi-encoders because they jointly encode (query, document) pairs.

Pipeline position:
  BM25/Vector → initial candidates (top_k=10) → Cross-Encoder → final top_k=5
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

import config
from utils import console, get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder re-ranker for improved retrieval precision.

    The cross-encoder takes (query, document) pairs and produces a
    relevance score. This is more accurate than cosine similarity
    because it can model token-level interactions between query and doc.
    """

    def __init__(self) -> None:
        console.print(
            f"[cyan]⚡ Loading Cross-Encoder Re-ranker:[/cyan] "
            f"[bold]{config.RERANKER_MODEL}[/bold]"
        )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(config.RERANKER_MODEL, device=device)
        console.print("  [green]✓[/green] Re-ranker ready\n")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Re-rank documents by cross-encoder relevance score.

        Args:
            query:     The user's query string.
            documents: List of candidate document texts.
            top_k:     Number of top results to return (default: config.TOP_K_RERANK).

        Returns:
            List of (document_text, score) sorted by descending relevance.
        """
        top_k = top_k or config.TOP_K_RERANK

        if not documents:
            return []

        # Skip expensive reranking if we already have ≤ top_k candidates
        if len(documents) <= top_k:
            return [(doc, 1.0) for doc in documents]

        # Create (query, doc) pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=len(pairs),       # score all at once
            convert_to_numpy=True,       # faster on CPU
        )

        # Sort by score descending
        scored = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(doc, float(score)) for doc, score in scored[:top_k]]
