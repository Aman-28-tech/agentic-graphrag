"""
bm25_retriever.py — Phase 3a: BM25 Keyword Search

Uses Okapi BM25 (rank-bm25) for exact-term / TF-IDF-style retrieval.
Complements vector search by catching specific entity names and technical terms
that semantic embeddings may underweight.

Persistence
-----------
The BM25 index can be serialized to disk with save() / loaded back with load()
so that re-indexing is not needed on every run. The ingest pipeline calls save()
after building the index; the pipeline loads it via load_or_build().
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from rank_bm25 import BM25Okapi

import config
from utils import console, get_logger, tokenize

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25 (Okapi) keyword-based retriever with optional disk persistence.

    Usage
    -----
    # Build fresh
    retriever = BM25Retriever(corpus)
    retriever.save()

    # Load from disk (skips re-building)
    retriever = BM25Retriever.load()
    results   = retriever.search("attention mechanism", top_k=5)
    # → [(text, score), ...]
    """

    def __init__(self, corpus: list[str]) -> None:
        """
        Build the BM25 index from a flat list of text chunks.

        Args:
            corpus: All chunk texts (same order as stored in Qdrant).
        """
        console.print(f"[cyan]⚡ Building BM25 index ({len(corpus)} chunks)…[/cyan]")
        self.corpus   = corpus
        tokenized     = [tokenize(doc) for doc in corpus]
        self.bm25     = BM25Okapi(tokenized)
        console.print("  [green]✓[/green] BM25 index ready\n")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """
        Serialize the index + corpus to *path* (default: config.BM25_INDEX_PATH).
        Subsequent runs can call :meth:`load` instead of rebuilding.
        """
        path = path or config.BM25_INDEX_PATH
        with open(path, "wb") as fh:
            pickle.dump({"corpus": self.corpus, "bm25": self.bm25}, fh)
        size_kb = os.path.getsize(path) / 1024
        console.print(
            f"  [green]✓[/green] BM25 index saved → {path} ({size_kb:.0f} KB)\n"
        )

    @classmethod
    def load(cls, path: str | None = None) -> "BM25Retriever":
        """
        Deserialize a previously saved index from *path*.

        Returns:
            BM25Retriever instance (skips tokenization/re-fitting).

        Raises:
            FileNotFoundError: if the index file does not exist.
        """
        path = path or config.BM25_INDEX_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found at {path!r}")
        console.print(f"[cyan]⚡ Loading BM25 index from {path}…[/cyan]")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj         = cls.__new__(cls)
        obj.corpus  = data["corpus"]
        obj.bm25    = data["bm25"]
        console.print(
            f"  [green]✓[/green] BM25 index loaded "
            f"([bold]{len(obj.corpus)}[/bold] chunks)\n"
        )
        return obj

    @classmethod
    def load_or_build(cls, corpus: list[str], path: str | None = None) -> "BM25Retriever":
        """
        Return a persisted index if it exists, otherwise build one from *corpus*.

        Args:
            corpus: Corpus to build from if no index is found on disk.
            path:   Path to the index file (default: config.BM25_INDEX_PATH).
        """
        path = path or config.BM25_INDEX_PATH
        if os.path.exists(path):
            return cls.load(path)
        instance = cls(corpus)
        instance.save(path)
        return instance

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """
        Score all corpus documents against *query* and return the top-k.

        Args:
            query:  Raw query string (tokenised internally).
            top_k:  Number of results; defaults to ``config.TOP_K_BM25``.

        Returns:
            List of (text, score) tuples, descending by BM25 score.
            Entries with score 0 are excluded.
        """
        top_k        = top_k or config.TOP_K_BM25
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores  = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self.corpus[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0.0
        ]
