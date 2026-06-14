"""
cache.py — Retrieval & Answer Caching Layer

Avoids redundant LLM calls and vector searches for repeated/similar queries.

Two cache levels:
  1. Retrieval cache  — stores (query → context_chunks) to skip vector/BM25/graph search
  2. Answer cache     — stores (query + context_hash → answer) to skip LLM generation

Cache is in-memory (dict-based) with TTL expiration.
For production, swap the backend to Redis with minimal code changes.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import config
from utils import console, get_logger

logger = get_logger(__name__)


class CacheEntry:
    """A single cache entry with TTL tracking."""
    __slots__ = ("value", "created_at", "ttl")

    def __init__(self, value: Any, ttl: int) -> None:
        self.value      = value
        self.created_at = time.time()
        self.ttl        = ttl

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class QueryCache:
    """
    In-memory cache for retrieval results and generated answers.

    Usage:
        cache = QueryCache()

        # Check retrieval cache
        cached = cache.get_retrieval("What is BERT?")
        if cached is None:
            results = retriever.retrieve(...)
            cache.set_retrieval("What is BERT?", results)

        # Check answer cache
        cached_answer = cache.get_answer("What is BERT?", context_chunks)
        if cached_answer is None:
            answer = llm.generate(...)
            cache.set_answer("What is BERT?", context_chunks, answer)
    """

    def __init__(self, enabled: bool = None, ttl: int = None) -> None:
        self.enabled = enabled if enabled is not None else config.CACHE_ENABLED
        self.ttl     = ttl or config.CACHE_TTL_SECONDS

        self._retrieval_cache: dict[str, CacheEntry] = {}
        self._answer_cache:    dict[str, CacheEntry] = {}

        self._hits   = 0
        self._misses = 0

        if self.enabled:
            console.print(
                f"  [green]✓[/green] Cache enabled (TTL={self.ttl}s)\n"
            )

    # ── Key generation ─────────────────────────────────────────────────────────

    @staticmethod
    def _query_key(query: str) -> str:
        """Normalise query into a cache key."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    @staticmethod
    def _answer_key(query: str, context_chunks: list[str]) -> str:
        """Combine query + context hash for answer cache key."""
        ctx_hash = hashlib.md5(
            "||".join(c[:200] for c in context_chunks).encode()
        ).hexdigest()
        q_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()
        return f"{q_hash}:{ctx_hash}"

    # ── Retrieval cache ────────────────────────────────────────────────────────

    def get_retrieval(self, query: str) -> Optional[list[str]]:
        """Return cached retrieval results, or None if miss/expired."""
        if not self.enabled:
            return None

        key   = self._query_key(query)
        entry = self._retrieval_cache.get(key)

        if entry is not None and not entry.is_expired():
            self._hits += 1
            console.print("  [dim]Cache HIT (retrieval)[/dim]")
            return entry.value

        self._misses += 1
        return None

    def set_retrieval(self, query: str, results: list[str]) -> None:
        if not self.enabled:
            return
        self._retrieval_cache[self._query_key(query)] = CacheEntry(results, self.ttl)

    # ── Answer cache ───────────────────────────────────────────────────────────

    def get_answer(self, query: str, context_chunks: list[str]) -> Optional[str]:
        """Return cached answer, or None if miss/expired."""
        if not self.enabled:
            return None

        key   = self._answer_key(query, context_chunks)
        entry = self._answer_cache.get(key)

        if entry is not None and not entry.is_expired():
            self._hits += 1
            console.print("  [dim]Cache HIT (answer)[/dim]")
            return entry.value

        self._misses += 1
        return None

    def set_answer(self, query: str, context_chunks: list[str], answer: str) -> None:
        if not self.enabled:
            return
        key = self._answer_key(query, context_chunks)
        self._answer_cache[key] = CacheEntry(answer, self.ttl)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._retrieval_cache.clear()
        self._answer_cache.clear()
        self._hits = self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  f"{self._hits / total:.1%}" if total else "N/A",
            "size":      len(self._retrieval_cache) + len(self._answer_cache),
        }
