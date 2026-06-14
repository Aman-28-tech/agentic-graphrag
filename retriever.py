"""
retriever.py — Multi-Strategy Retrieval with RRF Fusion + Cross-Encoder

Production retrieval pipeline:
  1. Fusion-All mode  → Graph + Vector + BM25 combined via RRF (default)
  2. Query Router     → selects single strategy (graph / hybrid / bm25)
  3. Cross-Encoder    → re-ranks initial candidates for precision
  4. Fallback chain   → rotates through strategies if primary fails

The re-ranker sits AFTER initial retrieval and BEFORE answer generation,
turning coarse recall into precise, high-quality context.
"""

from __future__ import annotations

import re
from typing import Literal

import config
from utils import console, get_logger, deduplicate

from bm25_retriever  import BM25Retriever
from knowledge_graph import KnowledgeGraph
from vector_store    import VectorStore
from reranker        import Reranker

logger = get_logger(__name__)

Strategy = Literal["graph", "hybrid", "bm25", "vector", "fusion_all"]
_STRATEGY_ORDER: list[Strategy] = ["fusion_all", "graph", "hybrid", "bm25", "vector"]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    """Merge ranked lists using RRF scoring."""
    k = k or config.RRF_K
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Query Router ───────────────────────────────────────────────────────────────

_RE_GRAPH = re.compile(
    r"\b(relation(ship)?|uses?|used by|created by|developed by|"
    r"introduced|between|connect|link(ed)?|depend)\b",
    re.IGNORECASE,
)
_RE_ENTITY = re.compile(r"^[A-Z][A-Za-z0-9\-]+([\s,][A-Z][A-Za-z0-9\-]*)*[\s?]*$")


def route_query(query: str) -> Strategy:
    """Select the primary retrieval strategy for a query."""
    q = query.strip()
    if _RE_GRAPH.search(q):
        return "graph"
    if _RE_ENTITY.match(q) and len(q.split()) <= 4:
        return "bm25"
    return "hybrid"


# ── Multi-Strategy Retriever ───────────────────────────────────────────────────

class MultiStrategyRetriever:
    """
    Unified retriever with BM25 + Vector + Graph strategies,
    RRF fusion, and Cross-Encoder re-ranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        kg: KnowledgeGraph,
        reranker: Reranker | None = None,
    ) -> None:
        self.vs       = vector_store
        self.kg       = kg
        self.reranker = reranker
        self._bm25: BM25Retriever | None = None
        self._last_kg_results: list[str] = []  # KG docs from last fusion_all call

    def _get_bm25(self) -> BM25Retriever:
        """Lazy-load BM25 from disk or build from Qdrant corpus."""
        if self._bm25 is None:
            corpus     = self.vs.get_all_texts()
            self._bm25 = BM25Retriever.load_or_build(corpus, config.BM25_INDEX_PATH)
        return self._bm25

    # ── Individual strategies ──────────────────────────────────────────────────

    def _graph(self, query: str) -> list[str]:
        return self.kg.search(query)

    def _vector(self, query: str) -> list[str]:
        return [text for text, _, _ in self.vs.search(query, top_k=config.TOP_K_QDRANT)]

    def _bm25_search(self, query: str) -> list[str]:
        return [text for text, _ in self._get_bm25().search(query, top_k=config.TOP_K_BM25)]

    def _hybrid(self, query: str) -> list[str]:
        """Vector + BM25 fused via RRF."""
        fused = reciprocal_rank_fusion([self._vector(query), self._bm25_search(query)])
        return [doc for doc, _ in fused]

    def _fusion_all(self, query: str) -> list[str]:
        """
        Run ALL three retrieval methods and fuse via RRF.

        Graph + Vector + BM25 → RRF → deduplicated candidates.
        Graph results are cached for KG slot reservation after reranking.
        """
        graph_docs  = self._graph(query)
        vector_docs = self._vector(query)
        bm25_docs   = self._bm25_search(query)

        # Cache KG results for slot reservation in retrieve()
        # Prefix with "IMPORTANT FACT:" so the LLM prioritizes graph-derived
        # knowledge over ambiguous document chunks
        self._last_kg_results = [
            f"IMPORTANT FACT: {doc}" if not doc.startswith("IMPORTANT FACT:") else doc
            for doc in graph_docs[:config.KG_RESERVED_SLOTS]
        ]

        # Log per-source counts for debugging
        console.print(
            f"  [dim]Retrieval sources: "
            f"Graph={len(graph_docs)}, "
            f"Vector={len(vector_docs)}, "
            f"BM25={len(bm25_docs)}[/dim]"
        )

        # Collect non-empty ranked lists for RRF
        ranked_lists = [r for r in [graph_docs, vector_docs, bm25_docs] if r]

        if not ranked_lists:
            return []

        if len(ranked_lists) == 1:
            return ranked_lists[0]

        fused = reciprocal_rank_fusion(ranked_lists)
        return [doc for doc, _ in fused]

    def _dispatch(self, query: str, strategy: Strategy) -> list[str]:
        """Run a single strategy; graph auto-falls back to hybrid if empty."""
        if strategy == "fusion_all":
            return self._fusion_all(query)
        if strategy == "graph":
            results = self._graph(query)
            if not results:
                console.print("  [dim]Graph empty → fallback to hybrid[/dim]")
                results = self._hybrid(query)
            return results
        if strategy == "bm25":
            return self._bm25_search(query)
        if strategy == "hybrid":
            return self._hybrid(query)
        return self._vector(query)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        strategy: Strategy,
        expanded_queries: list[str] | None = None,
    ) -> list[str]:
        """
        Retrieve context using the specified strategy + expanded queries.
        Results are fused via RRF and then re-ranked by cross-encoder.
        """
        all_queries = [query] + (expanded_queries or [])
        all_ranked:  list[list[str]] = []

        for q in all_queries:
            results = self._dispatch(q, strategy)
            if results:
                all_ranked.append(results)

        if not all_ranked:
            return []

        # Fuse across query phrasings
        if len(all_ranked) == 1:
            candidates = deduplicate(all_ranked[0])
        else:
            fused      = reciprocal_rank_fusion(all_ranked)
            candidates = deduplicate([doc for doc, _ in fused])

        # Cross-Encoder re-ranking (if available and worthwhile)
        if self.reranker and candidates and len(candidates) > config.TOP_K_RERANK:
            console.print("  [dim]Re-ranking with cross-encoder…[/dim]")
            
            # --- Entity-Aware Chunk Reservation ---
            import re
            query_lower = query.lower()
            known_entities = [node[0] for node in config.KG_NODES]
            
            query_entities = []
            for ent in known_entities:
                if re.search(rf"\b{re.escape(ent.lower())}\b", query_lower):
                    query_entities.append(ent)
            
            if len(query_entities) > 1:
                # Score all candidates so we can pick best per entity
                reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))
                console.print(f"  [cyan]Entity-aware retrieval active for:[/cyan] [bold]{', '.join(query_entities)}[/bold]")
                
                slots_per_entity = max(1, config.TOP_K_RERANK // (len(query_entities) + 1))
                entity_docs = {ent: [] for ent in query_entities}
                general_docs = []
                
                for doc, _ in reranked:
                    doc_lower = doc.lower()
                    assigned = False
                    for ent in query_entities:
                        if len(entity_docs[ent]) < slots_per_entity and re.search(rf"\b{re.escape(ent.lower())}\b", doc_lower):
                            entity_docs[ent].append(doc)
                            assigned = True
                            break
                    if not assigned:
                        general_docs.append(doc)
                
                final = []
                for docs in entity_docs.values():
                    final.extend(docs)
                
                remaining = config.TOP_K_RERANK - len(final)
                final.extend(general_docs[:remaining])
                
                # Fallback to ensure we hit TOP_K
                if len(final) < config.TOP_K_RERANK:
                    for doc, _ in reranked:
                        if doc not in final:
                            final.append(doc)
                        if len(final) >= config.TOP_K_RERANK:
                            break
            else:
                reranked = self.reranker.rerank(query, candidates, top_k=config.TOP_K_RERANK)
                final = [doc for doc, _ in reranked]

            # KG slot reservation: inject top graph results that the cross-encoder
            # may have dropped. ms-marco is trained for web search and systematically
            # under-scores KG edge context passages like "BERT -> MLM".
            if self._last_kg_results:
                kg_to_inject = [kg for kg in self._last_kg_results if kg not in final]
                if kg_to_inject:
                    console.print(
                        f"  [dim]Reserving {len(kg_to_inject)} KG slot(s) "
                        f"(protected from cross-encoder)[/dim]"
                    )
                    # Prepend KG facts so they appear first in context
                    final = kg_to_inject + final
                self._last_kg_results = []  # reset for next query

            return final

        return candidates[:config.TOP_K_RERANK]

    def retrieve_with_fallback(
        self,
        query: str,
        primary_strategy: Strategy,
        tried_strategies: list[Strategy],
        expanded_queries: list[str] | None = None,
    ) -> tuple[list[str], Strategy]:
        """Try primary strategy, then rotate through fallbacks."""
        ordered = [primary_strategy] + [
            s for s in _STRATEGY_ORDER if s != primary_strategy
        ]

        for strategy in ordered:
            if strategy in tried_strategies:
                continue
            tried_strategies.append(strategy)

            console.print(f"  [dim]Trying strategy: [bold]{strategy}[/bold]…[/dim]")
            results = self.retrieve(query, strategy, expanded_queries)
            if results:
                return results, strategy
            console.print(f"  [yellow]Strategy '{strategy}' returned no results.[/yellow]")

        return [], "none"
