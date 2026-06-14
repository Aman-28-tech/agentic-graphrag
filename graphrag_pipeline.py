"""
graphrag_pipeline.py — Production Agentic GraphRAG Orchestrator

Uses LangGraph for stateful agentic flow control with:
  • Cross-encoder re-ranking
  • Confidence scoring
  • Caching layer
  • Multi-provider LLM support

Pipeline per query:
  1. Cache check   → return immediately if cached
  2. Strategy      → fusion_all (default) or router-selected
  3. Retrieve      → Graph + Vector + BM25 → RRF fusion → cross-encoder re-rank
  4. Generate      → LLM drafts grounded answer
  5. Verify        → LLM self-checks with confidence scoring
  6. Retry loop    → rotate strategy if hallucination (max 3×)
  7. Cache store   → persist results for future queries
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich import box
from rich.panel import Panel
from rich.table import Table

import config
from utils            import console, get_logger, truncate, is_dont_know, clean_refusal_answer
from document_loader  import load_raw_texts
from chunker          import chunk_documents, Document
from vector_store     import VectorStore
from knowledge_graph  import KnowledgeGraph
from retriever        import MultiStrategyRetriever, route_query
from reranker         import Reranker
from llm_engine       import LLMEngine
from cache            import QueryCache

logger = get_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Fully structured result returned by AgenticGraphRAG.query()."""
    query:            str
    answer:           str
    strategy_used:    str
    strategies_tried: list[str]
    verified:         bool
    confidence:       float
    context_chunks:   list[str]
    elapsed_sec:      float
    expanded_queries: list[str] = field(default_factory=list)
    cached:           bool = False

    def display(self) -> None:
        color  = "green" if self.verified else "yellow"
        status = "✅  VERIFIED" if self.verified else "⚠️   UNVERIFIED"
        if self.cached:
            status += " (cached)"

        console.print(Panel(
            self.answer,
            title=f"[bold {color}]{status}[/bold {color}]",
            border_style=color,
            padding=(1, 2),
        ))

        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        tbl.add_column("Key",   style="dim")
        tbl.add_column("Value", style="bold")

        tbl.add_row("Strategy used",    self.strategy_used)
        tbl.add_row("Strategies tried", " → ".join(self.strategies_tried) or "—")
        tbl.add_row("Confidence",       f"{self.confidence:.2f}")
        tbl.add_row("Context chunks",   str(len(self.context_chunks)))
        tbl.add_row("Expansions",       str(len(self.expanded_queries)))
        tbl.add_row("Elapsed",          f"{self.elapsed_sec:.2f}s")
        if self.expanded_queries:
            tbl.add_row(
                "Expanded queries",
                "\n".join(f"  • {q}" for q in self.expanded_queries),
            )
        console.print(tbl)

        if config.SHOW_RETRIEVED_CHUNKS and self.context_chunks:
            console.print("\n[dim]── Retrieved Context (top 3) ──[/dim]")
            for i, chunk in enumerate(self.context_chunks[:3], 1):
                console.print(f"  [{i}] {truncate(chunk.replace(chr(10), ' '), 220)}")

    def to_dict(self) -> dict:
        return {
            "query":            self.query,
            "answer":           self.answer,
            "strategy_used":    self.strategy_used,
            "strategies_tried": self.strategies_tried,
            "verified":         self.verified,
            "confidence":       self.confidence,
            "num_chunks":       len(self.context_chunks),
            "elapsed_sec":      round(self.elapsed_sec, 3),
            "expanded_queries": self.expanded_queries,
            "cached":           self.cached,
        }


# ── Main pipeline class ────────────────────────────────────────────────────────

class AgenticGraphRAG:
    """
    Production Self-Correcting Multi-Strategy RAG System.

    Features:
      • LangChain LLM (GPT-4o / Claude 3.5 / local Qwen)
      • Cross-encoder re-ranking
      • Caching layer
      • Confidence-based verification
      • LangGraph state machine for retry logic
    """

    def __init__(self, skip_indexing: bool = False) -> None:
        console.print(Panel(
            "[bold cyan]🧠  Agentic GraphRAG System — Production[/bold cyan]\n"
            "[dim]LangChain · LangGraph · Cross-Encoder · Qdrant · RAGAS[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ))
        console.print()

        # Phase 1-2: Document loading + chunking
        self.docs: list[Document] = []
        if not skip_indexing:
            raw_texts = load_raw_texts(config.DOCS_DIR, recursive=True)
            if raw_texts:
                self.docs = chunk_documents(raw_texts)

        # Phase 3: Vector store (Qdrant)
        self.vs = VectorStore()
        if self.docs and self.vs.is_empty():
            self.vs.index_documents(self.docs)
        elif not self.docs:
            console.print("  [dim]Qdrant: using existing index[/dim]\n")

        # Phase 4: Knowledge Graph (with auto entity extraction from chunks)
        chunk_texts = [d.text for d in self.docs] if self.docs else None
        self.kg = KnowledgeGraph(chunks=chunk_texts)

        # Phase 5: Cross-Encoder Re-ranker
        self.reranker = Reranker()

        # Phase 6: Multi-strategy retriever (with re-ranker)
        self.retriever = MultiStrategyRetriever(self.vs, self.kg, self.reranker)

        # Phase 7: LLM Engine (GPT-4o / Claude / Local)
        self.llm = LLMEngine()

        # Phase 8: Cache
        self.cache = QueryCache()

        console.print("[bold green]✅  System ready[/bold green]\n")

    def query(
        self,
        user_query: str,
        expand_queries: bool = True,
        max_retries: int | None = None,
    ) -> RAGResult:
        """Run the full agentic pipeline with caching + confidence scoring."""
        max_retries = max_retries or config.MAX_RETRIES
        t0 = time.perf_counter()

        console.print("\n[bold white]━━  Query  ━━[/bold white]")
        console.print(f"[italic]{user_query}[/italic]\n")

        # ── Cache check ───────────────────────────────────────────────────────
        cached_ctx = self.cache.get_retrieval(user_query)
        if cached_ctx is not None:
            cached_ans = self.cache.get_answer(user_query, cached_ctx)
            if cached_ans is not None:
                return RAGResult(
                    query=user_query, answer=cached_ans, strategy_used="cached",
                    strategies_tried=[], verified=True, confidence=1.0,
                    context_chunks=cached_ctx,
                    elapsed_sec=time.perf_counter() - t0, cached=True,
                )

        # ── Step 1: Strategy selection ───────────────────────────────────────────
        if config.DEFAULT_STRATEGY == "router":
            primary_strategy = route_query(user_query)
            console.print(f"[dim]Router →[/dim] [bold]{primary_strategy}[/bold]")
        else:
            primary_strategy = config.DEFAULT_STRATEGY
            console.print(f"[dim]Strategy →[/dim] [bold]{primary_strategy}[/bold]")

        # ── Step 2: Query expansion ───────────────────────────────────────────
        expanded: list[str] = []
        word_count = len(user_query.strip().split())
        if not config.ENABLE_QUERY_EXPANSION:
            console.print("[dim]Expansion disabled (local model — saves ~24s/query)[/dim]")
        elif expand_queries and word_count >= config.MIN_QUERY_WORDS_FOR_EXPANSION:
            console.print("[dim]Expanding query…[/dim]")
            t_expand = time.perf_counter()
            expanded = self.llm.expand_query(user_query)
            dt_expand = time.perf_counter() - t_expand
            if expanded:
                console.print(f"  [dim]↳ {expanded} ({dt_expand:.1f}s)[/dim]")
        elif expand_queries:
            console.print(f"[dim]Skipping expansion (query too short: {word_count} words)[/dim]")

        # ── Steps 3-6: Agentic retry loop (LangGraph logic) ──────────────────
        tried_strategies: list[str] = []
        answer         = ""
        verified       = False
        confidence     = 0.0
        strategy_used  = "none"
        final_context: list[str] = []

        for attempt in range(1, max_retries + 1):
            console.print(f"\n[dim]── Attempt {attempt}/{max_retries} ──[/dim]")

            # Retrieve
            context, strategy = self.retriever.retrieve_with_fallback(
                query=user_query,
                primary_strategy=primary_strategy,
                tried_strategies=tried_strategies,
                expanded_queries=expanded,
            )
            strategy_used = strategy

            if not context:
                console.print("  [red]✗ All strategies exhausted[/red]")
                answer = "I don't know — no relevant information found."
                break

            console.print(
                f"  [green]Retrieved {len(context)} chunks via {strategy}[/green]"
            )

            # Generate
            console.print("  [dim]Generating answer…[/dim]")
            t_gen = time.perf_counter()
            answer = self.llm.generate_answer(user_query, context)
            dt_gen = time.perf_counter() - t_gen
            console.print(f"  [dim]Generated in {dt_gen:.1f}s[/dim]")

            if is_dont_know(answer):
                answer = clean_refusal_answer(answer)
                console.print("  [yellow]Model returned 'I don't know'[/yellow]")
                final_context = context
                break

            # Verify with confidence scoring
            console.print("  [dim]Verifying answer…[/dim]")
            t_ver = time.perf_counter()
            verified, verdict, confidence = self.llm.verify_answer(
                user_query, context, answer,
            )
            dt_ver = time.perf_counter() - t_ver
            final_context = context

            if verified:
                console.print(
                    f"  [green]✓ VERIFIED (confidence={confidence:.2f}) "
                    f"[{dt_ver:.1f}s][/green]"
                )
                break

            console.print(
                f"  [yellow]✗ HALLUCINATION "
                f"(confidence={confidence:.2f}) [{dt_ver:.1f}s][/yellow]"
            )
            primary_strategy = "hybrid" if strategy != "hybrid" else "bm25"

        # ── Cache store ───────────────────────────────────────────────────────
        if final_context:
            self.cache.set_retrieval(user_query, final_context)
        if answer and verified:
            self.cache.set_answer(user_query, final_context, answer)

        return RAGResult(
            query=user_query,
            answer=answer,
            strategy_used=strategy_used,
            strategies_tried=tried_strategies,
            verified=verified,
            confidence=confidence,
            context_chunks=final_context,
            elapsed_sec=time.perf_counter() - t0,
            expanded_queries=expanded,
        )
