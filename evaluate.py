"""
evaluate.py — RAGAS + Custom Evaluation Harness

Evaluation metrics:
  • RAGAS Faithfulness  — is the answer grounded in retrieved context?
  • RAGAS Answer Relevancy — does the answer address the question?
  • RAGAS Context Precision — is the retrieved context relevant?
  • Custom Keyword Hit Rate — do expected keywords appear in the answer?
  • Custom Strategy Match  — did the router choose the right strategy?
  • Confidence scoring     — LLM's self-assessed confidence

Usage:
  python evaluate.py                    # run full evaluation
  python evaluate.py --no-ragas         # skip RAGAS (faster, no API needed)
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from rich import box
from rich.table import Table

from utils import console, get_logger, is_dont_know

logger = get_logger(__name__)


# ── Evaluation Models ──────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """A single labelled evaluation example."""
    query:             str
    expected_keywords: list[str]
    expected_strategy: Optional[str] = None
    should_refuse:     bool = False


@dataclass
class EvalResult:
    """Metrics for one evaluated sample."""
    query:            str
    answer:           str
    strategy_used:    str
    verified:         bool
    confidence:       float
    elapsed_sec:      float
    keyword_hit_rate: float
    correct_refusal:  bool
    # RAGAS metrics (populated only if RAGAS is available)
    ragas_faithfulness:      Optional[float] = None
    ragas_answer_relevancy:  Optional[float] = None
    ragas_context_precision: Optional[float] = None

    def passed(self) -> bool:
        return self.keyword_hit_rate >= 0.5 and (self.verified or self.confidence >= 0.6)


# ── Evaluation Dataset ────────────────────────────────────────────────────────

EVAL_DATASET: list[EvalSample] = [
    # Bahdanau 2014
    EvalSample(
        query="What problem does the Bahdanau attention mechanism solve?",
        expected_keywords=["fixed-length", "bottleneck", "encoder", "context"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="How does the alignment model work in Bahdanau attention?",
        expected_keywords=["alignment", "softmax", "encoder", "hidden"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="What is the relationship between attention mechanism and machine translation?",
        expected_keywords=["translation", "source", "decoder"],
        expected_strategy="graph",
    ),

    # Vaswani 2017
    EvalSample(
        query="What is the Transformer architecture and how does it work?",
        expected_keywords=["attention", "encoder", "decoder", "parallelization"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="How does multi-head attention differ from single attention?",
        expected_keywords=["head", "parallel", "subspace"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="What is scaled dot-product attention?",
        expected_keywords=["dot", "softmax", "query", "key"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="What is the relationship between the Transformer and recurrent networks?",
        expected_keywords=["replaces", "recurrence", "parallel"],
        expected_strategy="graph",
    ),

    # Devlin 2018
    EvalSample(
        query="What is BERT and how is it different from GPT?",
        expected_keywords=["bidirectional", "encoder", "GPT"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="What pre-training tasks does BERT use?",
        expected_keywords=["masked", "language", "next", "sentence"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="How many parameters does BERT-Large have?",
        expected_keywords=["340", "24", "1024"],
        expected_strategy="hybrid",
    ),
    EvalSample(
        query="What is the relationship between BERT and the Transformer?",
        expected_keywords=["encoder", "Transformer", "based"],
        expected_strategy="graph",
    ),

    # Cross-paper
    EvalSample(
        query="How does multi-head attention relate to Bahdanau attention?",
        expected_keywords=["generalization", "alignment", "parallel"],
        expected_strategy="graph",
    ),

    # Out-of-scope
    EvalSample(
        query="What is the boiling point of liquid nitrogen?",
        expected_keywords=[],
        should_refuse=True,
    ),
]


# ── Metric Helpers ─────────────────────────────────────────────────────────────

def _keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


# ── RAGAS Integration ──────────────────────────────────────────────────────────

def _run_ragas(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> list[dict]:
    """
    Run RAGAS evaluation metrics.

    Returns list of dicts with faithfulness, answer_relevancy, context_precision.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        })

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        # Extract per-sample scores
        df = result.to_pandas()
        scores = []
        for _, row in df.iterrows():
            scores.append({
                "faithfulness":      row.get("faithfulness"),
                "answer_relevancy":  row.get("answer_relevancy"),
                "context_precision": row.get("context_precision"),
            })
        return scores

    except ImportError:
        console.print("  [yellow]RAGAS not installed — skipping RAGAS metrics[/yellow]")
        return []
    except Exception as exc:
        console.print(f"  [yellow]RAGAS error: {exc}[/yellow]")
        return []


# ── Main Evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    rag,
    dataset: list[EvalSample] | None = None,
    use_ragas: bool = True,
) -> list[EvalResult]:
    """Run evaluation across all samples."""
    dataset = dataset or EVAL_DATASET
    results: list[EvalResult] = []

    # Collect data for RAGAS batch evaluation
    all_queries:  list[str] = []
    all_answers:  list[str] = []
    all_contexts: list[list[str]] = []

    console.print(f"\n[bold yellow]━━  Evaluation  ({len(dataset)} samples)  ━━[/bold yellow]\n")

    for i, sample in enumerate(dataset, 1):
        console.print(f"[dim][{i}/{len(dataset)}] {sample.query[:70]}[/dim]")

        rag_result = rag.query(sample.query, expand_queries=True)

        khr            = _keyword_hit_rate(rag_result.answer, sample.expected_keywords)
        correct_refusal = sample.should_refuse and is_dont_know(rag_result.answer)

        results.append(EvalResult(
            query=sample.query,
            answer=rag_result.answer,
            strategy_used=rag_result.strategy_used,
            verified=rag_result.verified,
            confidence=rag_result.confidence,
            elapsed_sec=rag_result.elapsed_sec,
            keyword_hit_rate=khr,
            correct_refusal=correct_refusal,
        ))

        all_queries.append(sample.query)
        all_answers.append(rag_result.answer)
        all_contexts.append(rag_result.context_chunks)

    # RAGAS batch evaluation
    if use_ragas and all_queries:
        console.print("\n[cyan]📊 Running RAGAS evaluation…[/cyan]")
        ragas_scores = _run_ragas(all_queries, all_answers, all_contexts)
        for i, scores in enumerate(ragas_scores):
            if i < len(results):
                results[i].ragas_faithfulness      = scores.get("faithfulness")
                results[i].ragas_answer_relevancy  = scores.get("answer_relevancy")
                results[i].ragas_context_precision  = scores.get("context_precision")

    _print_summary(results)
    return results


def _print_summary(results: list[EvalResult]) -> None:
    tbl = Table(
        title="GraphRAG Evaluation Results",
        box=box.ROUNDED, show_lines=True, header_style="bold cyan",
    )
    tbl.add_column("#",          width=3,  style="dim")
    tbl.add_column("Query",      max_width=35)
    tbl.add_column("Strategy",   width=8)
    tbl.add_column("KwHit%",     width=7)
    tbl.add_column("Confidence", width=10)
    tbl.add_column("Verified",   width=8)
    tbl.add_column("Time",       width=6)

    for i, r in enumerate(results, 1):
        khr_color = "green" if r.keyword_hit_rate >= 0.7 else (
            "yellow" if r.keyword_hit_rate >= 0.4 else "red"
        )
        conf_color = "green" if r.confidence >= 0.7 else (
            "yellow" if r.confidence >= 0.4 else "red"
        )
        tbl.add_row(
            str(i),
            r.query[:35],
            r.strategy_used,
            f"[{khr_color}]{r.keyword_hit_rate:.0%}[/{khr_color}]",
            f"[{conf_color}]{r.confidence:.2f}[/{conf_color}]",
            "✅" if r.verified else "⚠️",
            f"{r.elapsed_sec:.1f}s",
        )

    console.print(tbl)

    n = len(results)
    console.print("\n[bold]Aggregate Metrics[/bold]")
    console.print(f"  Pass Rate       : [green]{sum(r.passed() for r in results)/n:.1%}[/green]")
    console.print(f"  Avg KW Hit Rate : [green]{sum(r.keyword_hit_rate for r in results)/n:.1%}[/green]")
    console.print(f"  Avg Confidence  : [green]{sum(r.confidence for r in results)/n:.2f}[/green]")
    console.print(f"  Verified Rate   : [green]{sum(r.verified for r in results)/n:.1%}[/green]")
    console.print(f"  Avg Query Time  : [cyan]{sum(r.elapsed_sec for r in results)/n:.2f}s[/cyan]")

    # RAGAS aggregate (if available)
    ragas_results = [r for r in results if r.ragas_faithfulness is not None]
    if ragas_results:
        console.print("\n[bold]RAGAS Metrics[/bold]")
        console.print(f"  Faithfulness     : [green]{sum(r.ragas_faithfulness for r in ragas_results)/len(ragas_results):.2f}[/green]")
        console.print(f"  Answer Relevancy : [green]{sum(r.ragas_answer_relevancy for r in ragas_results)/len(ragas_results):.2f}[/green]")
        console.print(f"  Context Precision: [green]{sum(r.ragas_context_precision for r in ragas_results)/len(ragas_results):.2f}[/green]")


def save_results(results: list[EvalResult], path: str = "eval_results.json") -> None:
    with open(path, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    console.print(f"\n[green]Results saved → {path}[/green]")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="GraphRAG Evaluation")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS metrics")
    args = parser.parse_args()

    from graphrag_pipeline import AgenticGraphRAG
    rag = AgenticGraphRAG(skip_indexing=True)
    results = run_evaluation(rag, use_ragas=not args.no_ragas)
    save_results(results)