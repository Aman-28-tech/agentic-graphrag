"""
demo.py — Quick-start demo script

  python demo.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.rule  import Rule
from rich.panel import Panel
from utils             import console
from graphrag_pipeline import AgenticGraphRAG

QUERIES = [
    ("Graph  ", "What is the relationship between the attention mechanism and machine translation?"),
    ("Hybrid ", "How does multi-head attention work in the Transformer architecture?"),
    ("BM25   ", "Vaswani Transformer authors"),
    ("Hybrid ", "What pre-training objectives does BERT use and why?"),
    ("Graph  ", "What is the relationship between the Transformer and BERT?"),
    ("Graph  ", "How does multi-head attention relate to the Bahdanau attention mechanism?"),
    ("Refusal", "What is the boiling point of liquid nitrogen?"),
]


def main() -> None:
    console.print(Panel(
        "[bold cyan]Agentic GraphRAG — Production Demo[/bold cyan]\n"
        "[dim]7 queries · GPT-4o/Claude · Cross-Encoder · Caching[/dim]",
        border_style="cyan", padding=(1, 4),
    ))

    rag = AgenticGraphRAG(skip_indexing=True)

    for i, (label, query) in enumerate(QUERIES, 1):
        console.print(Rule(f"[bold yellow]Query {i}/7  [{label.strip()}][/bold yellow]"))
        result = rag.query(query, expand_queries=True)
        result.display()

    console.print(Rule("[bold green]Demo complete[/bold green]"))
    console.print(f"\n  Cache stats: {rag.cache.stats}")


if __name__ == "__main__":
    main()
