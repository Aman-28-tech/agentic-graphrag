"""
main.py — CLI entry-point for Agentic GraphRAG (Production)

Modes:
  --interactive / -i  : REPL mode
  --demo       / -d   : Run demo queries
  --query TEXT / -q   : Single query
  --api               : Start FastAPI server
  --ui                : Start Streamlit UI
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.panel  import Panel
from rich.prompt import Prompt

from graphrag_pipeline import AgenticGraphRAG
from utils import console


DEMO_QUERIES = [
    ("Graph  ", "What is the relationship between the attention mechanism and machine translation?"),
    ("Hybrid ", "How does multi-head attention work in the Transformer architecture?"),
    ("BM25   ", "Vaswani Transformer authors"),
    ("Hybrid ", "What pre-training objectives does BERT use and why?"),
    ("Graph  ", "What is the relationship between the Transformer and BERT?"),
    ("Graph  ", "How does multi-head attention relate to the Bahdanau attention mechanism?"),
    ("Refusal", "What is the boiling point of liquid nitrogen?"),
]


def run_single(rag, query, expand):
    result = rag.query(query, expand_queries=expand)
    result.display()


def run_demo(rag, expand):
    console.print(Panel(
        f"[bold yellow]Demo Mode[/bold yellow] — {len(DEMO_QUERIES)} queries",
        border_style="yellow",
    ))
    for i, (label, query) in enumerate(DEMO_QUERIES, 1):
        console.print(f"\n[bold yellow]Demo {i}/{len(DEMO_QUERIES)} [{label.strip()}][/bold yellow]")
        result = rag.query(query, expand_queries=expand)
        result.display()


def run_interactive(rag, expand):
    console.print(Panel(
        "[bold]Interactive Mode[/bold]\n"
        "[dim]Commands: demo · quit · stats[/dim]",
        border_style="blue",
    ))
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Query[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "demo":
            run_demo(rag, expand)
            continue
        if query.lower() == "stats":
            console.print(f"  Cache: {rag.cache.stats}")
            continue
        result = rag.query(query, expand_queries=expand)
        result.display()


def main():
    p = argparse.ArgumentParser(description="Agentic GraphRAG CLI")
    p.add_argument("-q", "--query",       type=str)
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("-d", "--demo",       action="store_true")
    p.add_argument("--api",              action="store_true", help="Start FastAPI server")
    p.add_argument("--ui",              action="store_true", help="Start Streamlit UI")
    p.add_argument("--no-expand",        action="store_true")
    p.add_argument("--skip-indexing",    action="store_true")
    args = p.parse_args()

    expand = not args.no_expand

    if args.api:
        import uvicorn
        import config
        uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=True)
        return

    if args.ui:
        import config
        os.system(f"streamlit run {os.path.join(os.path.dirname(__file__), 'app.py')} --server.port {config.STREAMLIT_PORT}")
        return

    if not args.query and not args.demo:
        args.interactive = True

    rag = AgenticGraphRAG(skip_indexing=args.skip_indexing)

    if args.query:
        run_single(rag, args.query, expand)
    elif args.demo:
        run_demo(rag, expand)
    else:
        run_interactive(rag, expand)


if __name__ == "__main__":
    main()
