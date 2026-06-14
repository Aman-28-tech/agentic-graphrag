"""
ingest.py — End-to-End Ingestion Pipeline (Production)

Phases:
  0. Download research PDFs → docs/ai_papers/
  1. Parse documents (PyPDF + Unstructured.io)
  2. Chunk text (LangChain RecursiveCharacterTextSplitter)
  3. Embed chunks (SentenceTransformers)
  4. Store in Qdrant (local or server mode)
  5. Build & persist BM25 index
  6. Build Knowledge Graph (NetworkX)
  7. Smoke-test all retrieval paths

Usage:
  python ingest.py                    # full pipeline
  python ingest.py --skip-download    # use existing PDFs
  python ingest.py --force-reindex    # wipe and rebuild
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.panel import Panel
from rich.rule  import Rule
from rich.table import Table
from rich       import box

import config
from utils            import console, timer
from document_loader  import load_raw_texts
from chunker          import chunk_documents, Document
from vector_store     import VectorStore
from bm25_retriever   import BM25Retriever
from knowledge_graph  import KnowledgeGraph
from reranker         import Reranker
from retriever        import MultiStrategyRetriever


# ── Phase 0: Download papers ──────────────────────────────────────────────────

PAPERS = [
    ("attention_is_all_you_need.pdf", "1706.03762",
     "Attention Is All You Need — Vaswani et al., 2017"),
    ("bert_paper.pdf", "1810.04805",
     "BERT — Devlin et al., 2018"),
    ("gpt3_paper.pdf", "2005.14165",
     "GPT-3 — Brown et al., 2020"),
    ("bahdanau_attention.pdf", "1409.0473",
     "Bahdanau Attention — Bahdanau et al., 2014"),
]


def phase_download(dest_dir: str, force: bool = False) -> list[str]:
    """Download PDFs from arXiv."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    downloaded: list[str] = []

    console.print(Rule("[bold cyan]Phase 0 · Download Research Papers[/bold cyan]"))

    for filename, arxiv_id, description in PAPERS:
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path) and not force:
            console.print(f"  [dim]Skip[/dim]  {filename} (exists)")
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        console.print(f"  [yellow]↓[/yellow]  {filename} (arXiv:{arxiv_id})")
        try:
            urllib.request.urlretrieve(url, dest_path)
            with open(dest_path, "rb") as fh:
                if fh.read(5) != b"%PDF-":
                    os.remove(dest_path)
                    console.print(f"  [red]✗[/red]  {filename} — invalid PDF")
                    continue
            size_kb = os.path.getsize(dest_path) / 1024
            console.print(f"  [green]✓[/green]  {filename} ({size_kb:.0f} KB)")
            downloaded.append(dest_path)
        except Exception as exc:
            console.print(f"  [red]✗[/red]  {filename}: {exc}")

    console.print()
    return downloaded


# ── Phase 1-2: Load + Chunk ───────────────────────────────────────────────────

def phase_load_and_chunk() -> list[Document]:
    console.print(Rule("[bold cyan]Phase 1-2 · Parse & Chunk Documents[/bold cyan]"))
    with timer("Document parsing"):
        raw_texts = load_raw_texts(config.DOCS_DIR, recursive=True)
    if not raw_texts:
        return []
    with timer("Chunking"):
        docs = chunk_documents(raw_texts)
    return docs


# ── Phase 3-4: Embed + Store ──────────────────────────────────────────────────

def phase_embed_store(docs: list[Document], force: bool) -> VectorStore:
    console.print(Rule("[bold cyan]Phase 3-4 · Embed & Store (Qdrant)[/bold cyan]"))
    vs = VectorStore()
    if force and not vs.is_empty():
        vs.delete_collection()
    if vs.is_empty():
        with timer("Embedding + upload"):
            vs.index_documents(docs, batch_size=32)
    else:
        count = vs.client.get_collection(config.QDRANT_COLLECTION).points_count
        console.print(f"  [dim]Qdrant has {count} vectors — skip[/dim]\n")
    return vs


# ── Phase 5: BM25 ─────────────────────────────────────────────────────────────

def phase_bm25(vs: VectorStore, force: bool) -> BM25Retriever:
    console.print(Rule("[bold cyan]Phase 5 · BM25 Keyword Index[/bold cyan]"))
    if force and os.path.exists(config.BM25_INDEX_PATH):
        os.remove(config.BM25_INDEX_PATH)
    corpus = vs.get_all_texts()
    return BM25Retriever.load_or_build(corpus, config.BM25_INDEX_PATH)


# ── Phase 6: Knowledge Graph ──────────────────────────────────────────────────

def phase_kg(chunks: list[str] | None = None) -> KnowledgeGraph:
    console.print(Rule("[bold cyan]Phase 6 · Knowledge Graph[/bold cyan]"))
    with timer("Graph construction"):
        kg = KnowledgeGraph(chunks=chunks)
        kg.save()
        return kg


# ── Phase 7: Smoke Test ───────────────────────────────────────────────────────

def phase_smoketest(vs, bm25, kg):
    console.print(Rule("[bold cyan]Phase 7 · Smoke Test[/bold cyan]"))
    reranker  = Reranker()
    retriever = MultiStrategyRetriever(vs, kg, reranker)
    retriever._bm25 = bm25

    tests = [
        ("graph",  "What is the relationship between BERT and the Transformer?"),
        ("hybrid", "How does multi-head attention work?"),
        ("bm25",   "Bahdanau alignment model"),
    ]
    for strategy, query in tests:
        results = retriever.retrieve(query, strategy)
        status  = "✅" if results else "❌"
        console.print(f"  {status} [{strategy:6s}] {query:50s} → {len(results)} chunks")
    console.print()


# ── Report ─────────────────────────────────────────────────────────────────────

def report(docs, vs, kg, elapsed):
    console.print(Rule("[bold green]Ingestion Complete[/bold green]"))
    from collections import Counter
    counts = Counter(d.source for d in docs)

    tbl = Table(title="Chunks by Source", box=box.ROUNDED, header_style="bold cyan")
    tbl.add_column("File", max_width=50)
    tbl.add_column("Chunks", justify="right")
    for src, n in sorted(counts.items(), key=lambda x: -x[1]):
        tbl.add_row(src, str(n))
    tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{sum(counts.values())}[/bold]")
    console.print(tbl)

    qdrant_count = vs.client.get_collection(config.QDRANT_COLLECTION).points_count
    console.print(f"\n  Qdrant vectors : {qdrant_count}")
    console.print(f"  KG nodes/edges : {kg.graph.number_of_nodes()} / {kg.graph.number_of_edges()}")
    console.print(f"  Total time     : {elapsed:.1f}s\n")

    console.print(Panel(
        "[bold green]Ready![/bold green]\n\n"
        "  [cyan]python api.py[/cyan]         — FastAPI backend\n"
        "  [cyan]streamlit run app.py[/cyan]   — Streamlit UI\n"
        "  [cyan]python demo.py[/cyan]         — CLI demo\n"
        "  [cyan]python main.py -i[/cyan]      — Interactive REPL",
        border_style="green", padding=(0, 2),
    ))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GraphRAG Ingestion Pipeline")
    p.add_argument("--skip-download",  action="store_true")
    p.add_argument("--force-reindex",  action="store_true")
    p.add_argument("--no-smoketest",   action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()

    console.print(Panel(
        "[bold cyan]🚀 GraphRAG Ingestion Pipeline — Production[/bold cyan]\n"
        "[dim]PyPDF → LangChain Chunks → Qdrant → BM25 → KG → Re-ranker[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    if not args.skip_download:
        phase_download(config.DOCS_SUBDIR, force=args.force_reindex)

    docs = phase_load_and_chunk()
    if not docs:
        console.print("[red]✗ No documents — aborting[/red]")
        sys.exit(1)

    vs   = phase_embed_store(docs, args.force_reindex)
    bm25 = phase_bm25(vs, args.force_reindex)

    # Pass chunk texts to KG for auto entity/relation extraction
    chunk_texts = [d.text for d in docs]
    kg   = phase_kg(chunks=chunk_texts)

    if not args.no_smoketest:
        phase_smoketest(vs, bm25, kg)

    report(docs, vs, kg, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
