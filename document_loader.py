"""
document_loader.py — Phase 1: Document Parsing with PyPDF + Unstructured.io

Production parsing pipeline:
  1. PyMuPDF (fitz) — fast, reliable PDF text extraction
  2. Unstructured.io — handles tables, headers, complex layouts
  3. Plain text fallback — .txt / .md files

Unstructured.io is tried first for PDFs; if unavailable, falls back to PyMuPDF.
All extracted text passes through a garbage filter to remove watermarks,
page numbers, and arXiv stamps before chunking.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import config
from utils import console, get_logger

logger = get_logger(__name__)

# ── Garbage patterns ───────────────────────────────────────────────────────────
_GARBAGE: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"arXiv:\d{4}\.\d{4,5}v\d+",
        r"Preprint\.\s*Do not distribute",
        r"Page\s+\d+\s+of\s+\d+",
        r"©\s*\d{4}",
        r"All rights reserved",
        r"www\.\S+\.(com|org|net)",
        r"^\s*\d+\s*$",
        r"^\.{5,}$",
        r"^\s*[-_=]{5,}\s*$",
        r"^\s*[ivxlc]+\s*$",
        r"Under review",
        r"Anonymous authors",
    ]
]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single text chunk with full provenance metadata."""
    text:     str
    source:   str = "unknown"
    rel_path: str = ""
    page:     int = 0
    chunk_id: int = 0

    @property
    def metadata(self) -> dict:
        return {
            "source":   self.source,
            "rel_path": self.rel_path,
            "page":     self.page,
            "chunk_id": self.chunk_id,
        }

    def __repr__(self) -> str:
        return (
            f"Document(src={self.source!r}, page={self.page}, "
            f"chunk={self.chunk_id}, len={len(self.text)})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_garbage(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 30:
        return True
    return any(p.search(stripped) for p in _GARBAGE)


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)


# ── PDF Loaders ────────────────────────────────────────────────────────────────

def _load_pdf_unstructured(path: str) -> str:
    """
    Load PDF using Unstructured.io — handles tables, headers, complex layouts.
    Returns full document text as a single string.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=path, strategy="fast")
        paragraphs = [str(el).strip() for el in elements if str(el).strip()]
        clean = [p for p in paragraphs if not _is_garbage(p)]
        return "\n\n".join(clean)
    except ImportError:
        logger.debug("unstructured not available, falling back to PyMuPDF")
        return ""
    except Exception as exc:
        logger.warning(f"Unstructured failed for {path}: {exc}, falling back to PyMuPDF")
        return ""


def _load_pdf_pymupdf(path: str) -> str:
    """
    Load PDF using PyMuPDF (fitz) — fast, reliable text extraction.
    Returns full document text as a single string.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    all_text = []
    with fitz.open(path) as pdf:
        for page in pdf:
            blocks = page.get_text("blocks")
            clean_blocks = [
                b[4].strip()
                for b in sorted(blocks, key=lambda b: (b[1], b[0]))
                if b[6] == 0 and not _is_garbage(b[4])
            ]
            page_text = " ".join(clean_blocks)
            if page_text.strip():
                all_text.append(page_text)

    return "\n\n".join(all_text)


def _load_pdf(path: str) -> str:
    """
    Load PDF: try Unstructured.io first, fall back to PyMuPDF.
    Returns the full cleaned text for chunking.
    """
    text = _load_pdf_unstructured(path)
    if not text:
        text = _load_pdf_pymupdf(path)
    return text


def _load_text_file(path: str) -> str:
    """Load plain text / markdown file."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    paragraphs = re.split(r"\n{2,}", raw)
    clean = "\n\n".join(p.strip() for p in paragraphs if not _is_garbage(p))
    return clean


# ── File discovery ─────────────────────────────────────────────────────────────

_SUPPORTED = {".txt", ".md", ".pdf"}


def _discover_files(root: str, recursive: bool) -> list[str]:
    """Return sorted list of all supported files under root."""
    paths: list[str] = []
    if recursive:
        for dirpath, _dirs, filenames in os.walk(root):
            for fname in sorted(filenames):
                if os.path.splitext(fname)[1].lower() in _SUPPORTED:
                    paths.append(os.path.join(dirpath, fname))
    else:
        for fname in sorted(os.listdir(root)):
            full = os.path.join(root, fname)
            if os.path.isfile(full) and os.path.splitext(fname)[1].lower() in _SUPPORTED:
                paths.append(full)
    return paths


# ── Public API ─────────────────────────────────────────────────────────────────

def load_raw_texts(
    docs_dir: str = None,
    recursive: bool = None,
) -> list[tuple[str, str, str]]:
    """
    Load raw text from all documents (NO chunking).

    Returns:
        List of (raw_text, filename, rel_path) tuples.
    """
    docs_dir  = docs_dir  or config.DOCS_DIR
    recursive = recursive if recursive is not None else config.SCAN_SUBDIRS

    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir!r}")

    files = _discover_files(docs_dir, recursive)
    if not files:
        console.print(f"[yellow]⚠  No documents found under {docs_dir!r}[/yellow]")
        return []

    console.print(f"[cyan]📥 Loading {len(files)} file(s) from {docs_dir}[/cyan]")
    results: list[tuple[str, str, str]] = []

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                text = _load_pdf(path)
            else:
                text = _load_text_file(path)

            if text.strip():
                basename = os.path.basename(path)
                rel = _rel(path, docs_dir)
                results.append((text, basename, rel))
                console.print(
                    f"  [green]✓[/green] {rel:50s} ({len(text):,} chars)"
                )
            else:
                console.print(
                    f"  [yellow]⚠[/yellow] {os.path.basename(path)} — empty after cleaning"
                )
        except Exception as exc:
            console.print(f"  [red]✗[/red] {os.path.basename(path)}: {exc}")

    console.print(f"  [bold green]Loaded {len(results)} document(s)[/bold green]\n")
    return results
