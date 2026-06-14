"""
chunker.py — Phase 2: LangChain RecursiveCharacterTextSplitter

Splits raw document text into overlapping chunks that preserve context.
Uses LangChain's RecursiveCharacterTextSplitter which intelligently splits
on paragraph → sentence → word boundaries in that priority order.

Returns Document dataclass instances with full provenance metadata.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from document_loader import Document
from utils import console, get_logger

logger = get_logger(__name__)


def create_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a LangChain text splitter with configured parameters.

    The separator hierarchy ensures clean splits:
      1. Double newline (paragraph boundary)
      2. Single newline
      3. Period + space (sentence boundary)
      4. Space (word boundary)
      5. Empty string (character-level fallback)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    raw_texts: list[tuple[str, str, str]],
    chunk_size: int = None,
    chunk_overlap: int = None,
    min_length: int = None,
) -> list[Document]:
    """
    Split raw document texts into overlapping chunks.

    Args:
        raw_texts:     List of (raw_text, filename, rel_path) from document_loader.
        chunk_size:    Characters per chunk (default: config.CHUNK_SIZE).
        chunk_overlap: Overlap between chunks (default: config.CHUNK_OVERLAP).
        min_length:    Discard chunks shorter than this (default: config.MIN_CHUNK_LENGTH).

    Returns:
        List of Document dataclass instances with metadata.
    """
    min_length = min_length or config.MIN_CHUNK_LENGTH
    splitter   = create_splitter(chunk_size, chunk_overlap)

    console.print(
        f"[cyan]✂️  Chunking with LangChain RecursiveCharacterTextSplitter[/cyan]"
        f"\n  [dim]chunk_size={splitter._chunk_size}, "
        f"overlap={splitter._chunk_overlap}[/dim]"
    )

    all_docs: list[Document] = []

    for raw_text, filename, rel_path in raw_texts:
        chunks = splitter.split_text(raw_text)

        # Filter short/garbage chunks
        valid_chunks = [c for c in chunks if len(c.strip()) >= min_length]

        for i, chunk_text in enumerate(valid_chunks):
            all_docs.append(Document(
                text=chunk_text.strip(),
                source=filename,
                rel_path=rel_path,
                page=0,   # page-level tracking lost after full-text extraction
                chunk_id=i,
            ))

        console.print(
            f"  [green]✓[/green] {rel_path:50s} → "
            f"[bold]{len(valid_chunks):4d}[/bold] chunks"
        )

    console.print(
        f"\n  [bold green]Total: {len(all_docs)} chunks "
        f"from {len(raw_texts)} file(s)[/bold green]\n"
    )
    return all_docs
