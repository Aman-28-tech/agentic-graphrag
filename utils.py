"""
utils.py — Shared utilities: logging, text helpers, timing
Used across all modules to avoid duplication.
"""

from __future__ import annotations

import re
import time
import logging
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.logging import RichHandler

# ── Single shared console instance ────────────────────────────────────────────
console = Console()


# ── Structured logger (wraps Rich) ────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger that renders via Rich."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)


# ── Timing context manager ─────────────────────────────────────────────────────

@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that prints elapsed time after the block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    console.print(f"  [dim]{label}: {elapsed:.2f}s[/dim]")


# ── Text utilities ─────────────────────────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "by", "from", "as", "it", "its",
    "was", "are", "be", "been", "this", "that", "these", "those",
    "which", "who", "what", "how", "when", "where", "why", "not",
    "also", "can", "may", "will", "would", "could", "should",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lowercase tokenizer.
    Splits on non-alphanumeric characters and optionally removes stop words.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Truncate text to max_chars and append suffix if shortened."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def deduplicate(items: list[str]) -> list[str]:
    """Return list with duplicates removed, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def is_dont_know(text: str) -> bool:
    """
    Return True if the text is a known 'I don't know' response.

    Only matches when the refusal phrase appears at or near the START of the
    answer (within the first 80 chars), preventing false positives from
    prompt instruction text that may have leaked into the output.
    """
    # Check only the beginning of the answer (not embedded prompt text)
    # Take the last meaningful paragraph if multiple are present
    text = text.strip()

    # If the answer contains a clear "Answer:" marker, only check after it
    if "Answer (using only the context above):" in text:
        text = text.split("Answer (using only the context above):")[-1].strip()
    elif "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # Only check the first 120 characters for refusal signals
    prefix = text[:120].lower()
    return any(phrase in prefix for phrase in (
        "i don't know",
        "i do not know",
        "cannot answer",
        "no information",
        "not in the context",
        "not enough information",
    ))


def clean_refusal_answer(text: str) -> str:
    """
    If the answer starts with a refusal phrase, return ONLY that phrase.

    Prevents the model from appending explanations like:
      'I don't know based on the provided context.
       The context discusses scaling model sizes...'
    """
    text = text.strip()

    # Strip prompt leakage
    if "Answer (using only the context above):" in text:
        text = text.split("Answer (using only the context above):")[-1].strip()
    elif "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    refusal_phrases = (
        "i don't know based on the provided context",
        "i do not know based on the provided context",
        "i don't know",
        "i do not know",
        "cannot answer",
    )

    lower = text.lower()
    for phrase in refusal_phrases:
        if lower.startswith(phrase):
            # Return just the refusal sentence (up to first period/newline)
            end = text.find(".", len(phrase) - 1)
            if end != -1 and end < len(phrase) + 20:
                return text[:end + 1].strip()
            return "I don't know based on the provided context."

    return text
