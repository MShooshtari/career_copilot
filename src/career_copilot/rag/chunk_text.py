"""Split long job descriptions into overlapping chunks for embedding + RAG."""

from __future__ import annotations

from career_copilot.constants import (
    MARKET_ANALYSIS_CHUNK_MAX_CHARS,
    MARKET_ANALYSIS_CHUNK_OVERLAP_CHARS,
)


def chunk_description(
    text: str,
    *,
    max_chars: int = MARKET_ANALYSIS_CHUNK_MAX_CHARS,
    overlap: int = MARKET_ANALYSIS_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    if not text or not text.strip():
        return []
    body = text.strip()
    max_chars = max(200, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def append_hard_splits(piece: str) -> None:
        if len(piece) <= max_chars:
            chunks.append(piece)
            return
        start = 0
        while start < len(piece):
            end = min(start + max_chars, len(piece))
            chunks.append(piece[start:end].strip())
            if end >= len(piece):
                break
            start = max(0, end - overlap)

    for para in paragraphs:
        if len(para) > max_chars:
            flush_current()
            append_hard_splits(para)
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            if len(para) <= max_chars:
                current = para
            else:
                append_hard_splits(para)

    flush_current()
    return [c for c in chunks if c]
