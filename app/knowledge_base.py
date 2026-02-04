from __future__ import annotations

"""Text chunking utility for knowledge ingestion."""


def chunk_text(content: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split long text into overlapping chunks for embedding/indexing."""

    text = str(content or "").strip()
    if not text:
        return []

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return parts
