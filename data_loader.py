from __future__ import annotations

"""Utilities to load manual JSON docs and preview chunking behavior."""

import json
from pathlib import Path
from typing import Any, Union

from app.knowledge_base import chunk_text


def load_documents_from_json(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read one JSON file and normalize entries to ingestion document shape."""

    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]

    documents: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        source_key = str(item.get("source_key", f"manual:{index}"))
        title = str(item.get("title", f"Doc {index}"))
        uri = str(item.get("uri", ""))
        content = str(item.get("content", "")).strip()
        metadata = dict(item.get("metadata", {}))
        if not content:
            continue
        documents.append(
            {
                "source_key": source_key,
                "title": title,
                "uri": uri,
                "content": content,
                "metadata": metadata,
            }
        )
    return documents


def preview_chunking(content: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Expose chunking helper for quick local debugging/tests."""

    return chunk_text(content, chunk_size=chunk_size, overlap=overlap)
