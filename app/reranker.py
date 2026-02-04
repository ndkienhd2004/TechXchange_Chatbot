from __future__ import annotations

"""Heuristic reranker over retrieved KB chunks."""

from datetime import datetime, timezone
from typing import Optional


def _compute_freshness_score(updated_at: Optional[str]) -> float:
    """Convert updated timestamp to a bounded freshness score."""

    if not updated_at:
        return 0.5
    try:
        ts = datetime.fromisoformat(updated_at).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5

    age_days = max((datetime.now(timezone.utc) - ts).days, 0)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.8
    if age_days <= 90:
        return 0.6
    return 0.4


def rerank(query: str, candidates: list[dict], topk: int) -> list[dict]:
    """Blend retrieval score, trust score, freshness, and intent boost."""

    lowered = query.lower()
    rows: list[dict] = []

    for item in candidates:
        trust = float(item.get("metadata", {}).get("trust_score", 0.7))
        fresh = _compute_freshness_score(item.get("metadata", {}).get("updated_at"))
        retrieval = float(item.get("retrieval_score", 0.0))
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        intent_boost = 0.08 if "build" in lowered and "build" in text else 0.0
        rerank_score = retrieval * 0.6 + trust * 0.25 + fresh * 0.15 + intent_boost
        rows.append(
            {
                **item,
                "trust_score": trust,
                "freshness_score": fresh,
                "rerank_score": rerank_score,
            }
        )

    return sorted(rows, key=lambda item: item["rerank_score"], reverse=True)[: max(1, topk)]
