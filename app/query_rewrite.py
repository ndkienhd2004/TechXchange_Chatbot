from __future__ import annotations

"""Query rewriting utilities to improve retrieval recall."""

import unicodedata
from typing import Optional


def _normalize(text: str) -> str:
    """Lowercase and keep alphanumeric tokens with collapsed spaces."""

    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def _strip_accents(text: str) -> str:
    """Remove Vietnamese accents so matching works for both typed forms."""

    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def rewrite_query(message: str, history: Optional[list[dict[str, str]]] = None) -> list[str]:
    """Generate up to 4 deduplicated query variants for hybrid retrieval."""

    clean = _normalize(message)
    if not clean:
        return []
    clean_no_accent = _normalize(_strip_accents(clean))

    rewrites: list[str] = [clean]
    if clean_no_accent and clean_no_accent != clean:
        rewrites.append(clean_no_accent)

    normalized_for_intent = clean_no_accent or clean

    if "build" in normalized_for_intent or "pc" in normalized_for_intent:
        rewrites.append(f"{clean} cau hinh pc")
        rewrites.append(f"{clean} gpu cpu ram ssd")
        if clean_no_accent and clean_no_accent != clean:
            rewrites.append(f"{clean_no_accent} cau hinh pc")
            rewrites.append(f"{clean_no_accent} gpu cpu ram ssd")
    if "bao hanh" in normalized_for_intent:
        rewrites.append("chinh sach bao hanh dieu kien")
    if "doi tra" in normalized_for_intent:
        rewrites.append("chinh sach doi tra san pham")
    if "hoan" in normalized_for_intent and (
        "tien" in normalized_for_intent or "hang" in normalized_for_intent
    ):
        rewrites.append("chinh sach doi tra va hoan tien")
    if "thanh toan" in normalized_for_intent or "phuong thuc" in normalized_for_intent:
        rewrites.append("chinh sach thanh toan")
        rewrites.append("phuong thuc thanh toan cod chuyen khoan online")

    if history:
        for item in reversed(history):
            if item.get("role") == "user" and item.get("content"):
                prior = _normalize(item["content"])
                prior_no_accent = _normalize(_strip_accents(prior))
                if prior and prior != clean:
                    rewrites.append(f"{prior} {clean}")
                if prior_no_accent and prior_no_accent != clean_no_accent:
                    rewrites.append(f"{prior_no_accent} {clean_no_accent}")
                break

    deduped: list[str] = []
    for item in rewrites:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]
