from __future__ import annotations

"""Query rewriting utilities to improve retrieval recall."""

import unicodedata
from typing import Optional

from app.intent_router import IntentRoute


def _normalize(text: str) -> str:
    """Lowercase and keep alphanumeric tokens with collapsed spaces."""

    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def _strip_accents(text: str) -> str:
    """Remove Vietnamese accents so matching works for both typed forms."""

    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")
    

def rewrite_query(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
    route: Optional[IntentRoute] = None,
) -> list[str]:
    """Generate up to 4 deduplicated query variants for hybrid retrieval."""

    clean = _normalize(message)
    if not clean:
        return []
    clean_no_accent = _normalize(_strip_accents(clean))

    rewrites: list[str] = [clean]
    if clean_no_accent and clean_no_accent != clean:
        rewrites.append(clean_no_accent)

    normalized_for_intent = clean_no_accent or clean
    intent = route.intent if route else "general"
    category = (route.category or "").strip().lower() if route else ""

    if intent == "build_pc":
        rewrites.append(f"{clean} cau hinh pc")
        rewrites.append(f"{clean} gpu cpu ram ssd")
        if clean_no_accent and clean_no_accent != clean:
            rewrites.append(f"{clean_no_accent} cau hinh pc")
            rewrites.append(f"{clean_no_accent} gpu cpu ram ssd")
    elif "build" in normalized_for_intent or "pc" in normalized_for_intent:
        rewrites.append(f"{clean} cau hinh pc")

    if intent == "policy" or "bao hanh" in normalized_for_intent:
        rewrites.append("chinh sach bao hanh dieu kien")
    if intent == "policy" or "doi tra" in normalized_for_intent:
        rewrites.append("chinh sach doi tra san pham")
    if "hoan" in normalized_for_intent and (
        "tien" in normalized_for_intent or "hang" in normalized_for_intent
    ):
        rewrites.append("chinh sach doi tra va hoan tien")
    if intent == "policy" or "thanh toan" in normalized_for_intent or "phuong thuc" in normalized_for_intent:
        rewrites.append("chinh sach thanh toan")
        rewrites.append("phuong thuc thanh toan cod chuyen khoan online")
    if intent == "top_selling":
        if category:
            rewrites.append(f"{category} ban chay")
            rewrites.append(f"top selling {category}")
        else:
            rewrites.append("san pham ban chay")
    if intent == "product_search":
        if category:
            rewrites.append(f"{category} gia tot")
            rewrites.append(f"{category} phu hop ngan sach")
        if route and route.budget_vnd:
            rewrites.append(f"{clean} {_normalize(str(route.budget_vnd))}")

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
