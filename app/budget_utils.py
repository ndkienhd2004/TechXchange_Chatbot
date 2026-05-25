from __future__ import annotations

import re

from app.text_normalization import is_build_pc_intent, normalize, strip_accents


def parse_budget_vnd(message: str) -> int | None:
    raw = strip_accents(str(message or "")).lower().strip()
    if not raw:
        return None

    million_match = re.search(r"(?:^|[^\d])(\d+(?:[.,]\d+)?)\s*(trieu|tr|m)\b", raw)
    if million_match:
        value = float(million_match.group(1).replace(",", "."))
        return int(round(value * 1_000_000))

    billion_match = re.search(r"(?:^|[^\d])(\d+(?:[.,]\d+)?)\s*(ty|ti)\b", raw)
    if billion_match:
        value = float(billion_match.group(1).replace(",", "."))
        return int(round(value * 1_000_000_000))

    plain_match = re.search(r"(\d[\d.,]{5,})", raw)
    if plain_match:
        text = plain_match.group(1).replace(",", "").replace(".", "")
        if text.isdigit():
            parsed = int(text)
            if parsed >= 1_000_000:
                return parsed
    return None


def parse_budget_request(message: str) -> dict | None:
    raw = strip_accents(str(message or "")).lower().strip()
    if raw:
        compact_range_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:-|den|toi|to)\s*(\d+(?:[.,]\d+)?)\s*(trieu|tr|m)\b",
            raw,
        )
        if compact_range_match:
            left = float(compact_range_match.group(1).replace(",", "."))
            right = float(compact_range_match.group(2).replace(",", "."))
            min_m, max_m = sorted([left, right])
            min_budget = int(round(min_m * 1_000_000))
            max_budget = int(round(max_m * 1_000_000))
            return {
                "budget_vnd": max_budget,
                "mode": "range",
                "min_budget": min_budget,
                "max_budget": max_budget,
            }

        has_range_sep = any(sep in raw for sep in ("-", " den ", " toi ", " to "))
        million_tokens = re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:trieu|tr|m)\b", raw)
        if has_range_sep and len(million_tokens) >= 2:
            left = float(million_tokens[0].replace(",", "."))
            right = float(million_tokens[1].replace(",", "."))
            min_m, max_m = sorted([left, right])
            min_budget = int(round(min_m * 1_000_000))
            max_budget = int(round(max_m * 1_000_000))
            return {
                "budget_vnd": max_budget,
                "mode": "range",
                "min_budget": min_budget,
                "max_budget": max_budget,
            }

    budget_vnd = parse_budget_vnd(message)
    if not budget_vnd:
        return None

    normalized = normalize(message)
    upper_markers = ["duoi", "toi da", "khong qua", "under", "max", "den"]
    lower_markers = ["tren", "toi thieu", "at least", "min"]

    mode = "target"
    if any(marker in normalized for marker in upper_markers):
        mode = "upper"
    elif any(marker in normalized for marker in lower_markers):
        mode = "lower"

    if mode == "upper":
        min_budget = 0
        max_budget = budget_vnd
    elif mode == "lower":
        min_budget = budget_vnd
        max_budget = int(round(budget_vnd * 1.3))
    else:
        min_budget = int(round(budget_vnd * 0.95))
        max_budget = int(round(budget_vnd * 1.05))

    return {
        "budget_vnd": budget_vnd,
        "mode": mode,
        "min_budget": min_budget,
        "max_budget": max_budget,
    }


def has_recent_build_pc_context(message: str, history: list[dict]) -> bool:
    current = normalize(message)
    checked = 0
    skipped_current = False

    for item in reversed(history):
        if item.get("role") != "user":
            continue
        text = normalize(str(item.get("content") or ""))
        if not text:
            continue
        if not skipped_current and text == current:
            skipped_current = True
            continue
        checked += 1
        if is_build_pc_intent(text) or "linh kien" in text or "cpu" in text or "gpu" in text:
            return True
        if checked >= 4:
            break

    return False


def should_use_build_pc_route(message: str, history: list[dict]) -> tuple[bool, dict | None]:
    budget_request = parse_budget_request(message)
    if is_build_pc_intent(message):
        return True, budget_request
    if budget_request and has_recent_build_pc_context(message, history):
        return True, budget_request
    return False, budget_request


def format_vnd(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")
