from __future__ import annotations

"""Intent routing helpers backed by Vertex AI with lightweight rule fallback."""

import json
import re
from typing import Any
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import Field

from app.config import settings
from app.providers.vertex_client import build_vertex_client
from app.providers.vertex_client import require_vertex_configuration
from app.text_normalization import normalize as normalize_text

ALLOWED_INTENTS = {"build_pc", "top_selling", "product_search", "policy", "general"}
ALLOWED_BUDGET_MODES = {"target", "upper", "lower"}


class IntentRoute(BaseModel):
    """Structured route result used by orchestrator and query rewrite."""

    intent: str = "general"
    confidence: float = 0.0
    budget_vnd: Optional[int] = None
    budget_mode: Optional[str] = None
    category: Optional[str] = None
    normalized_query: str = ""
    source: str = "fallback"
    rationale: Optional[str] = None


EXPLICIT_NON_PC_CATEGORY_TOKENS = (
    "laptop",
    "notebook",
    "macbook",
    "dien thoai",
    "smartphone",
    "iphone",
    "ipad",
    "tablet",
)

TOP_SELLING_MARKERS = (
    "ban chay",
    "best seller",
    "bestseller",
    "top selling",
    "mua nhieu",
    "ban nhieu",
)


def _has_explicit_non_pc_category(normalized: str) -> bool:
    """Return True when the query clearly targets non-PC (portable / prebuilt) categories."""

    if any(token in normalized for token in EXPLICIT_NON_PC_CATEGORY_TOKENS):
        return True
    collapsed = normalized.replace(" ", "")
    # Split spellings / OCR-ish spacing: "lap top", hyphen stripped elsewhere
    for needle in (
        "laptop",
        "notebook",
        "macbook",
        "ipad",
        "tablet",
        "iphone",
        "smartphone",
        "dienthoai",
        "maytinhxachtay",
    ):
        if needle in collapsed:
            return True
    for phrase in ("may tinh xach tay", "dien thoai di dong", "may tinh bang"):
        if phrase in normalized:
            return True
    return False


def message_blocks_build_pc_history_fallback(message: str) -> bool:
    """True when the current user text targets a device category (e.g. laptop), not desktop build follow-up."""

    return _has_explicit_non_pc_category(normalize_text(message))


def _coarse_shopping_category_from_normalized(normalized: str) -> str | None:
    """Map normalized query text to coarse shopping category (laptop / phone / tablet)."""

    collapsed = normalized.replace(" ", "")
    if (
        "laptop" in normalized
        or "notebook" in normalized
        or "macbook" in normalized
        or "laptop" in collapsed
        or "notebook" in collapsed
        or "macbook" in collapsed
        or "maytinhxachtay" in collapsed
        or "may tinh xach tay" in normalized
    ):
        return "laptop"
    if (
        any(t in normalized for t in ("dien thoai", "smartphone", "iphone"))
        or "dienthoai" in collapsed
        or "iphone" in collapsed
    ):
        return "phone"
    if "ipad" in normalized or "tablet" in normalized or "ipad" in collapsed:
        return "tablet"
    return None


def derive_shopping_category_from_message(message: str) -> str | None:
    """Map user wording to coarse shopping category (laptop / phone / tablet)."""

    return _coarse_shopping_category_from_normalized(normalize_text(message))


def _derive_explicit_category(normalized: str) -> str | None:
    """Map explicit category tokens to a normalized category label."""

    return _coarse_shopping_category_from_normalized(normalized)


def _looks_top_selling_query(normalized: str) -> bool:
    """Return True when query wording explicitly asks for best-selling items."""

    return any(marker in normalized for marker in TOP_SELLING_MARKERS)


def parse_budget_vnd(message: str) -> int | None:
    """Extract budget in VND from common Vietnamese shorthand formats."""

    raw = normalize_text(message)
    if not raw:
        return None

    million_match = re.search(r"(?:^|[^\d])(\d+(?:[.,]\d+)?)\s*(trieu|tr|m|cu)\b", raw)
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


def build_budget_request(budget_vnd: int, mode: str = "target") -> dict[str, int | str]:
    """Build normalized budget range payload for build-PC business rules."""

    safe_budget = max(int(budget_vnd), 1)
    safe_mode = mode if mode in ALLOWED_BUDGET_MODES else "target"
    if safe_mode == "upper":
        min_budget = 0
        max_budget = safe_budget
    elif safe_mode == "lower":
        min_budget = safe_budget
        max_budget = int(round(safe_budget * 1.3))
    else:
        min_budget = int(round(safe_budget * 0.95))
        max_budget = int(round(safe_budget * 1.05))
    return {
        "budget_vnd": safe_budget,
        "mode": safe_mode,
        "min_budget": min_budget,
        "max_budget": max_budget,
    }


def _parse_budget_mode(message: str) -> str | None:
    """Infer user budget mode from common phrasing."""

    normalized = normalize_text(message)
    if not normalized:
        return None
    if any(marker in normalized for marker in ["duoi", "toi da", "khong qua", "under", "max", "den"]):
        return "upper"
    if any(marker in normalized for marker in ["tren", "toi thieu", "at least", "min"]):
        return "lower"
    return "target"


def _fallback_route(message: str, history: list[dict[str, Any]] | None = None) -> IntentRoute:
    """Resolve obvious intents when Vertex routing is unavailable or uncertain."""

    normalized = normalize_text(message)
    budget_vnd = parse_budget_vnd(message)
    budget_mode = _parse_budget_mode(message)

    has_explicit_non_pc_category = _has_explicit_non_pc_category(normalized)

    if any(phrase in normalized for phrase in ["bao hanh", "doi tra", "hoan tien", "thanh toan", "giao hang", "chinh sach"]):
        return IntentRoute(
            intent="policy",
            confidence=0.74,
            budget_vnd=budget_vnd,
            budget_mode=budget_mode,
            normalized_query=normalized,
            source="fallback",
            rationale="keyword_policy",
        )

    build_terms = ["pc", "may tinh", "linh kien", "cpu", "gpu", "mainboard", "ram", "ssd", "psu", "case"]
    build_actions = ["build", "lap", "cau hinh", "xay dung", "tu van", "de xuat"]
    if any(term in normalized for term in build_terms) and any(action in normalized for action in build_actions):
        return IntentRoute(
            intent="build_pc",
            confidence=0.78,
            budget_vnd=budget_vnd,
            budget_mode=budget_mode,
            category="pc",
            normalized_query=normalized,
            source="fallback",
            rationale="keyword_build_pc",
        )

    if any(phrase in normalized for phrase in ["ban chay", "best seller", "bestseller", "top selling", "mua nhieu", "ban nhieu"]):
        category = None
        if "dien thoai" in normalized or "iphone" in normalized:
            category = "phone"
        elif "laptop" in normalized:
            category = "laptop"
        elif "pc" in normalized or "may tinh" in normalized:
            category = "pc"
        return IntentRoute(
            intent="top_selling",
            confidence=0.76,
            category=category,
            normalized_query=normalized,
            source="fallback",
            rationale="keyword_top_selling",
        )

    explicit_category = _derive_explicit_category(normalized) if has_explicit_non_pc_category else None
    search_verbs = ("tim", "goi y", "tu van", "chon", "mua", "tham khao", "de xuat")
    if explicit_category and (
        budget_vnd
        or any(marker in normalized for marker in search_verbs)
    ):
        return IntentRoute(
            intent="product_search",
            confidence=0.77 if budget_vnd else 0.72,
            budget_vnd=budget_vnd,
            budget_mode=budget_mode,
            category=explicit_category,
            normalized_query=normalized,
            source="fallback",
            rationale="keyword_product_search",
        )

    if budget_vnd and history and not has_explicit_non_pc_category:
        for item in reversed(history[-4:]):
            if item.get("role") != "user":
                continue
            prior = normalize_text(str(item.get("content") or ""))
            if any(term in prior for term in build_terms):
                return IntentRoute(
                    intent="build_pc",
                    confidence=0.65,
                    budget_vnd=budget_vnd,
                    budget_mode=budget_mode,
                    category="pc",
                    normalized_query=normalized,
                    source="fallback",
                    rationale="follow_up_budget",
                )

    return IntentRoute(
        intent="general",
        confidence=0.55,
        budget_vnd=budget_vnd,
        budget_mode=budget_mode,
        category=_derive_explicit_category(normalized) if has_explicit_non_pc_category else None,
        normalized_query=normalized,
        source="fallback",
        rationale="default_general",
    )


def _build_intent_prompt(message: str, history: list[dict[str, Any]] | None) -> str:
    """Build strict JSON prompt for intent routing."""

    history_lines: list[str] = []
    for item in (history or [])[-6:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history_lines.append(f"{role}: {content}")

    return "\n".join(
        [
            "Classify the ecommerce assistant user query.",
            "Return JSON only.",
            "Allowed intent values: build_pc, top_selling, product_search, policy, general.",
            "build_pc is ONLY for custom desktop assembly (chọn CPU/GPU/mainboard/RAM…).",
            "If the user asks for laptop/notebook/MacBook/phone/tablet with a budget or asks to find/recommend/buy products, intent should be product_search with category set when obvious.",
            "If the user asks for laptop/notebook/MacBook/phone/tablet (including with a budget), intent must never be build_pc.",
            "Extract budget_vnd as integer VND if present, otherwise null.",
            "Allowed budget_mode values: target, upper, lower, null.",
            "Extract category if obvious. Allowed examples: pc, laptop, phone, accessory. Otherwise null.",
            "Confidence must be a number between 0 and 1.",
            "Do not answer the user question.",
            "",
            "JSON schema:",
            '{"intent":"general","confidence":0.0,"budget_vnd":null,"budget_mode":null,"category":null,"rationale":"short_reason"}',
            "",
            f"user_message: {message}",
            "recent_history:",
            "\n".join(history_lines) or "(empty)",
        ]
    )


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from model output."""

    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for index, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _normalize_model_route(payload: dict[str, Any], message: str) -> IntentRoute:
    """Normalize raw model JSON into a validated route object."""

    fallback = _fallback_route(message)
    intent = str(payload.get("intent") or fallback.intent).strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = fallback.intent

    try:
        confidence = float(payload.get("confidence", fallback.confidence))
    except (TypeError, ValueError):
        confidence = fallback.confidence
    confidence = max(0.0, min(1.0, confidence))

    budget_vnd = payload.get("budget_vnd")
    try:
        budget_vnd = int(float(budget_vnd)) if budget_vnd is not None else fallback.budget_vnd
    except (TypeError, ValueError):
        budget_vnd = fallback.budget_vnd
    if budget_vnd is not None and budget_vnd <= 0:
        budget_vnd = fallback.budget_vnd

    budget_mode = str(payload.get("budget_mode") or "").strip().lower() or fallback.budget_mode
    if budget_mode not in ALLOWED_BUDGET_MODES:
        budget_mode = fallback.budget_mode

    category = payload.get("category")
    category = str(category).strip().lower() if category is not None else fallback.category
    if category == "":
        category = fallback.category

    rationale = payload.get("rationale")
    rationale = str(rationale).strip() if rationale is not None else None

    route = IntentRoute(
        intent=intent,
        confidence=confidence,
        budget_vnd=budget_vnd,
        budget_mode=budget_mode,
        category=category,
        normalized_query=normalize_text(message),
        source="vertex_ai",
        rationale=rationale,
    )

    normalized = route.normalized_query
    if _looks_top_selling_query(normalized):
        route.intent = "top_selling"
        route.category = route.category or _derive_explicit_category(normalized)
        route.confidence = max(route.confidence, 0.8)
    if _has_explicit_non_pc_category(normalized) and route.intent == "build_pc":
        # Hard guard: if user explicitly asked for laptop/phone/tablet, do not route to build_pc.
        # This prevents "laptop 20 triệu" from being answered with PC parts after a prior PC chat.
        route.intent = "general"
        route.category = _derive_explicit_category(normalized) or route.category
        route.confidence = min(route.confidence, 0.6)
        route.source = "vertex_ai_guard"
        route.rationale = (route.rationale or "vertex_build_pc") + "|explicit_non_pc_override"

    return route


async def classify_intent(message: str, history: list[dict[str, Any]] | None = None) -> IntentRoute:
    """Classify one user message into build/top-selling/policy/general."""

    fallback = _fallback_route(message, history)
    if not settings.enable_intent_router:
        return fallback

    require_vertex_configuration()

    try:
        client = build_vertex_client()
        response = await client.aio.models.generate_content(
            model=settings.intent_model,
            contents=_build_intent_prompt(message, history),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=220,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (response.text or "").strip()
        await client.aio.aclose()
    except Exception:
        return fallback

    fragment = _extract_first_json_object(text)
    if not fragment:
        return fallback

    try:
        payload = json.loads(fragment)
    except Exception:
        return fallback
    if not isinstance(payload, dict):
        return fallback

    route = _normalize_model_route(payload, message)
    if route.confidence < settings.intent_min_confidence and fallback.confidence >= route.confidence:
        return fallback
    return route
