from __future__ import annotations

"""Gemini generation provider wrapper for grounded chatbot answers."""

import json
import re
from time import perf_counter

import httpx

from app.config import settings


def _format_contexts(contexts: list[dict]) -> str:
    """Render retrieved chunks into a deterministic context block for prompting."""

    lines: list[str] = []
    for index, item in enumerate(contexts, start=1):
        score = round(
            float(item.get("rerank_score", item.get("retrieval_score", 0.0))),
            3,
        )
        lines.extend(
            [
                f"[CTX_{index}]",
                f"title: {item.get('title', 'Untitled')}",
                f"uri: {item.get('uri', '')}",
                f"score: {score}",
                f"content: {item.get('content', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _build_prompt(query: str, locale: str, contexts: list[dict]) -> str:
    """Build final generation prompt with grounding and output format rules."""

    return "\n".join(
        [
            "You are a grounded ecommerce assistant.",
            "Rules:",
            "1. Answer using only the provided contexts.",
            "2. If context is insufficient, say so directly.",
            "3. Keep the answer practical and concise.",
            "4. Return plain answer text only (no markdown code fence, no JSON wrapper).",
            f"Locale: {locale}",
            f"User question: {query}",
            "",
            "Contexts:",
            _format_contexts(contexts),
        ]
    )


def _require_generation_configuration() -> None:
    """Fail fast when generation provider settings are incomplete."""

    if not settings.enable_gemini_chat:
        raise RuntimeError(
            "Gemini generation is disabled. Set ENABLE_GEMINI_CHAT=true in .env."
        )
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini generation.")


def _extract_first_candidate_text(data: dict) -> str:
    """Extract plain text from first Gemini candidate payload."""

    return "".join(
        part.get("text", "")
        for part in data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    ).strip()


def _extract_json_text(text: str) -> str:
    """Handle cases where model wraps JSON output in markdown fences."""

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        inner = cleaned[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner
    return cleaned


def _extract_first_json_object(text: str) -> str:
    """Extract first balanced JSON object from mixed model output."""

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


def _extract_answer_from_json_like(text: str) -> str:
    """Extract answer field from malformed JSON-like model output."""

    match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)', text, flags=re.DOTALL)
    if not match:
        return ""
    raw_value = match.group(1)
    try:
        decoded = json.loads(f'"{raw_value}"')
        return str(decoded).strip()
    except Exception:
        return raw_value.replace('\\"', '"').replace("\\n", "\n").strip()


def _normalize_confidence(value: object) -> float:
    """Clamp confidence output to [0, 1]."""

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def _normalize_citation_indexes(values: object, max_index: int) -> list[int]:
    """Validate citation indexes returned by the model."""

    if not isinstance(values, list):
        return []
    normalized: list[int] = []
    for item in values:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index == 0 and max_index >= 1:
            index = 1
        if 1 <= index <= max_index and index not in normalized:
            normalized.append(index)
    return normalized


async def generate_answer(query: str, locale: str, contexts: list[dict]) -> dict:
    """Generate final answer from contexts using Gemini chat model."""

    started = perf_counter()

    if not contexts:
        return {
            "answer": (
                "Hien tai kho tri thuc chua co du context phu hop de tra loi chinh xac "
                "cho cau hoi nay."
            ),
            "confidence": 0.2,
            "citation_indexes": [],
            "provider": "guardrail",
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    _require_generation_configuration()

    model = settings.gemini_chat_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={settings.gemini_api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _build_prompt(query, locale, contexts)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
            # Reduce hidden reasoning token usage so visible answer is not truncated.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini generation request failed: {exc}") from exc

    text = _extract_first_candidate_text(data)
    if not text:
        raise RuntimeError("Gemini generation response was empty.")

    cleaned_text = _extract_json_text(text)
    parsed: dict | None = None
    parse_candidates = [cleaned_text]
    fragment = _extract_first_json_object(cleaned_text)
    if fragment and fragment != cleaned_text:
        parse_candidates.append(fragment)

    for candidate in parse_candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            parsed = payload
            break

    if parsed is not None:
        answer = str(parsed.get("answer", "")).strip() or cleaned_text
        confidence = _normalize_confidence(parsed.get("confidence", 0.7))
        citation_indexes = _normalize_citation_indexes(
            parsed.get("citation_indexes", []),
            len(contexts),
        )
    else:
        extracted_answer = _extract_answer_from_json_like(cleaned_text)
        answer = extracted_answer or cleaned_text
        confidence = 0.7
        citation_indexes = []

    return {
        "answer": answer,
        "confidence": confidence,
        "citation_indexes": citation_indexes,
        "provider": "gemini",
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
