from __future__ import annotations

"""Gemini embedding provider wrapper."""

from time import perf_counter

import httpx

from app.config import settings


def _require_embedding_configuration() -> None:
    """Fail fast when embedding provider settings are incomplete."""

    if not settings.enable_gemini_embed:
        raise RuntimeError(
            "Gemini embedding is disabled. Set ENABLE_GEMINI_EMBED=true in .env."
        )
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")


async def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> dict:
    """Create one Gemini embedding vector for the given text payload."""

    started = perf_counter()
    _require_embedding_configuration()

    model = settings.gemini_embed_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:embedContent?key={settings.gemini_api_key}"
    )
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

    values = data.get("embedding", {}).get("values", [])
    if not values:
        raise RuntimeError("Gemini embedding response was empty.")

    return {
        "vector": [float(item) for item in values],
        "provider": "gemini",
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
