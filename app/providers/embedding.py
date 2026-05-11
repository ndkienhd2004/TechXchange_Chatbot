from __future__ import annotations

"""Vertex AI embedding provider wrapper."""

from time import perf_counter

from google.genai import types

from app.config import settings
from app.providers.vertex_client import build_vertex_client
from app.providers.vertex_client import require_vertex_configuration


def _require_embedding_configuration() -> None:
    """Fail fast when embedding provider settings are incomplete."""

    if not settings.enable_gemini_embed:
        raise RuntimeError(
            "Gemini embedding is disabled. Set ENABLE_GEMINI_EMBED=true in .env."
        )
    require_vertex_configuration()


async def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> dict:
    """Create one Vertex AI embedding vector for the given text payload."""

    started = perf_counter()
    _require_embedding_configuration()

    try:
        client = build_vertex_client()
        response = await client.aio.models.embed_content(
            model=settings.gemini_embed_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dimensions,
            ),
        )
        await client.aio.aclose()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Vertex AI embedding request failed: {exc}") from exc

    embeddings = list(response.embeddings or [])
    values = list(embeddings[0].values or []) if embeddings else []
    if not values:
        raise RuntimeError("Vertex AI embedding response was empty.")

    return {
        "vector": [float(item) for item in values],
        "provider": "vertex_ai",
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
