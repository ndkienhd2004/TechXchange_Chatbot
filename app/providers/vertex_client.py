from __future__ import annotations

"""Shared Vertex AI client factory using ADC authentication."""

from google import genai
from google.genai import types

from app.config import settings


def require_vertex_configuration() -> None:
    """Fail fast when Vertex AI runtime settings are incomplete."""

    if not settings.google_genai_use_vertexai:
        raise RuntimeError(
            "Vertex AI is disabled. Set GOOGLE_GENAI_USE_VERTEXAI=true in .env."
        )
    if not settings.google_cloud_project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex AI.")
    if not settings.google_cloud_location:
        raise RuntimeError("GOOGLE_CLOUD_LOCATION is required for Vertex AI.")


def build_vertex_client() -> genai.Client:
    """Create a Gen AI client bound to Vertex AI using ADC."""

    require_vertex_configuration()
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        http_options=types.HttpOptions(api_version="v1"),
    )
