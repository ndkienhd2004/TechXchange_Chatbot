from __future__ import annotations

"""Centralized environment configuration for chatbot service."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    """Read env value as non-empty string, otherwise use default."""

    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned if cleaned else default


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse optional env string to bool using strict `true` check."""

    if value is None or value == "":
        return default
    return value.strip().lower() == "true"


def _to_int(value: Optional[str], default: int) -> int:
    """Parse optional env string to int with fallback on invalid input."""

    try:
        return int(value) if value is not None and str(value).strip() != "" else default
    except (TypeError, ValueError):
        return default


def _to_float(value: Optional[str], default: float) -> float:
    """Parse optional env string to float with fallback on invalid input."""

    try:
        return float(value) if value is not None and str(value).strip() != "" else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from `.env` once at startup."""

    # API runtime.
    host: str = _env_str("HOST", "0.0.0.0")
    port: int = _to_int(os.getenv("PORT"), 8000)
    node_env: str = _env_str("NODE_ENV", "development")

    # Database URLs.
    app_database_url: str = _env_str("APP_DATABASE_URL", "sqlite:///./app_main.db")
    chatbot_database_url: str = _env_str(
        "CHATBOT_DATABASE_URL",
        "sqlite:///./chatbot_app.db",
    )
    source_database_url: str = _env_str("SOURCE_DATABASE_URL", "")

    # Chatbot DB bootstrap/vector options.
    auto_bootstrap_chatbot_database: bool = _to_bool(
        os.getenv("AUTO_BOOTSTRAP_CHATBOT_DATABASE"),
        True,
    )
    auto_enable_pgvector: bool = _to_bool(
        os.getenv("AUTO_ENABLE_PGVECTOR"),
        True,
    )
    embedding_dimensions: int = _to_int(os.getenv("EMBEDDING_DIMENSIONS"), 24)

    # Auth and request timeout.
    assistant_require_auth: bool = _to_bool(
        os.getenv("ASSISTANT_REQUIRE_AUTH"),
        False,
    )
    assistant_default_user_id: int = _to_int(
        os.getenv("ASSISTANT_DEFAULT_USER_ID"),
        1,
    )
    assistant_timeout_ms: float = _to_float(
        os.getenv("ASSISTANT_TIMEOUT_MS"),
        12.0,
    )

    # RAG pipeline limits.
    rag_topk_vector: int = _to_int(os.getenv("RAG_TOPK_VECTOR"), 30)
    rag_topk_keyword: int = _to_int(os.getenv("RAG_TOPK_KEYWORD"), 20)
    rag_topk_rerank: int = _to_int(os.getenv("RAG_TOPK_RERANK"), 8)
    rag_max_context_tokens: int = _to_int(
        os.getenv("RAG_MAX_CONTEXT_TOKENS"),
        3500,
    )

    # Gemini providers.
    gemini_api_key: str = _env_str("GEMINI_API_KEY", "")
    gemini_chat_model: str = _env_str("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    gemini_embed_model: str = _env_str(
        "GEMINI_EMBED_MODEL",
        "gemini-embedding-001",
    )
    enable_gemini_chat: bool = _to_bool(os.getenv("ENABLE_GEMINI_CHAT"), False)
    enable_gemini_embed: bool = _to_bool(os.getenv("ENABLE_GEMINI_EMBED"), False)
    google_genai_use_vertexai: bool = _to_bool(
        os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
        False,
    )
    google_cloud_project: str = _env_str("GOOGLE_CLOUD_PROJECT", "")
    google_cloud_location: str = _env_str("GOOGLE_CLOUD_LOCATION", "us-central1")
    enable_intent_router: bool = _to_bool(os.getenv("ENABLE_INTENT_ROUTER"), True)
    intent_model: str = _env_str("INTENT_MODEL", "gemini-2.5-flash")
    intent_min_confidence: float = _to_float(os.getenv("INTENT_MIN_CONFIDENCE"), 0.7)

    # Background sync scheduling.
    sync_enabled: bool = _to_bool(os.getenv("SYNC_ENABLED"), True)
    sync_on_startup: bool = _to_bool(os.getenv("SYNC_ON_STARTUP"), True)
    sync_interval_seconds: int = _to_int(os.getenv("SYNC_INTERVAL_SECONDS"), 900)


settings = Settings()
