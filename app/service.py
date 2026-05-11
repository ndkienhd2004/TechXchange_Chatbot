from __future__ import annotations

"""Service layer orchestrating chat, persistence, and KB sync operations."""

import asyncio
from typing import Any
from typing import Optional
from uuid import uuid4

from app.config import settings
from app.orchestrator import run_rag_pipeline
from app.repository import app_message_repository
from app.repository import chatbot_repository
from app.source_sync import sync_from_source_database


class AssistantService:
    """Application service facade used by API endpoints."""

    def _build_conversation_title(self, text: str, max_length: int = 60) -> str:
        """Create a short conversation title from the first user message."""

        text = text.strip()
        return text if len(text) <= max_length else f"{text[:max_length]}..."

    async def chat(
        self,
        user_id: int,
        message: str,
        locale: str,
        conversation_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Process one user message and return assistant response payload."""

        content = str(message or "").strip()
        if not content:
            raise ValueError("message is required")
        if len(content) > 5000:
            raise ValueError("message too long (max 5000 chars)")

        if conversation_id:
            conversation = app_message_repository.get_conversation(user_id, conversation_id)
            conversation_id = conversation.id
        else:
            conversation = app_message_repository.create_conversation(
                user_id=user_id,
                title=self._build_conversation_title(content),
            )
            conversation_id = conversation.id

        request_id = uuid4().hex
        app_message_repository.append_message(conversation_id, "user", content)
        history = app_message_repository.get_history(conversation_id, limit=10)

        rag_result = await asyncio.wait_for(
            run_rag_pipeline(content, locale, history),
            timeout=settings.assistant_timeout_ms,
        )
        rag_result.setdefault("usage", {})
        rag_result.setdefault("debug", {})
        rag_result["usage"]["request_id"] = request_id
        rag_result["usage"]["conversation_id"] = conversation_id
        rag_result["usage"]["locale"] = locale
        debug = dict(rag_result.get("debug") or {})
        rag_result["usage"]["route"] = debug.get("intent_route")
        rag_result["usage"]["retrieval_backend"] = debug.get("retrieval_backend")
        rag_result["debug"]["request_id"] = request_id

        assistant_message = app_message_repository.append_message(
            conversation_id,
            "assistant",
            rag_result["answer"],
            citations=rag_result["citations"],
            confidence=rag_result["confidence"],
            usage=rag_result["usage"],
        )

        return {
            "conversation_id": conversation_id,
            "answer": rag_result["answer"],
            "confidence": rag_result["confidence"],
            "citations": rag_result["citations"],
            "usage": rag_result["usage"],
            "debug": rag_result["debug"],
            "message_id": assistant_message.id,
        }

    def list_conversations(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        """Return conversation list for current user."""

        return app_message_repository.list_conversations(user_id, limit)

    def get_messages(
        self,
        user_id: int,
        conversation_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return full message history for one conversation."""

        return app_message_repository.get_messages(user_id, conversation_id, limit)

    def health(self) -> dict[str, Any]:
        """Expose service readiness, provider flags, and storage capabilities."""

        app_database_backend = settings.app_database_url.split(":", 1)[0]
        chatbot_database_backend = settings.chatbot_database_url.split(":", 1)[0]
        vertex_ready = bool(
            settings.google_genai_use_vertexai
            and settings.google_cloud_project
            and settings.google_cloud_location
        )
        return {
            "status": "ok",
            "app_database_backend": app_database_backend,
            "chatbot_database_backend": chatbot_database_backend,
            "providers": {
                "vertex_ai_enabled": settings.google_genai_use_vertexai,
                "google_cloud_project": settings.google_cloud_project,
                "google_cloud_location": settings.google_cloud_location,
                "gemini_chat_enabled": settings.enable_gemini_chat,
                "gemini_embed_enabled": settings.enable_gemini_embed,
                "generation_ready": vertex_ready and settings.enable_gemini_chat,
                "embedding_ready": vertex_ready and settings.enable_gemini_embed,
            },
            "knowledge": chatbot_repository.get_kb_stats(),
            "storage": chatbot_repository.get_storage_capabilities(),
            "sync": chatbot_repository.get_sync_state(),
        }

    async def ingest_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalize and ingest manually provided knowledge documents."""

        prepared = []
        for index, document in enumerate(documents, start=1):
            title = str(document.get("title", f"Doc {index}"))
            prepared.append(
                {
                    "source_key": str(document.get("source_key") or f"manual:{index}:{title}"),
                    "title": title,
                    "uri": str(document.get("uri", "")),
                    "content": str(document.get("content", "")),
                    "metadata": {
                        **dict(document.get("metadata", {})),
                        "source": "manual_ingest",
                    },
                }
            )
        return await chatbot_repository.upsert_documents(
            "manual_ingest",
            prepared,
            purge_missing=False,
        )

    async def reindex(self) -> dict[str, Any]:
        """Sync current source product data into chatbot knowledge storage."""

        result = await sync_from_source_database()
        result["stats"] = chatbot_repository.get_kb_stats()
        return result


assistant_service = AssistantService()
