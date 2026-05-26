from __future__ import annotations

"""Database repository layer for chatbot KB and assistant conversations."""

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from typing import Optional

from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import AppBase
from app.db import AppSessionLocal
from app.db import AssistantConversation
from app.db import AssistantMessage
from app.db import ChatbotBase
from app.db import ChatbotSessionLocal
from app.db import KBChunk
from app.db import KBDocument
from app.db import SyncState
from app.db import app_engine
from app.db import chatbot_engine
from app.db import utcnow
from app.knowledge_base import chunk_text
from app.providers.embedding import embed_text
from app.text_normalization import tokenize_text_preserve_accents


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse flexible datetime input into `datetime` object when possible."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _checksum_for(document: dict[str, Any]) -> str:
    """Build stable checksum for change detection on document payload."""

    payload = json.dumps(
        {
            "title": document.get("title"),
            "uri": document.get("uri"),
            "content": document.get("content"),
            "metadata": document.get("metadata", {}),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_literal(values: list[float]) -> str:
    """Convert list of floats to pgvector literal format (`[x,y,z]`)."""

    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for fallback Python vector search."""

    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _terms(text: str) -> list[str]:
    """Tokenize string into lowercase alphanumeric terms."""

    return tokenize_text_preserve_accents(text)


def _query_terms(queries: list[str], limit: int = 12) -> list[str]:
    """Collect unique query terms from multiple queries with a hard cap."""

    ordered: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for token in _terms(query):
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
            if len(ordered) >= limit:
                return ordered
    return ordered


@contextmanager
def app_session_scope() -> Session:
    """Transactional scope for the app database (messages/conversations)."""

    session = AppSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def chatbot_session_scope() -> Session:
    """Transactional scope for chatbot knowledge database."""

    session = ChatbotSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class ChatbotKnowledgeRepository:
    """Access layer for knowledge documents/chunks and sync metadata."""

    def __init__(self) -> None:
        self._capabilities_cache: Optional[dict[str, Any]] = None

    def init_database(self) -> None:
        """Create KB tables and vector indexes if available."""

        ChatbotBase.metadata.create_all(chatbot_engine)
        self._ensure_vector_support()
        self._capabilities_cache = None

    def _detect_capabilities(self) -> dict[str, Any]:
        with chatbot_session_scope() as session:
            dialect = session.bind.dialect.name
            is_postgres = dialect == "postgresql"
            vector_extension = False
            vector_column = False

            if is_postgres:
                vector_extension = bool(
                    session.execute(
                        text(
                            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                        )
                    ).scalar()
                )
                vector_column = bool(
                    session.execute(
                        text(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM information_schema.columns
                                WHERE table_name = 'kb_chunks'
                                  AND column_name = 'embedding_vector'
                            )
                            """
                        )
                    ).scalar()
                )

            return {
                "backend": dialect,
                "vector_extension": vector_extension,
                "vector_column": vector_column,
                "vector_search_enabled": is_postgres and vector_extension and vector_column,
                "embedding_dimensions": settings.embedding_dimensions,
            }

    def _ensure_vector_support(self) -> None:
        if chatbot_engine.dialect.name != "postgresql":
            return

        with chatbot_engine.begin() as connection:
            if settings.auto_enable_pgvector:
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception:
                    return

            extension_exists = bool(
                connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                ).scalar()
            )
            if not extension_exists:
                return

            connection.execute(
                text(
                    f"""
                    ALTER TABLE kb_chunks
                    ADD COLUMN IF NOT EXISTS embedding_vector vector({settings.embedding_dimensions})
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id
                    ON kb_chunks (document_id)
                    """
                )
            )
            try:
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_vector
                        ON kb_chunks
                        USING hnsw (embedding_vector vector_cosine_ops)
                        """
                    )
                )
            except Exception:
                try:
                    connection.execute(
                        text(
                            """
                            CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_vector
                            ON kb_chunks
                            USING ivfflat (embedding_vector vector_cosine_ops)
                            """
                        )
                    )
                except Exception:
                    return

    def get_storage_capabilities(self, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh or self._capabilities_cache is None:
            self._capabilities_cache = self._detect_capabilities()
        return dict(self._capabilities_cache)

    async def upsert_documents(
        self,
        source_type: str,
        documents: list[dict[str, Any]],
        purge_missing: bool = False,
    ) -> dict[str, Any]:
        """Insert/update documents, re-chunk content, and refresh embeddings."""

        prepared: list[dict[str, Any]] = []
        for document in documents:
            source_key = str(document.get("source_key") or "").strip()
            if not source_key:
                raise ValueError("source_key is required for upsert")
            metadata = dict(document.get("metadata", {}))
            prepared.append(
                {
                    "source_key": source_key,
                    "title": str(document.get("title", "Untitled")),
                    "uri": str(document.get("uri", "")),
                    "content": str(document.get("content", "")),
                    "metadata": metadata,
                    "checksum": _checksum_for(document),
                    "source_updated_at": _parse_datetime(
                        metadata.get("updated_at") or document.get("updated_at")
                    ),
                }
            )

        synced = 0
        changed = 0
        active_keys = {item["source_key"] for item in prepared}
        capabilities = self.get_storage_capabilities(force_refresh=True)
        vector_write_enabled = capabilities["vector_search_enabled"]

        with chatbot_session_scope() as session:
            existing_rows = session.execute(
                select(KBDocument).where(KBDocument.source_type == source_type)
            ).scalars().all()
            existing_by_key = {row.source_key: row for row in existing_rows}

            for item in prepared:
                synced += 1
                row = existing_by_key.get(item["source_key"])
                if row and row.checksum == item["checksum"]:
                    row.updated_at = utcnow()
                    continue

                if row is None:
                    row = KBDocument(
                        source_type=source_type,
                        source_key=item["source_key"],
                        title=item["title"],
                        uri=item["uri"],
                        content=item["content"],
                        metadata_json=item["metadata"],
                        checksum=item["checksum"],
                        source_updated_at=item["source_updated_at"],
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(row)
                    session.flush()
                else:
                    row.title = item["title"]
                    row.uri = item["uri"]
                    row.content = item["content"]
                    row.metadata_json = item["metadata"]
                    row.checksum = item["checksum"]
                    row.source_updated_at = item["source_updated_at"]
                    row.updated_at = utcnow()
                    session.execute(delete(KBChunk).where(KBChunk.document_id == row.id))
                    session.flush()

                parts = chunk_text(item["content"])
                for index, part in enumerate(parts, start=1):
                    embedding = await embed_text(part, task_type="RETRIEVAL_DOCUMENT")
                    chunk_row = KBChunk(
                        document_id=row.id,
                        chunk_id=f"{item['source_key']}:{index}",
                        chunk_index=index,
                        title=item["title"],
                        uri=item["uri"],
                        content=part,
                        metadata_json=item["metadata"],
                        embedding_json=embedding.get("vector"),
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(chunk_row)
                    session.flush()

                    vector = embedding.get("vector") or []
                    if vector_write_enabled and len(vector) == settings.embedding_dimensions:
                        session.execute(
                            text(
                                """
                                UPDATE kb_chunks
                                SET embedding_vector = CAST(:embedding AS vector)
                                WHERE id = :chunk_id
                                """
                            ),
                            {
                                "embedding": _vector_literal(vector),
                                "chunk_id": chunk_row.id,
                            },
                        )
                changed += 1

            deleted = 0
            if purge_missing:
                stale_rows = [
                    row for row in existing_rows if row.source_key not in active_keys
                ]
                for row in stale_rows:
                    session.delete(row)
                deleted = len(stale_rows)

        return {
            "synced_documents": synced,
            "changed_documents": changed,
            "deleted_documents": deleted if purge_missing else 0,
            "stats": self.get_kb_stats(),
        }

    def get_chunks(self, include_embeddings: bool = False) -> list[dict[str, Any]]:
        """Return all KB chunks; embeddings are optional for heavier workflows."""

        with chatbot_session_scope() as session:
            rows = session.execute(select(KBChunk).order_by(KBChunk.id.asc())).scalars().all()
            return [
                {
                    "chunk_id": row.chunk_id,
                    "title": row.title,
                    "uri": row.uri or "",
                    "content": row.content,
                    "metadata": row.metadata_json or {},
                    "chunk_index": row.chunk_index,
                    **(
                        {"embedding": row.embedding_json or []}
                        if include_embeddings
                        else {}
                    ),
                }
                for row in rows
            ]

    def keyword_search(self, queries: list[str], limit: int) -> list[dict[str, Any]]:
        """Run simple SQL LIKE retrieval against chunk text."""

        search_terms = _query_terms(queries)
        if not search_terms:
            return []

        score_parts = []
        filters = []
        params: dict[str, Any] = {
            "term_count": max(1, len(search_terms)),
            "limit": max(1, limit),
        }

        for index, term in enumerate(search_terms):
            param_name = f"term_{index}"
            score_parts.append(
                f"CASE WHEN LOWER(COALESCE(title, '') || ' ' || COALESCE(content, '')) LIKE :{param_name} THEN 1 ELSE 0 END"
            )
            filters.append(
                f"LOWER(COALESCE(title, '') || ' ' || COALESCE(content, '')) LIKE :{param_name}"
            )
            params[param_name] = f"%{term}%"

        query = text(
            f"""
            SELECT
                chunk_id,
                title,
                uri,
                content,
                metadata_json,
                chunk_index,
                ((1.0 * ({" + ".join(score_parts)})) / :term_count) AS keyword_score
            FROM kb_chunks
            WHERE {" OR ".join(filters)}
            ORDER BY keyword_score DESC, id ASC
            LIMIT :limit
            """
        )

        with chatbot_session_scope() as session:
            rows = session.execute(query, params).mappings().all()
            return [
                {
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "uri": row["uri"] or "",
                    "content": row["content"],
                    "metadata": row["metadata_json"] or {},
                    "chunk_index": row["chunk_index"],
                    "keyword_score": float(row["keyword_score"] or 0.0),
                }
                for row in rows
            ]

    def vector_search(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        """Run pgvector nearest-neighbor retrieval when extension is available."""

        capabilities = self.get_storage_capabilities()
        if not capabilities["vector_search_enabled"]:
            return []
        if len(query_vector) != settings.embedding_dimensions:
            return []

        query = text(
            """
            SELECT
                chunk_id,
                title,
                uri,
                content,
                metadata_json,
                chunk_index,
                1 - (embedding_vector <=> CAST(:embedding AS vector)) AS semantic_score
            FROM kb_chunks
            WHERE embedding_vector IS NOT NULL
            ORDER BY embedding_vector <=> CAST(:embedding AS vector), id ASC
            LIMIT :limit
            """
        )

        with chatbot_session_scope() as session:
            rows = session.execute(
                query,
                {
                    "embedding": _vector_literal(query_vector),
                    "limit": max(1, limit),
                },
            ).mappings().all()
            return [
                {
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "uri": row["uri"] or "",
                    "content": row["content"],
                    "metadata": row["metadata_json"] or {},
                    "chunk_index": row["chunk_index"],
                    "semantic_score": float(row["semantic_score"] or 0.0),
                }
                for row in rows
            ]

    def get_kb_stats(self) -> dict[str, Any]:
        """Compute KB totals grouped by document type."""

        with chatbot_session_scope() as session:
            rows = session.execute(select(KBChunk)).scalars().all()
            grouped: dict[str, int] = {}
            for row in rows:
                doc_type = str((row.metadata_json or {}).get("doc_type", "unknown"))
                grouped[doc_type] = grouped.get(doc_type, 0) + 1
            return {
                "total_chunks": len(rows),
                "total_documents": session.scalar(select(func.count()).select_from(KBDocument)) or 0,
                "by_type": grouped,
            }

    def get_sync_state(self, name: str = "source_sync") -> dict[str, Any]:
        """Read last sync state for operational visibility."""

        with chatbot_session_scope() as session:
            row = session.execute(
                select(SyncState).where(SyncState.name == name)
            ).scalar_one_or_none()
            if row is None:
                return {
                    "name": name,
                    "last_success_at": None,
                    "last_run_at": None,
                    "last_error": None,
                    "synced_count": 0,
                }
            return {
                "name": row.name,
                "last_success_at": row.last_success_at.isoformat()
                if row.last_success_at
                else None,
                "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
                "last_error": row.last_error,
                "synced_count": row.synced_count,
            }

    def update_sync_state(
        self,
        name: str,
        synced_count: int,
        error: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Persist sync status after each source ingestion run."""

        with chatbot_session_scope() as session:
            row = session.execute(
                select(SyncState).where(SyncState.name == name)
            ).scalar_one_or_none()
            if row is None:
                row = SyncState(
                    name=name,
                    synced_count=synced_count,
                    last_error=error,
                    last_run_at=utcnow(),
                    last_success_at=utcnow() if success else None,
                    updated_at=utcnow(),
                )
                session.add(row)
                return

            row.synced_count = synced_count
            row.last_error = error
            row.last_run_at = utcnow()
            if success:
                row.last_success_at = utcnow()
            row.updated_at = utcnow()


class AppMessageRepository:
    """Access layer for user conversations and assistant messages."""

    def init_database(self) -> None:
        AppBase.metadata.create_all(app_engine)

    def create_conversation(self, user_id: int, title: str) -> AssistantConversation:
        with app_session_scope() as session:
            row = AssistantConversation(
                user_id=user_id,
                title=title,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return row

    def get_conversation(self, user_id: int, conversation_id: int) -> AssistantConversation:
        with app_session_scope() as session:
            row = session.execute(
                select(AssistantConversation).where(
                    AssistantConversation.id == int(conversation_id),
                    AssistantConversation.user_id == int(user_id),
                )
            ).scalar_one_or_none()
            if row is None:
                raise ValueError("Conversation not found")
            return row

    def append_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        citations: Optional[list[dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> AssistantMessage:
        with app_session_scope() as session:
            conversation = session.get(AssistantConversation, int(conversation_id))
            if conversation is None:
                raise ValueError("Conversation not found")
            conversation.updated_at = utcnow()
            row = AssistantMessage(
                conversation_id=conversation.id,
                role=role,
                content=content,
                citations_json=citations or [],
                confidence=confidence,
                usage_json=usage,
                created_at=utcnow(),
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return row

    def list_conversations(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with app_session_scope() as session:
            rows = session.execute(
                select(AssistantConversation)
                .where(AssistantConversation.user_id == int(user_id))
                .order_by(AssistantConversation.updated_at.desc())
                .limit(max(1, min(int(limit), 100)))
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "updated_at": row.updated_at.isoformat(),
                    "message_count": len(row.messages),
                }
                for row in rows
            ]

    def get_messages(
        self,
        user_id: int,
        conversation_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with app_session_scope() as session:
            conversation = session.execute(
                select(AssistantConversation).where(
                    AssistantConversation.id == int(conversation_id),
                    AssistantConversation.user_id == int(user_id),
                )
            ).scalar_one_or_none()
            if conversation is None:
                raise ValueError("Conversation not found")

            rows = session.execute(
                select(AssistantMessage)
                .where(AssistantMessage.conversation_id == conversation.id)
                .order_by(AssistantMessage.id.desc())
                .limit(max(1, min(int(limit), 200)))
            ).scalars().all()
            rows.reverse()
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "citations": row.citations_json or [],
                    "confidence": row.confidence,
                    "usage": row.usage_json,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def get_history(self, conversation_id: int, limit: int = 10) -> list[dict[str, str]]:
        with app_session_scope() as session:
            rows = session.execute(
                select(AssistantMessage)
                .where(AssistantMessage.conversation_id == int(conversation_id))
                .order_by(AssistantMessage.id.desc())
                .limit(max(1, min(int(limit), 100)))
            ).scalars().all()
            rows.reverse()
            return [{"role": row.role, "content": row.content} for row in rows]


chatbot_repository = ChatbotKnowledgeRepository()
app_message_repository = AppMessageRepository()
