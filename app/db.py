from __future__ import annotations

"""SQLAlchemy models and engine/session factories for app + chatbot databases."""

from datetime import datetime, timezone
from typing import Optional

import psycopg
from sqlalchemy import JSON
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

from app.config import settings


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp for model defaults."""

    return datetime.now(timezone.utc)


class AppBase(DeclarativeBase):
    """Declarative base for conversation/message tables in app database."""

    pass


class ChatbotBase(DeclarativeBase):
    """Declarative base for knowledge/sync tables in chatbot database."""

    pass


class AssistantConversation(AppBase):
    """Conversation header for one user chat thread."""

    __tablename__ = "assistant_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list["AssistantMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.id",
    )


class AssistantMessage(AppBase):
    """Persisted message turn (user or assistant) inside a conversation."""

    __tablename__ = "assistant_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("assistant_conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usage_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[AssistantConversation] = relationship(back_populates="messages")


class KBDocument(ChatbotBase):
    """Source document record before being split into retrieval chunks."""

    __tablename__ = "kb_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    source_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    uri: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    source_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chunks: Mapped[list["KBChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="KBChunk.chunk_index",
    )


class KBChunk(ChatbotBase):
    """Retrieval unit containing text chunk + embedding payload."""

    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("kb_documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(255))
    uri: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[KBDocument] = relationship(back_populates="chunks")


class SyncState(ChatbotBase):
    """Track latest sync execution status for observability/health endpoints."""

    __tablename__ = "sync_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    synced_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def _normalize_url(url: str) -> str:
    """Normalize postgres URL variants to SQLAlchemy psycopg driver format."""

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _is_postgres_url(url: str) -> bool:
    """Return whether the provided DB URL targets PostgreSQL."""

    normalized = _normalize_url(url)
    return normalized.startswith("postgresql+psycopg://")


def _ensure_postgres_database(url: str, auto_create: bool = True) -> None:
    """Create target PostgreSQL database when enabled and it does not exist."""

    if not auto_create or not _is_postgres_url(url):
        return

    target_url = make_url(_normalize_url(url))
    database_name = str(target_url.database or "").strip()
    if not database_name:
        return

    raw_target_url = target_url.set(drivername="postgresql")
    maintenance_url = raw_target_url.set(database="postgres")
    with psycopg.connect(str(maintenance_url), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,),
            )
            exists = cursor.fetchone() is not None
            if exists:
                return
            try:
                cursor.execute(
                    f'CREATE DATABASE "{database_name}"'
                )
            except psycopg.Error:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (database_name,),
                )
                exists = cursor.fetchone() is not None
                if not exists:
                    raise


def _build_engine(url: str):
    """Create SQLAlchemy engine with backend-appropriate defaults."""

    normalized = _normalize_url(url)
    kwargs = {"future": True, "pool_pre_ping": True}
    if normalized.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(normalized, **kwargs)


_ensure_postgres_database(
    settings.chatbot_database_url,
    auto_create=settings.auto_bootstrap_chatbot_database,
)

# Engine for existing app tables (messages/conversations).
app_engine = _build_engine(settings.app_database_url)
# Engine for dedicated chatbot KB/vector storage.
chatbot_engine = _build_engine(settings.chatbot_database_url)

# Session factory for app database operations.
AppSessionLocal = sessionmaker(
    bind=app_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# Session factory for chatbot database operations.
ChatbotSessionLocal = sessionmaker(
    bind=chatbot_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# Optional source engine used for periodic sync from backend product tables.
resolved_source_url = settings.source_database_url or settings.app_database_url
source_engine = _build_engine(resolved_source_url) if resolved_source_url else None
