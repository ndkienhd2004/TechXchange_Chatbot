from __future__ import annotations

"""Pydantic schemas used by the chatbot API request layer."""

from typing import Any
from typing import Optional

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """Single knowledge document payload for manual ingestion."""

    title: str
    uri: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """User chat request payload."""

    message: str
    conversation_id: Optional[int] = None
    locale: str = "vi-VN"


class IngestRequest(BaseModel):
    """Batch ingestion request payload."""

    documents: list[DocumentIn]
