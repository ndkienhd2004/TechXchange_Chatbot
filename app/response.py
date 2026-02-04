from __future__ import annotations

"""Small helpers to keep API response envelopes consistent."""

from typing import Any
from typing import Optional


def success(message: str, data: Optional[Any] = None) -> dict[str, Any]:
    """Build standard success response with optional payload."""

    body: dict[str, Any] = {"code": "200", "success": True, "message": message}
    if data is not None:
        body["data"] = data
    return body


def created(message: str, data: Optional[Any] = None) -> dict[str, Any]:
    """Build standard created response with optional payload."""

    body: dict[str, Any] = {"code": "201", "success": True, "message": message}
    if data is not None:
        body["data"] = data
    return body


def failure(code: int, message: str) -> dict[str, Any]:
    """Build standard failure response body."""

    return {"code": str(code), "success": False, "message": message}
