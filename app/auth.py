from __future__ import annotations

"""Authentication helpers for assistant API endpoints."""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException

from app.config import settings


@dataclass
class CurrentUser:
    """Resolved user identity used by route dependencies."""

    user_id: int
    role: str


def _parse_bearer_user_id(authorization: Optional[str]) -> Optional[int]:
    """Extract numeric user id from Bearer token for local/simple auth."""

    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        return None
    token = parts[1].strip()
    if token.startswith("demo-user-"):
        try:
            value = int(token.replace("demo-user-", ""))
        except ValueError:
            return None
        return value if value > 0 else None
    try:
        value = int(token)
    except ValueError:
        return None
    return value if value > 0 else None


def require_user(
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[int] = Header(default=None),
    x_user_role: Optional[str] = Header(default=None),
) -> CurrentUser:
    """Resolve current user from headers, with optional local auth bypass."""

    role = (x_user_role or "user").strip() or "user"
    user_id = x_user_id or _parse_bearer_user_id(authorization)

    if not settings.assistant_require_auth:
        return CurrentUser(user_id or settings.assistant_default_user_id, role)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid auth. Use Bearer demo-user-<id> for local testing.",
        )

    return CurrentUser(user_id, role)


def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    """Require admin role for protected management endpoints."""

    if user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin permission required")
    return user
