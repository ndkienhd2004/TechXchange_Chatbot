from __future__ import annotations

"""Background scheduler for periodic source-to-chatbot synchronization."""

import asyncio
import contextlib
from typing import Any
from typing import Optional

from app.config import settings
from app.repository import app_message_repository
from app.repository import chatbot_repository
from app.source_sync import sync_from_source_database


class SyncScheduler:
    """Manage periodic source sync lifecycle tied to FastAPI app lifespan."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Initialize databases and start periodic sync loop."""

        app_message_repository.init_database()
        chatbot_repository.init_database()

        if not settings.sync_enabled:
            return

        if settings.sync_on_startup:
            await sync_from_source_database()

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop background sync task gracefully."""

        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run_loop(self) -> None:
        """Run sync forever with configured interval."""

        interval = max(30, settings.sync_interval_seconds)
        while True:
            await asyncio.sleep(interval)
            await sync_from_source_database()

    async def run_once(self) -> dict[str, Any]:
        """Manual single-run helper for diagnostics."""

        return await sync_from_source_database()


sync_scheduler = SyncScheduler()
