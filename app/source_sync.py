from __future__ import annotations

"""Source database sync entrypoint plus backward-compatible candidate exports."""

from app.build_pc_candidates import get_build_pc_candidates
from app.db import source_engine
from app.product_search_candidates import get_product_search_candidates
from app.product_source_utils import PRODUCT_SYNC_QUERY, build_product_document
from app.repository import chatbot_repository
from app.top_selling_candidates import get_top_selling_candidates
from sqlalchemy import text


async def sync_from_source_database() -> dict[str, object]:
    """Sync active products from source DB into chatbot knowledge base."""

    if source_engine is None:
        return {"status": "skipped", "reason": "SOURCE_DATABASE_URL is not configured"}

    try:
        with source_engine.connect() as connection:
            rows = connection.execute(text(PRODUCT_SYNC_QUERY)).mappings().all()
        documents = [build_product_document(dict(row)) for row in rows]
        result = await chatbot_repository.upsert_documents(
            "source_product",
            documents,
            purge_missing=True,
        )
        chatbot_repository.update_sync_state(
            "source_sync",
            synced_count=result["synced_documents"],
            error=None,
            success=True,
        )
        return {"status": "ok", **result}
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc).lower()
        if "no such table" in error_text or "undefined table" in error_text:
            return {
                "status": "skipped",
                "reason": "Source database does not expose the expected product tables yet",
            }
        chatbot_repository.update_sync_state(
            "source_sync",
            synced_count=0,
            error=str(exc),
            success=False,
        )
        return {"status": "error", "reason": str(exc)}
