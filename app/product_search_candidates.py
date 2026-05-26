from __future__ import annotations

from sqlalchemy import text

from app.db import source_engine
from app.product_source_utils import (
    PRODUCT_SYNC_QUERY,
    collapse_kb_product_chunks,
    derive_device_category,
    extract_product_terms,
    extract_top_selling_category_hints,
    matches_any_hint,
    matches_explicit_device_category,
    matches_terms,
)
from app.repository import chatbot_repository
from app.top_selling_candidates import to_top_selling_candidate
from app.product_source_utils import build_product_document


def get_product_search_candidates(
    query: str,
    category: str,
    budget_vnd: int | None = None,
    budget_mode: str | None = None,
    limit: int = 12,
) -> dict[str, object]:
    capped_limit = max(1, min(int(limit), 50))
    terms = extract_product_terms(query)
    category_hints = extract_top_selling_category_hints(query)
    explicit_category = category or derive_device_category(query)
    if not category_hints and explicit_category:
        category_hints = [explicit_category]

    def _price_matches(row: dict) -> bool:
        if not budget_vnd or budget_vnd <= 0:
            return True
        try:
            price = int(float(row.get("product_price") or row.get("price_vnd") or 0))
        except (TypeError, ValueError):
            price = 0
        if price <= 0:
            return True
        mode = str(budget_mode or "").strip().lower() or "upper"
        if mode == "lower":
            return price >= budget_vnd
        return price <= budget_vnd

    if source_engine is not None:
        try:
            with source_engine.connect() as connection:
                rows = connection.execute(text(PRODUCT_SYNC_QUERY)).mappings().all()
            mapped_rows = [dict(row) for row in rows]
            if explicit_category:
                hinted_rows = [row for row in mapped_rows if matches_explicit_device_category(row, explicit_category)]
            else:
                hinted_rows = [row for row in mapped_rows if matches_any_hint(row, category_hints)]
            if not hinted_rows:
                hinted_rows = mapped_rows
            filtered_rows = [row for row in hinted_rows if matches_terms(row, terms)]
            if not filtered_rows:
                filtered_rows = hinted_rows
            filtered_rows = [row for row in filtered_rows if _price_matches(row)]
            filtered_rows = sorted(
                filtered_rows,
                key=lambda row: (int(row.get("buyturn") or 0), float(row.get("product_rating") or 0.0)),
                reverse=True,
            )
            if filtered_rows:
                max_buyturn = max(int(row.get("buyturn") or 0) for row in filtered_rows)
                candidates = []
                for row in filtered_rows[:capped_limit]:
                    document = build_product_document(row)
                    document.update(
                        {
                            "chunk_id": f"product_search:product:{int(row['product_id'])}",
                            "chunk_index": 1,
                            "buyturn": int(row.get("buyturn") or 0),
                            "product_rating": row.get("product_rating"),
                            "product_price": row.get("product_price"),
                        }
                    )
                    candidates.append(
                        to_top_selling_candidate(document, max_buyturn=max_buyturn, source="source_db_product_search")
                    )
                return {
                    "retrieval_backend": "source_sql_product_search",
                    "vector_candidates": len(candidates),
                    "keyword_candidates": 0,
                    "candidates": candidates,
                }
        except Exception:
            pass

    product_chunks = [
        row
        for row in chatbot_repository.get_chunks(include_embeddings=False)
        if str((row.get("metadata") or {}).get("doc_type")) == "product"
    ]
    collapsed_rows = collapse_kb_product_chunks(product_chunks)
    filtered_rows = [
        row
        for row in collapsed_rows
        if (
            matches_explicit_device_category(row, explicit_category)
            if explicit_category
            else matches_any_hint(
                {
                    "product_name": row.get("title"),
                    "catalog_name": "",
                    "product_description": row.get("content"),
                    "catalog_description": "",
                    "brand_name": (row.get("metadata") or {}).get("brand"),
                    "category_name": (row.get("metadata") or {}).get("category"),
                },
                category_hints,
            )
        )
    ]
    if not filtered_rows:
        filtered_rows = collapsed_rows
    filtered_rows = [
        row
        for row in filtered_rows
        if matches_terms(
            {
                "product_name": row.get("title"),
                "catalog_name": "",
                "product_description": row.get("content"),
                "catalog_description": "",
                "brand_name": (row.get("metadata") or {}).get("brand"),
                "category_name": (row.get("metadata") or {}).get("category"),
            },
            terms,
        )
    ] or filtered_rows
    filtered_rows = [row for row in filtered_rows if _price_matches(row)]
    sorted_rows = sorted(
        filtered_rows,
        key=lambda item: (int((item.get("metadata") or {}).get("buyturn") or 0), int((item.get("metadata") or {}).get("price_vnd") or 0) > 0),
        reverse=True,
    )[:capped_limit]
    max_buyturn = max([int((item.get("metadata") or {}).get("buyturn") or 0) for item in sorted_rows] or [1])
    candidates = [
        to_top_selling_candidate(row, max_buyturn=max_buyturn, source="kb_product_search_fallback")
        for row in sorted_rows
    ]
    return {
        "retrieval_backend": "kb_product_search_fallback",
        "vector_candidates": len(candidates),
        "keyword_candidates": 0,
        "candidates": candidates,
    }
