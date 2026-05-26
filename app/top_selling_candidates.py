from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from app.product_source_utils import (
    build_product_document,
    collapse_kb_product_chunks,
    derive_device_category,
    extract_buyturn_from_content,
    extract_product_terms,
    extract_top_selling_category_hints,
    fetch_kb_product_rows,
    matches_any_hint,
    matches_explicit_device_category,
    matches_terms,
    to_float,
)
from app.repository import chatbot_repository
from app.db import source_engine

TOP_SELLING_QUERY = """
SELECT
    p.id AS product_id,
    p.name AS product_name,
    p.description AS product_description,
    p.price AS product_price,
    p.quality AS product_quality,
    p.condition_percent AS condition_percent,
    p.rating AS product_rating,
    p.buyturn AS buyturn,
    p.quantity AS quantity,
    p.status AS product_status,
    p.updated_at AS product_updated_at,
    pc.name AS catalog_name,
    pc.description AS catalog_description,
    pc.specs AS catalog_specs,
    pc.updated_at AS catalog_updated_at,
    b.name AS brand_name,
    c.id AS category_id,
    c.slug AS category_slug,
    c.name AS category_name,
    s.name AS store_name,
    s.city AS store_city,
    s.province AS store_province
FROM products p
LEFT JOIN product_catalog pc ON pc.id = p.catalog_id
LEFT JOIN brand b ON b.id = p.brand_id
LEFT JOIN product_categories c ON c.id = p.category_id
LEFT JOIN stores s ON s.id = p.store_id
WHERE p.status = 'active'
ORDER BY COALESCE(p.buyturn, 0) DESC, COALESCE(p.rating, 0) DESC, p.updated_at DESC, p.id DESC
LIMIT :limit
"""


def to_top_selling_candidate(
    row: dict[str, Any],
    max_buyturn: int,
    source: str,
) -> dict[str, Any]:
    product_id = int(row.get("product_id") or 0)
    title = str(row.get("product_name") or row.get("title") or f"Product {product_id}")
    uri = str(row.get("uri") or f"/products/{product_id}")
    buyturn = int(row.get("buyturn") or 0)
    rating_raw = row.get("product_rating")
    try:
        rating = float(rating_raw) if rating_raw is not None else 0.0
    except (TypeError, ValueError):
        rating = 0.0
    normalized_buyturn = (buyturn / max_buyturn) if max_buyturn > 0 else 0.0
    rating_score = max(0.0, min(1.0, rating / 5.0))
    retrieval_score = max(0.0, min(1.0, normalized_buyturn * 0.8 + rating_score * 0.2))
    updated_at = row.get("updated_at") or row.get("product_updated_at")
    updated_iso = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or "")
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "doc_type": "product",
            "product_id": product_id,
            "buyturn": buyturn,
            "trust_score": float(metadata.get("trust_score", 0.93)),
            "updated_at": updated_iso or metadata.get("updated_at"),
            "source": source,
            "ranking_signal": "buyturn",
        }
    )
    return {
        "chunk_id": str(row.get("chunk_id") or f"top_selling:product:{product_id}"),
        "title": title,
        "uri": uri,
        "content": str(row.get("content") or ""),
        "metadata": metadata,
        "chunk_index": int(row.get("chunk_index") or 1),
        "semantic_score": retrieval_score,
        "keyword_score": 0.0,
        "retrieval_score": retrieval_score,
    }


def get_top_selling_candidates(query: str, limit: int = 12) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit), 50))
    terms = extract_product_terms(query)
    category_hints = extract_top_selling_category_hints(query)
    explicit_category = derive_device_category(query)

    if source_engine is not None:
        try:
            with source_engine.connect() as connection:
                rows = connection.execute(text(TOP_SELLING_QUERY), {"limit": max(capped_limit * 6, 30)}).mappings().all()
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
            if filtered_rows:
                max_buyturn = max(int(row.get("buyturn") or 0) for row in filtered_rows)
                candidates = []
                for row in filtered_rows[:capped_limit]:
                    document = build_product_document(row)
                    document.update(
                        {
                            "chunk_id": f"top_selling:product:{int(row['product_id'])}",
                            "chunk_index": 1,
                            "buyturn": int(row.get("buyturn") or 0),
                            "product_rating": row.get("product_rating"),
                            "product_price": row.get("product_price"),
                        }
                    )
                    candidates.append(
                        to_top_selling_candidate(document, max_buyturn=max_buyturn, source="source_db_top_selling")
                    )
                return {
                    "retrieval_backend": "source_sql_top_selling",
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
    if not product_chunks:
        return {"retrieval_backend": "top_selling_unavailable", "vector_candidates": 0, "keyword_candidates": 0, "candidates": []}

    for row in product_chunks:
        metadata = dict(row.get("metadata") or {})
        buyturn_value = metadata.get("buyturn")
        if buyturn_value is None:
            buyturn_value = extract_buyturn_from_content(row.get("content", ""))
        metadata["buyturn"] = int(buyturn_value or 0)
        row["metadata"] = metadata
        row["buyturn"] = int(buyturn_value or 0)
        row["product_id"] = int(metadata.get("product_id") or 0)
        if row["product_id"] <= 0:
            product_match = re.search(r"/products/(\d+)", str(row.get("uri") or ""))
            if product_match:
                row["product_id"] = int(product_match.group(1))

    product_rows = collapse_kb_product_chunks(product_chunks)
    if explicit_category:
        filtered_chunks = [row for row in product_rows if matches_explicit_device_category(row, explicit_category)]
    else:
        filtered_chunks = [
            row
            for row in product_rows
            if matches_any_hint(
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
        ]
    if not filtered_chunks:
        filtered_chunks = product_rows

    filtered_chunks = [
        row
        for row in filtered_chunks
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
    ]
    if not filtered_chunks:
        filtered_chunks = [
            row
            for row in product_rows
            if matches_any_hint(
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
        ]
    if not filtered_chunks:
        filtered_chunks = product_rows
    sorted_chunks = sorted(filtered_chunks, key=lambda item: int(item.get("buyturn") or 0), reverse=True)[:capped_limit]
    max_buyturn = max([int(item.get("buyturn") or 0) for item in sorted_chunks] or [1])
    fallback_candidates = [
        to_top_selling_candidate(row, max_buyturn=max_buyturn, source="kb_fallback_top_selling")
        for row in sorted_chunks
    ]
    return {
        "retrieval_backend": "kb_top_selling_fallback",
        "vector_candidates": len(fallback_candidates),
        "keyword_candidates": 0,
        "candidates": fallback_candidates,
    }
