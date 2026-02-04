from __future__ import annotations

"""Data sync and intent-specific retrieval helpers based on real product data."""

import re
import unicodedata
from typing import Any

from sqlalchemy import text

from app.db import source_engine
from app.repository import chatbot_repository

# Base query used by periodic sync to pull active products from backend DB.
PRODUCT_SYNC_QUERY = """
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
ORDER BY p.updated_at DESC, p.id DESC
"""

# Query optimized for "top selling" intent (ordered by buyturn/rating).
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

# Generic terms ignored when extracting product filters from top-selling query.
TOP_SELLING_STOPWORDS = {
    "san",
    "pham",
    "ban",
    "chay",
    "top",
    "nhieu",
    "nguoi",
    "mua",
    "loai",
    "nhung",
    "cac",
    "co",
    "the",
    "nao",
    "nhat",
    "best",
    "seller",
    "bestseller",
    "topselling",
    "hang",
}

# Generic words ignored for build-pc intent term extraction.
BUILD_PC_STOPWORDS = {
    "build",
    "pc",
    "cau",
    "hinh",
    "may",
    "tinh",
    "cho",
    "toi",
    "giup",
    "minh",
    "khoang",
    "tam",
    "duoi",
    "tren",
    "trieu",
    "tr",
    "vnd",
    "dong",
    "budget",
    "ngan",
    "sach",
}

# Canonical part order and keyword map used to classify products into PC roles.
PC_ROLE_ORDER = ["cpu", "gpu", "motherboard", "ram", "ssd", "psu", "case"]
PC_ROLE_KEYWORDS: dict[str, tuple[list[str], list[str]]] = {
    "cpu": (
        [" cpu ", "processor", "vi xu ly"],
        ["cpu cooler", "tan nhiet cpu"],
    ),
    "gpu": (
        ["video card", "gpu", "graphics card", "vga", "card do hoa", "card man hinh"],
        ["card mang", "card am thanh"],
    ),
    "motherboard": (
        ["motherboard", "mainboard", "bo mach chu"],
        [],
    ),
    "ram": ([" memory ", " ram ", "ddr"], []),
    "ssd": (
        [" ssd ", "nvme", "m 2", "solid state", "internal hard drive", "o cung trong"],
        [],
    ),
    "psu": (
        ["power supply", " psu ", "nguon may tinh", "bo nguon"],
        ["ups", "bo luu dien"],
    ),
    "case": (
        [" case ", "chassis", "vo case", "vo may"],
        ["case fan", "quat case", "phu kien vo case", "bo dieu khien quat"],
    ),
}

# Prefer category-based role mapping to avoid misclassifying non-component products
# (for example laptops mentioning CPU/GPU in specs).
PC_ROLE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cpu": ["cpu", "vi xu ly"],
    "gpu": ["card do hoa", "vga", "card man hinh", "gpu"],
    "motherboard": ["bo mach chu", "mainboard", "motherboard"],
    "ram": ["ram", "memory"],
    "ssd": ["ssd", "o cung trong", "nvme"],
    "psu": ["nguon may tinh", "psu", "power supply"],
    "case": ["vo case", "vo may", "chassis"],
}

PC_NON_BUILD_CATEGORY_KEYWORDS = {
    "laptop",
    "man hinh",
    "tai nghe",
    "chuot",
    "ban phim",
    "loa",
    "webcam",
    "card mang",
    "card am thanh",
    "ups",
    "bo luu dien",
    "he dieu hanh",
    "o dia quang",
    "quat case",
    "phu kien vo case",
    "bo dieu khien quat",
    "tan nhiet",
    "keo tan nhiet",
}

# Display labels used in final build-pc candidate content blocks.
PC_ROLE_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
    "motherboard": "Mainboard",
    "ram": "RAM",
    "ssd": "SSD",
    "psu": "PSU",
    "case": "Case",
}


def _format_specs(specs: Any) -> str:
    """Render arbitrary specs payload to compact single text line."""

    if not specs:
        return ""
    if isinstance(specs, dict):
        parts = []
        for key, value in specs.items():
            parts.append(f"{key}: {value}")
        return "; ".join(parts)
    return str(specs)


def _build_product_document(row: dict[str, Any]) -> dict[str, Any]:
    """Map one source SQL row to normalized knowledge document payload."""

    title = str(row.get("product_name") or row.get("catalog_name") or f"Product {row['product_id']}")
    description = str(row.get("product_description") or row.get("catalog_description") or "").strip()
    specs_text = _format_specs(row.get("catalog_specs"))
    location = " - ".join(
        [item for item in [row.get("store_city"), row.get("store_province")] if item]
    )
    content_parts = [
        f"Ten san pham: {title}",
        f"Category: {row.get('category_name') or 'N/A'}",
        f"Brand: {row.get('brand_name') or 'N/A'}",
        f"Shop: {row.get('store_name') or 'N/A'}",
        f"Gia: {row.get('product_price') or 'N/A'}",
        f"So luong: {row.get('quantity') or 0}",
        f"Chat luong: {row.get('product_quality') or 'N/A'}",
        f"Condition percent: {row.get('condition_percent') or 'N/A'}",
        f"Rating: {row.get('product_rating') or 'N/A'}",
        f"Luot mua: {row.get('buyturn') or 0}",
    ]
    if location:
        content_parts.append(f"Dia diem shop: {location}")
    if description:
        content_parts.append(f"Mo ta: {description}")
    if specs_text:
        content_parts.append(f"Thong so: {specs_text}")

    updated_at = row.get("product_updated_at") or row.get("catalog_updated_at")
    updated_iso = updated_at.isoformat() if updated_at else None

    return {
        "source_key": f"product:{row['product_id']}",
        "title": title,
        "uri": f"/products/{row['product_id']}",
        "content": "\n".join(content_parts),
        "metadata": {
            "doc_type": "product",
            "category": row.get("category_name") or "unknown",
            "brand": row.get("brand_name") or "",
            "store_name": row.get("store_name") or "",
            "product_id": row["product_id"],
            "buyturn": int(row.get("buyturn") or 0),
            "product_status": row.get("product_status") or "",
            "trust_score": 0.93,
            "updated_at": updated_iso,
            "source": "backend_db",
        },
        "updated_at": updated_iso,
    }


def _strip_accents(text: str) -> str:
    """Remove accents so terms match both accented and non-accented queries."""

    decomposed = unicodedata.normalize("NFD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    """Normalize input to lowercase alphanumeric tokens joined by spaces."""

    lowered = _strip_accents(text).lower()
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered).split())


def _extract_product_terms(query: str, limit: int = 4) -> list[str]:
    """Extract limited meaningful terms for top-selling filtering."""

    terms: list[str] = []
    for token in _normalize(query).split():
        if len(token) < 2 or token in TOP_SELLING_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _extract_build_terms(query: str, limit: int = 5) -> list[str]:
    """Extract limited meaningful terms for build-pc filtering."""

    terms: list[str] = []
    for token in _normalize(query).split():
        if len(token) < 2:
            continue
        if token in BUILD_PC_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _matches_terms(row: dict[str, Any], terms: list[str]) -> bool:
    """Check whether a product-like row matches all extracted terms."""

    if not terms:
        return True
    text_blob = _normalize(
        " ".join(
            [
                str(row.get("product_name") or ""),
                str(row.get("catalog_name") or ""),
                str(row.get("product_description") or ""),
                str(row.get("catalog_description") or ""),
                str(row.get("brand_name") or ""),
                str(row.get("category_name") or ""),
            ]
        )
    )
    return all(term in text_blob for term in terms)


def _to_float(value: Any) -> float:
    """Parse loosely formatted numeric value to float with safe fallback."""

    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


def _parse_price_from_chunk_content(content: str) -> float:
    """Parse price value from chunk text content when metadata is missing."""

    text = str(content or "")
    line_match = re.search(r"Gia:\s*([0-9][0-9,.\s]*)", text, flags=re.IGNORECASE)
    if line_match:
        return _to_float(line_match.group(1))
    number_match = re.search(r"([0-9][0-9,.\s]{4,})", text)
    if number_match:
        return _to_float(number_match.group(1))
    return 0.0


def _extract_buyturn_from_content(content: str) -> int:
    """Parse buyturn count from chunk text content when metadata is missing."""

    match = re.search(r"luot mua:\s*(\d+)", str(content or "").lower())
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _match_pc_role(row: dict[str, Any]) -> str | None:
    """Classify one product row into a PC role using keyword heuristics."""

    category_text = _normalize(str(row.get("category_name") or ""))
    if category_text:
        if any(keyword in category_text for keyword in PC_NON_BUILD_CATEGORY_KEYWORDS):
            return None
        for role in PC_ROLE_ORDER:
            if any(keyword in category_text for keyword in PC_ROLE_CATEGORY_KEYWORDS[role]):
                return role
        # Category exists but is outside known build-PC component groups.
        return None

    normalized = _normalize(
        " ".join(
            [
                str(row.get("category_name") or ""),
                str(row.get("product_name") or row.get("title") or ""),
                str(row.get("product_description") or row.get("content") or ""),
                str(row.get("catalog_description") or ""),
                _format_specs(row.get("catalog_specs")),
            ]
        )
    )
    padded = f" {normalized} "
    for role in PC_ROLE_ORDER:
        include_keywords, exclude_keywords = PC_ROLE_KEYWORDS[role]
        if any(keyword in padded for keyword in include_keywords):
            if any(keyword in padded for keyword in exclude_keywords):
                continue
            return role
    return None


def _build_pc_candidate(
    row: dict[str, Any],
    role: str,
    budget_vnd: int,
    max_buyturn: int,
    source: str,
) -> dict[str, Any]:
    """Build retriever candidate record for one build-pc product row."""

    product_id = int(row.get("product_id") or 0)
    price = _to_float(row.get("product_price") or row.get("price"))
    rating = _to_float(row.get("product_rating") or row.get("rating"))
    buyturn = int(row.get("buyturn") or 0)
    title = str(row.get("product_name") or row.get("title") or f"Product {product_id}")
    uri = str(row.get("uri") or f"/products/{product_id}")
    brand = str(row.get("brand_name") or (row.get("metadata") or {}).get("brand") or "")
    category = str(row.get("category_name") or (row.get("metadata") or {}).get("category") or "")
    quantity = int(row.get("quantity") or 0)
    normalized_buyturn = (buyturn / max_buyturn) if max_buyturn > 0 else 0.0
    price_ratio = (price / float(max(budget_vnd, 1))) if price > 0 else 1.0
    affordability = max(0.0, min(1.0, 1.0 - price_ratio))
    rating_score = max(0.0, min(1.0, rating / 5.0))
    retrieval_score = max(
        0.0,
        min(1.0, normalized_buyturn * 0.35 + rating_score * 0.35 + affordability * 0.3),
    )
    updated_at = row.get("updated_at") or row.get("product_updated_at")
    updated_iso = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or "")
    content = "\n".join(
        [
            f"PC role: {PC_ROLE_LABELS.get(role, role)}",
            f"Ten san pham: {title}",
            f"Category: {category or 'N/A'}",
            f"Brand: {brand or 'N/A'}",
            f"Gia VND: {int(round(price)) if price else 0}",
            f"Rating: {rating}",
            f"Luot mua: {buyturn}",
            f"So luong: {quantity}",
            f"Link: {uri}",
        ]
    )
    return {
        "chunk_id": str(row.get("chunk_id") or f"buildpc:{role}:product:{product_id}"),
        "title": title,
        "uri": uri,
        "content": content,
        "metadata": {
            "doc_type": "product",
            "pc_role": role,
            "category": category or "unknown",
            "brand": brand,
            "product_id": product_id,
            "price_vnd": int(round(price)) if price else 0,
            "buyturn": buyturn,
            "trust_score": 0.95,
            "updated_at": updated_iso,
            "source": source,
            "ranking_signal": "build_pc_pool",
        },
        "chunk_index": int(row.get("chunk_index") or 1),
        "semantic_score": retrieval_score,
        "keyword_score": 0.0,
        "retrieval_score": retrieval_score,
    }


def _rank_pc_rows(
    rows: list[dict[str, Any]],
    budget_vnd: int,
    query_terms: list[str],
) -> list[dict[str, Any]]:
    """Rank products for one role by rating, demand, affordability, and term hits."""

    if not rows:
        return []
    max_buyturn = max(int(item.get("buyturn") or 0) for item in rows) or 1
    ranked_rows = []
    for row in rows:
        price = _to_float(row.get("product_price") or row.get("price"))
        if price <= 0:
            continue
        if price > budget_vnd * 0.9:
            continue
        rating = _to_float(row.get("product_rating") or row.get("rating"))
        buyturn = int(row.get("buyturn") or 0)
        normalized_buyturn = buyturn / max_buyturn
        rating_score = max(0.0, min(1.0, rating / 5.0))
        affordability = max(0.0, min(1.0, 1 - (price / float(max(budget_vnd, 1)))))
        text_blob = _normalize(
            " ".join(
                [
                    str(row.get("product_name") or row.get("title") or ""),
                    str(row.get("category_name") or ""),
                    str(row.get("brand_name") or ""),
                    str(row.get("product_description") or row.get("content") or ""),
                ]
            )
        )
        term_hit = 0.0
        if query_terms:
            hit_count = sum(1 for term in query_terms if term in text_blob)
            term_hit = hit_count / len(query_terms)
        total_score = rating_score * 0.35 + normalized_buyturn * 0.35 + affordability * 0.2 + term_hit * 0.1
        ranked_rows.append({**row, "_score": total_score})
    return sorted(ranked_rows, key=lambda item: item["_score"], reverse=True)


def _fetch_source_rows() -> list[dict[str, Any]]:
    """Fetch active products directly from source database."""

    if source_engine is None:
        return []
    with source_engine.connect() as connection:
        rows = connection.execute(text(PRODUCT_SYNC_QUERY)).mappings().all()
    return [dict(row) for row in rows]


def _fetch_kb_product_rows() -> list[dict[str, Any]]:
    """Build product-like fallback rows from existing KB chunks."""

    rows = []
    for chunk in chatbot_repository.get_chunks(include_embeddings=False):
        metadata = dict(chunk.get("metadata") or {})
        if str(metadata.get("doc_type")) != "product":
            continue
        product_id = int(metadata.get("product_id") or 0)
        if product_id <= 0:
            product_match = re.search(r"/products/(\d+)", str(chunk.get("uri") or ""))
            if product_match:
                product_id = int(product_match.group(1))
        content = str(chunk.get("content") or "")
        rating_match = re.search(r"rating:\s*([0-9.]+)", content.lower())
        quantity_match = re.search(r"so luong:\s*(\d+)", content.lower())
        rows.append(
            {
                "product_id": product_id,
                "product_name": chunk.get("title"),
                "category_name": metadata.get("category"),
                "brand_name": metadata.get("brand"),
                "buyturn": int(
                    _to_float(metadata.get("buyturn"))
                    or _extract_buyturn_from_content(content)
                ),
                "product_rating": _to_float(rating_match.group(1)) if rating_match else 0.0,
                "product_price": _to_float(metadata.get("price_vnd")) or _parse_price_from_chunk_content(content),
                "quantity": int(quantity_match.group(1)) if quantity_match else 0,
                "product_status": metadata.get("product_status") or "active",
                "uri": chunk.get("uri") or f"/products/{product_id}",
                "title": chunk.get("title"),
                "content": content,
                "metadata": metadata,
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index") or 1,
                "updated_at": metadata.get("updated_at"),
            }
        )
    return rows


def _to_top_selling_candidate(
    row: dict[str, Any],
    max_buyturn: int,
    source: str,
) -> dict[str, Any]:
    """Convert one row/chunk into normalized top-selling candidate shape."""

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
    """Return top-selling product candidates (source DB first, KB fallback)."""

    capped_limit = max(1, min(int(limit), 50))
    terms = _extract_product_terms(query)

    if source_engine is not None:
        try:
            with source_engine.connect() as connection:
                rows = connection.execute(
                    text(TOP_SELLING_QUERY),
                    {"limit": max(capped_limit * 6, 30)},
                ).mappings().all()
            mapped_rows = [dict(row) for row in rows]
            filtered_rows = [row for row in mapped_rows if _matches_terms(row, terms)]
            if filtered_rows:
                max_buyturn = max(int(row.get("buyturn") or 0) for row in filtered_rows)
                candidates = []
                for row in filtered_rows[:capped_limit]:
                    document = _build_product_document(row)
                    document.update(
                        {
                            "chunk_id": f"top_selling:product:{int(row['product_id'])}",
                            "chunk_index": 1,
                            "buyturn": int(row.get("buyturn") or 0),
                        }
                    )
                    candidates.append(
                        _to_top_selling_candidate(
                            document,
                            max_buyturn=max_buyturn,
                            source="source_db_top_selling",
                        )
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
        return {
            "retrieval_backend": "top_selling_unavailable",
            "vector_candidates": 0,
            "keyword_candidates": 0,
            "candidates": [],
        }

    for row in product_chunks:
        metadata = dict(row.get("metadata") or {})
        buyturn_value = metadata.get("buyturn")
        if buyturn_value is None:
            buyturn_value = _extract_buyturn_from_content(row.get("content", ""))
        metadata["buyturn"] = int(buyturn_value or 0)
        row["metadata"] = metadata
        row["buyturn"] = int(buyturn_value or 0)
        row["product_id"] = int(metadata.get("product_id") or 0)
        if row["product_id"] <= 0:
            product_match = re.search(r"/products/(\d+)", str(row.get("uri") or ""))
            if product_match:
                row["product_id"] = int(product_match.group(1))

    filtered_chunks = [
        row
        for row in product_chunks
        if _matches_terms(
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
    sorted_chunks = sorted(
        filtered_chunks,
        key=lambda item: int(item.get("buyturn") or 0),
        reverse=True,
    )[:capped_limit]
    max_buyturn = max([int(item.get("buyturn") or 0) for item in sorted_chunks] or [1])
    fallback_candidates = [
        _to_top_selling_candidate(
            row,
            max_buyturn=max_buyturn,
            source="kb_fallback_top_selling",
        )
        for row in sorted_chunks
    ]
    return {
        "retrieval_backend": "kb_top_selling_fallback",
        "vector_candidates": len(fallback_candidates),
        "keyword_candidates": 0,
        "candidates": fallback_candidates,
    }


def get_build_pc_candidates(
    query: str,
    budget_vnd: int,
    per_role: int = 4,
) -> dict[str, Any]:
    """Build candidate pool for PC assembly under a target budget."""

    safe_per_role = max(1, min(int(per_role), 8))
    safe_budget = max(int(budget_vnd), 1)
    query_terms = _extract_build_terms(query)

    def _select_rows(rows: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
        if not rows:
            return {
                "retrieval_backend": f"{source_name}_empty",
                "vector_candidates": 0,
                "keyword_candidates": 0,
                "candidates": [],
            }

        grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in PC_ROLE_ORDER}
        for row in rows:
            quantity = int(row.get("quantity") or 0)
            if quantity <= 0:
                continue
            role = _match_pc_role(row)
            if role is None:
                continue
            grouped[role].append(row)

        selected_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in PC_ROLE_ORDER}
        for role in PC_ROLE_ORDER:
            ranked_rows = _rank_pc_rows(grouped[role], safe_budget, query_terms)
            if ranked_rows:
                selected_rows[role] = ranked_rows[:safe_per_role]

        if not all(selected_rows[role] for role in PC_ROLE_ORDER):
            return {
                "retrieval_backend": f"{source_name}_insufficient_parts",
                "vector_candidates": 0,
                "keyword_candidates": 0,
                "candidates": [],
            }

        max_buyturn = max(
            [int(item.get("buyturn") or 0) for role in PC_ROLE_ORDER for item in selected_rows[role]]
            or [1]
        )
        interleaved: list[dict[str, Any]] = []
        for slot in range(safe_per_role):
            for role in PC_ROLE_ORDER:
                items = selected_rows[role]
                if slot >= len(items):
                    continue
                interleaved.append(
                    _build_pc_candidate(
                        items[slot],
                        role=role,
                        budget_vnd=safe_budget,
                        max_buyturn=max_buyturn,
                        source=source_name,
                    )
                )

        return {
            "retrieval_backend": source_name,
            "vector_candidates": len(interleaved),
            "keyword_candidates": 0,
            "candidates": interleaved,
        }

    source_result = _select_rows(_fetch_source_rows(), "source_sql_build_pc")
    if source_result["candidates"]:
        return source_result

    kb_result = _select_rows(_fetch_kb_product_rows(), "kb_build_pc_fallback")
    if kb_result["candidates"]:
        return kb_result

    return {
        "retrieval_backend": "build_pc_unavailable",
        "vector_candidates": 0,
        "keyword_candidates": 0,
        "candidates": [],
    }


async def sync_from_source_database() -> dict[str, Any]:
    """Sync active products from source DB into chatbot knowledge base."""

    if source_engine is None:
        return {"status": "skipped", "reason": "SOURCE_DATABASE_URL is not configured"}

    try:
        with source_engine.connect() as connection:
            rows = connection.execute(text(PRODUCT_SYNC_QUERY)).mappings().all()
        documents = [_build_product_document(dict(row)) for row in rows]
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
