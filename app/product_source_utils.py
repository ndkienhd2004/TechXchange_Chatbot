from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from app.db import source_engine
from app.repository import chatbot_repository
from app.text_normalization import normalize

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
ORDER BY p.updated_at DESC, p.id DESC
"""

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
    "may",
    "tinh",
    "dien",
    "thoai",
    "laptop",
    "pc",
    "desktop",
    "computer",
}

TOP_SELLING_CATEGORY_HINTS: dict[str, list[str]] = {
    "dien thoai": ["dien thoai", "smartphone", "iphone", "samsung"],
    "laptop": ["laptop", "notebook"],
    "may tinh": [
        "laptop",
        "cpu",
        "bo mach chu",
        "ram",
        "ssd",
        "nguon may tinh",
        "vo case",
        "card do hoa",
        "mainboard",
        "gpu",
    ],
    "pc": [
        "laptop",
        "cpu",
        "bo mach chu",
        "ram",
        "ssd",
        "nguon may tinh",
        "vo case",
        "card do hoa",
        "mainboard",
        "gpu",
    ],
    "tai nghe": ["tai nghe", "headphone", "earbuds"],
    "man hinh": ["man hinh", "monitor"],
    "chuot": ["chuot", "mouse"],
    "ban phim": ["ban phim", "keyboard"],
}


def format_specs(specs: Any) -> str:
    if not specs:
        return ""
    if isinstance(specs, dict):
        return "; ".join(f"{key}: {value}" for key, value in specs.items())
    return str(specs)


def build_product_document(row: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("product_name") or row.get("catalog_name") or f"Product {row['product_id']}")
    description = str(row.get("product_description") or row.get("catalog_description") or "").strip()
    specs_text = format_specs(row.get("catalog_specs"))
    location = " - ".join([item for item in [row.get("store_city"), row.get("store_province")] if item])
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
            "category_id": row.get("category_id"),
            "category_slug": str(row.get("category_slug") or "").strip(),
            "brand": row.get("brand_name") or "",
            "store_name": row.get("store_name") or "",
            "product_id": row["product_id"],
            "price_vnd": int(to_float(row.get("product_price")) or 0),
            "buyturn": int(row.get("buyturn") or 0),
            "product_status": row.get("product_status") or "",
            "trust_score": 0.93,
            "updated_at": updated_iso,
            "source": "backend_db",
        },
        "updated_at": updated_iso,
    }
def extract_product_terms(query: str, limit: int = 4) -> list[str]:
    terms: list[str] = []
    for token in normalize(query).split():
        if len(token) < 2 or token in TOP_SELLING_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def extract_top_selling_category_hints(query: str) -> list[str]:
    normalized = normalize(query)
    hints: list[str] = []
    for phrase, mapped_hints in TOP_SELLING_CATEGORY_HINTS.items():
        if phrase in normalized:
            for hint in mapped_hints:
                if hint not in hints:
                    hints.append(hint)
    return hints


def derive_device_category(query: str) -> str | None:
    normalized = normalize(query)
    collapsed = normalized.replace(" ", "")
    if any(marker in normalized for marker in ("laptop", "notebook", "macbook", "may tinh xach tay")):
        return "laptop"
    if any(marker in normalized for marker in ("dien thoai", "smartphone", "iphone")):
        return "phone"
    if any(marker in normalized for marker in ("tablet", "ipad", "may tinh bang")):
        return "tablet"
    if any(marker in normalized for marker in ("pc", "desktop", "may tinh de ban")) or "pc" in collapsed:
        return "pc"
    return None


def matches_explicit_device_category(row: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    category_norm = normalize(
        " ".join(
            [
                str(row.get("category_name") or ""),
                str(row.get("category_slug") or "").replace("-", " "),
                str((row.get("metadata") or {}).get("category") or ""),
                str((row.get("metadata") or {}).get("category_slug") or "").replace("-", " "),
            ]
        )
    )
    if not category_norm:
        return False
    if category == "laptop":
        return any(token in category_norm for token in ("laptop", "notebook", "macbook", "xach tay"))
    if category == "phone":
        return any(token in category_norm for token in ("dien thoai", "smartphone", "iphone", "mobile"))
    if category == "tablet":
        return any(token in category_norm for token in ("tablet", "ipad"))
    if category == "pc":
        return any(token in category_norm for token in ("pc", "desktop", "may tinh de ban"))
    return category in category_norm


def matches_terms(row: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return True
    text_blob = normalize(
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


def matches_any_hint(row: dict[str, Any], hints: list[str]) -> bool:
    if not hints:
        return True
    text_blob = normalize(
        " ".join(
            [
                str(row.get("product_name") or row.get("title") or ""),
                str(row.get("catalog_name") or ""),
                str(row.get("product_description") or row.get("content") or ""),
                str(row.get("catalog_description") or ""),
                str(row.get("brand_name") or ""),
                str(row.get("category_name") or (row.get("metadata") or {}).get("category") or ""),
            ]
        )
    )
    return any(hint in text_blob for hint in hints)


def to_float(value: Any) -> float:
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


def parse_price_from_chunk_content(content: str) -> float:
    text = str(content or "")
    line_match = re.search(r"Gia:\s*([0-9][0-9,.\s]*)", text, flags=re.IGNORECASE)
    if line_match:
        return to_float(line_match.group(1))
    number_match = re.search(r"([0-9][0-9,.\s]{4,})", text)
    if number_match:
        return to_float(number_match.group(1))
    return 0.0


def extract_buyturn_from_content(content: str) -> int:
    match = re.search(r"luot mua:\s*(\d+)", str(content or "").lower())
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def fetch_source_rows() -> list[dict[str, Any]]:
    if source_engine is None:
        return []
    with source_engine.connect() as connection:
        rows = connection.execute(text(PRODUCT_SYNC_QUERY)).mappings().all()
    return [dict(row) for row in rows]


def fetch_kb_product_rows() -> list[dict[str, Any]]:
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
                "category_slug": metadata.get("category_slug"),
                "brand_name": metadata.get("brand"),
                "buyturn": int(to_float(metadata.get("buyturn")) or extract_buyturn_from_content(content)),
                "product_rating": to_float(rating_match.group(1)) if rating_match else 0.0,
                "product_price": to_float(metadata.get("price_vnd")) or parse_price_from_chunk_content(content),
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


def pick_preferred_product_chunk(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def _sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
        metadata = dict(item.get("metadata") or {})
        has_price = 1 if int(metadata.get("price_vnd") or 0) > 0 else 0
        chunk_index = int(item.get("chunk_index") or 9999)
        content_len = len(str(item.get("content") or ""))
        return (has_price, -chunk_index, content_len)

    return sorted(candidates, key=_sort_key, reverse=True)[0]


def collapse_kb_product_chunks(product_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in product_chunks:
        metadata = dict(row.get("metadata") or {})
        product_id = int(metadata.get("product_id") or row.get("product_id") or 0)
        if product_id <= 0:
            product_match = re.search(r"/products/(\d+)", str(row.get("uri") or ""))
            if product_match:
                product_id = int(product_match.group(1))
        if product_id <= 0:
            continue
        metadata["product_id"] = product_id
        row["metadata"] = metadata
        row["product_id"] = product_id
        grouped.setdefault(product_id, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for product_id, rows in grouped.items():
        representative = dict(pick_preferred_product_chunk(rows))
        representative["product_id"] = product_id
        representative["chunk_id"] = str(representative.get("chunk_id") or f"product:{product_id}:representative")
        collapsed.append(representative)
    return collapsed
