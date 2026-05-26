from __future__ import annotations

from typing import Any

from app.product_source_utils import (
    fetch_kb_product_rows,
    fetch_source_rows,
    format_specs,
    normalize,
    to_float,
)

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

PC_ROLE_ORDER = ["cpu", "gpu", "motherboard", "ram", "ssd", "psu", "case"]
PC_ROLE_KEYWORDS: dict[str, tuple[list[str], list[str]]] = {
    "cpu": ([" cpu ", "processor", "vi xu ly"], ["cpu cooler", "tan nhiet cpu"]),
    "gpu": (["video card", "gpu", "graphics card", "vga", "card do hoa", "card man hinh"], ["card mang", "card am thanh"]),
    "motherboard": (["motherboard", "mainboard", "bo mach chu"], []),
    "ram": ([" memory ", " ram ", "ddr"], []),
    "ssd": ([" ssd ", "nvme", "m 2", "solid state", "internal hard drive", "o cung trong"], []),
    "psu": (["power supply", " psu ", "nguon may tinh", "bo nguon"], ["ups", "bo luu dien"]),
    "case": ([" case ", "chassis", "vo case", "vo may"], ["case fan", "quat case", "phu kien vo case", "bo dieu khien quat"]),
}
PC_ROLE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cpu": ["cpu", "vi xu ly", "bộ vi xu ly"],
    "gpu": ["card do hoa", "vga", "card man hinh", "gpu", "card đồ họa", "card màn hình"],
    "motherboard": ["bo mach chu", "mainboard", "motherboard", "bo mạch chủ"],
    "ram": ["ram", "memory", "bo nho", "bộ nhớ"],
    "ssd": ["ssd", "o cung trong", "nvme", "ổ cứng trong"],
    "psu": ["nguon may tinh", "psu", "power supply", "nguon", "nguồn máy tính"],
    "case": ["vo case", "vo may", "chassis", "vỏ case", "vỏ máy"],
}
PC_NON_BUILD_CATEGORY_KEYWORDS = {
    "laptop", "man hinh", "tai nghe", "chuot", "ban phim", "loa", "webcam", "card mang", "card am thanh",
    "ups", "bo luu dien", "he dieu hanh", "o dia quang", "quat case", "phu kien vo case", "bo dieu khien quat", "tan nhiet", "keo tan nhiet",
}
PC_NON_COMPONENT_PRODUCT_KEYWORDS = {"laptop", "macbook", "notebook", "ultrabook", "chromebook", "imac", "all in one", "aio", "mini pc", "pc gaming", "bo pc", "desktop pc"}
PC_ROLE_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
    "motherboard": "Mainboard",
    "ram": "RAM",
    "ssd": "SSD",
    "psu": "PSU",
    "case": "Case",
}


def extract_build_terms(query: str, limit: int = 5) -> list[str]:
    terms: list[str] = []
    for token in normalize(query).split():
        if len(token) < 2 or token in BUILD_PC_STOPWORDS or token.isdigit():
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def match_pc_role(row: dict[str, Any]) -> str | None:
    slug_text = normalize(str(row.get("category_slug") or "").replace("-", " "))
    category_text = normalize(str(row.get("category_name") or ""))
    product_text = normalize(
        " ".join(
            [
                str(row.get("product_name") or row.get("title") or ""),
                str(row.get("product_description") or row.get("content") or ""),
                str(row.get("catalog_description") or ""),
            ]
        )
    )
    if any(keyword in product_text for keyword in PC_NON_COMPONENT_PRODUCT_KEYWORDS):
        return None

    merged_category = f"{category_text} {slug_text}".strip()
    if merged_category:
        if any(keyword in merged_category for keyword in PC_NON_BUILD_CATEGORY_KEYWORDS):
            return None
        for role in PC_ROLE_ORDER:
            if any(keyword in merged_category for keyword in PC_ROLE_CATEGORY_KEYWORDS[role]):
                return role
        return None

    normalized = normalize(
        " ".join(
            [
                str(row.get("category_name") or ""),
                str(row.get("category_slug") or "").replace("-", " "),
                str(row.get("product_name") or row.get("title") or ""),
                str(row.get("product_description") or row.get("content") or ""),
                str(row.get("catalog_description") or ""),
                format_specs(row.get("catalog_specs")),
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


def build_pc_candidate(row: dict[str, Any], role: str, budget_vnd: int, max_buyturn: int, source: str) -> dict[str, Any]:
    product_id = int(row.get("product_id") or 0)
    price = to_float(row.get("product_price") or row.get("price"))
    rating = to_float(row.get("product_rating") or row.get("rating"))
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
    retrieval_score = max(0.0, min(1.0, normalized_buyturn * 0.35 + rating_score * 0.35 + affordability * 0.3))
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


def rank_pc_rows(rows: list[dict[str, Any]], budget_vnd: int, query_terms: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    max_buyturn = max(int(item.get("buyturn") or 0) for item in rows) or 1
    ranked_rows = []
    for row in rows:
        price = to_float(row.get("product_price") or row.get("price"))
        if price <= 0 or price > budget_vnd * 1.8:
            continue
        rating = to_float(row.get("product_rating") or row.get("rating"))
        buyturn = int(row.get("buyturn") or 0)
        normalized_buyturn = buyturn / max_buyturn
        rating_score = max(0.0, min(1.0, rating / 5.0))
        affordability = max(0.0, min(1.0, 1 - (price / float(max(budget_vnd, 1)))))
        text_blob = normalize(
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


def get_build_pc_candidates(query: str, budget_vnd: int, per_role: int = 4) -> dict[str, Any]:
    safe_per_role = max(1, min(int(per_role), 8))
    safe_budget = max(int(budget_vnd), 1)
    query_terms = extract_build_terms(query)

    def _select_rows(rows: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
        if not rows:
            return {"retrieval_backend": f"{source_name}_empty", "vector_candidates": 0, "keyword_candidates": 0, "candidates": []}

        grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in PC_ROLE_ORDER}
        for row in rows:
            quantity = int(row.get("quantity") or 0)
            if quantity <= 0:
                continue
            role = match_pc_role(row)
            if role is None:
                continue
            grouped[role].append(row)

        selected_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in PC_ROLE_ORDER}
        for role in PC_ROLE_ORDER:
            ranked_rows = rank_pc_rows(grouped[role], safe_budget, query_terms)
            if ranked_rows:
                top_by_score = ranked_rows[:safe_per_role]
                top_by_price = sorted(ranked_rows, key=lambda item: to_float(item.get("product_price")), reverse=True)[: max(1, safe_per_role // 2)]
                merged: list[dict[str, Any]] = []
                seen_ids: set[int] = set()
                for row in [*top_by_score, *top_by_price]:
                    product_id = int(row.get("product_id") or 0)
                    if product_id in seen_ids:
                        continue
                    seen_ids.add(product_id)
                    merged.append(row)
                    if len(merged) >= safe_per_role:
                        break
                selected_rows[role] = merged

        if not all(selected_rows[role] for role in PC_ROLE_ORDER):
            return {"retrieval_backend": f"{source_name}_insufficient_parts", "vector_candidates": 0, "keyword_candidates": 0, "candidates": []}

        max_buyturn = max([int(item.get("buyturn") or 0) for role in PC_ROLE_ORDER for item in selected_rows[role]] or [1])
        interleaved: list[dict[str, Any]] = []
        for slot in range(safe_per_role):
            for role in PC_ROLE_ORDER:
                items = selected_rows[role]
                if slot >= len(items):
                    continue
                interleaved.append(
                    build_pc_candidate(items[slot], role=role, budget_vnd=safe_budget, max_buyturn=max_buyturn, source=source_name)
                )

        return {
            "retrieval_backend": source_name,
            "vector_candidates": len(interleaved),
            "keyword_candidates": 0,
            "candidates": interleaved,
        }

    source_result = _select_rows(fetch_source_rows(), "source_sql_build_pc")
    if source_result["candidates"]:
        return source_result

    kb_result = _select_rows(fetch_kb_product_rows(), "kb_build_pc_fallback")
    if kb_result["candidates"]:
        return kb_result

    return {"retrieval_backend": "build_pc_unavailable", "vector_candidates": 0, "keyword_candidates": 0, "candidates": []}
