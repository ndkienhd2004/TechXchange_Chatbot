from __future__ import annotations

import re

from app.budget_utils import format_vnd
from app.repository import chatbot_repository
from app.text_normalization import normalize

STRICT_SHOPPING_CATEGORIES = frozenset({"laptop", "phone", "tablet", "pc"})


def select_policy_candidates(candidates: list[dict]) -> list[dict]:
    allowed_types = {"policy", "faq"}
    return [
        item
        for item in candidates
        if str((item.get("metadata") or {}).get("doc_type", "")).lower() in allowed_types
    ]


def candidate_score(item: dict) -> float:
    return float(item.get("rerank_score", item.get("retrieval_score", 0.0)) or 0.0)


def candidate_product_id(item: dict) -> int:
    metadata = dict(item.get("metadata") or {})
    try:
        product_id = int(metadata.get("product_id") or 0)
    except (TypeError, ValueError):
        product_id = 0
    if product_id > 0:
        return product_id
    match = re.search(r"/products/(\d+)", str(item.get("uri") or ""))
    return int(match.group(1)) if match else 0


def dedupe_candidates_by_product(candidates: list[dict], limit: int | None = None) -> list[dict]:
    if not candidates:
        return []

    deduped: list[dict] = []
    seen_products: set[int] = set()
    seen_chunks: set[str] = set()
    for item in sorted(candidates, key=candidate_score, reverse=True):
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_chunks:
            continue
        product_id = candidate_product_id(item)
        if product_id > 0:
            if product_id in seen_products:
                continue
            seen_products.add(product_id)
        if chunk_id:
            seen_chunks.add(chunk_id)
        deduped.append(item)
        if limit and len(deduped) >= max(1, int(limit)):
            break
    return deduped


def category_signals_device_accessory(normalized_cat: str, device: str) -> bool:
    common = (
        "phu kien",
        "accessories",
        "accessory",
        "usb",
        "hub",
        "hub usb",
        "dock",
        "docking",
        "cap sac",
        "cu sac",
        "sac du phong",
        "sac nhanh",
        "day cap",
        "cap ket noi",
        "type c hub",
        "multiport",
        "chuot",
        "mouse",
        "ban phim",
        "keyboard",
        "tai nghe",
        "headphone",
        "loa bluetooth",
        "speaker",
        "o cung ngoai",
        "external",
        "hdd box",
        "card mang",
        "network card",
        "router wifi",
        "switch mang",
        "gia do",
        "gia treo",
        "stand",
        "nem ban phim",
        "tan nhiet",
        "keo tan nhiet",
        "chuyen doi",
        "bo chuyen",
        "adapter",
    )
    if device == "laptop":
        extra = ("op lung", "bao da", "op macbook", "mieng dan", "kinh cuong luc", "balo", "tui chong soc")
    elif device == "phone":
        extra = ("op lung", "bao da", "mieng dan", "kinh cuong luc", "mieng dan cuong luc")
    elif device == "tablet":
        extra = ("op lung", "bao da", "but cam ung", "stylus", "mieng dan")
    else:
        extra = ()
    return any(marker in normalized_cat for marker in common + extra)


def metadata_matches_shopping_category(
    category_text: str,
    wanted: str,
    category_slug: str | None = None,
) -> bool:
    slug = normalize(str(category_slug or "").replace("-", " "))
    cat = normalize(str(category_text or ""))
    merged = f"{cat} {slug}".strip()
    if not merged:
        return False
    wanted_norm = normalize(str(wanted or ""))
    if not wanted_norm:
        return False

    if wanted_norm == "laptop":
        if category_signals_device_accessory(merged, "laptop"):
            return False
        return any(marker in merged for marker in ("laptop", "notebook", "macbook", "xach tay", "mtxt", "may tinh xach tay"))

    if wanted_norm == "phone":
        if category_signals_device_accessory(merged, "phone"):
            return False
        return any(marker in merged for marker in ("dien thoai", "smartphone", "iphone", "android", "mobile"))

    if wanted_norm == "tablet":
        if category_signals_device_accessory(merged, "tablet"):
            return False
        return any(marker in merged for marker in ("tablet", "ipad"))

    if wanted_norm == "pc":
        if any(marker in merged for marker in ("laptop", "notebook", "macbook", "xach tay", "mtxt")):
            return False
        if category_signals_device_accessory(merged, "pc"):
            return False
        return any(
            marker in merged
            for marker in ("pc", "may tinh de ban", "desktop", "bo may tinh", "computer", "linh kien", "cpu", "vga", "mainboard")
        )

    return wanted_norm in merged


def resolve_candidate_price_vnd(item: dict) -> int:
    metadata = dict(item.get("metadata") or {})
    try:
        value = int(float(metadata.get("price_vnd") or 0))
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else 0


def filter_product_candidates(
    candidates: list[dict],
    category: str | None,
    budget_vnd: int | None,
    budget_mode: str | None,
) -> list[dict]:
    if not candidates:
        return []

    normalized_category = normalize(str(category or ""))
    if not normalized_category and not (budget_vnd and budget_vnd > 0):
        return candidates

    def _matches_category(item: dict) -> bool:
        if not normalized_category:
            return True
        metadata = dict(item.get("metadata") or {})
        raw_category = str(metadata.get("category") or "")
        if normalized_category in STRICT_SHOPPING_CATEGORIES:
            return metadata_matches_shopping_category(
                raw_category,
                normalized_category,
                str(metadata.get("category_slug") or "") or None,
            )
        category_text = normalize(raw_category)
        if category_text and normalized_category in category_text:
            return True
        blob = normalize(" ".join([str(item.get("title") or ""), str(item.get("content") or "")]))
        return normalized_category in blob

    def _matches_budget(item: dict) -> bool:
        if not budget_vnd or budget_vnd <= 0:
            return True
        price_vnd = resolve_candidate_price_vnd(item)
        if price_vnd <= 0:
            return True
        mode = str(budget_mode or "").strip().lower() or "target"
        if mode == "lower":
            return price_vnd >= budget_vnd
        return price_vnd <= budget_vnd

    filtered: list[dict] = []
    for item in candidates:
        metadata = dict(item.get("metadata") or {})
        if str(metadata.get("doc_type") or "") != "product":
            continue
        if not _matches_category(item):
            continue
        if not _matches_budget(item):
            continue
        filtered.append(item)

    return filtered


def keyword_hit_ratio(query: str, content: str) -> float:
    terms = [t for t in normalize(query).split() if t]
    if not terms:
        return 0.0
    hay = normalize(content)
    hits = sum(1 for term in set(terms) if term in hay)
    return hits / len(set(terms))


def category_budget_fallback_candidates(
    query: str,
    category: str,
    budget_vnd: int,
    budget_mode: str | None,
    limit: int = 60,
) -> list[dict]:
    normalized_category = normalize(category)
    if not normalized_category:
        return []
    budget = max(int(budget_vnd), 0)
    mode = str(budget_mode or "").strip().lower() or "upper"

    rows = []
    for chunk in chatbot_repository.get_chunks(include_embeddings=False):
        metadata = dict(chunk.get("metadata") or {})
        if str(metadata.get("doc_type") or "") != "product":
            continue
        if not metadata_matches_shopping_category(
            str(metadata.get("category") or ""),
            category,
            str(metadata.get("category_slug") or "") or None,
        ):
            continue
        price = resolve_candidate_price_vnd(chunk)
        if budget > 0 and price > 0:
            if mode == "lower" and price < budget:
                continue
            if mode != "lower" and price > budget:
                continue
        try:
            buyturn = int(metadata.get("buyturn") or 0)
        except (TypeError, ValueError):
            buyturn = 0
        keyword = keyword_hit_ratio(query, str(chunk.get("title") or "") + " " + str(chunk.get("content") or ""))
        score = keyword * 0.7 + min(buyturn / 500.0, 1.0) * 0.3
        rows.append({**chunk, "semantic_score": 0.0, "keyword_score": keyword, "retrieval_score": score})

    rows.sort(key=lambda item: float(item.get("retrieval_score") or 0.0), reverse=True)
    return rows[: max(1, int(limit))]


def looks_insufficient_answer(text: str) -> bool:
    normalized = normalize(str(text or ""))
    if not normalized:
        return True
    markers = [
        "khong du du lieu",
        "khong du thong tin",
        "khong tim thay",
        "context khong du",
        "insufficient",
    ]
    return any(marker in normalized for marker in markers)


def build_shopping_generation_query(message: str) -> str:
    return "\n".join(
        [
            message,
            "",
            "SHOPPING_RULES:",
            "- Day la danh sach ung vien tu kho tri thuc (contexts). Neu contexts khong rong, bat buoc de xuat it nhat 3 lua chon tu contexts neu co du san pham phan biet.",
            "- Moi san pham chi duoc xuat hien toi da 1 lan. Khong lap lai cung title hoac uri.",
            "- Neu co gia trong contexts, hay ghi gia (VND) va ly do ngan gon cho tung lua chon.",
            "- Khong duoc yeu cau nguoi dung cung cap gia neu gia da co trong contexts.",
            "- Neu nguoi dung hoi laptop/dien thoai/may tinh bang, khong duoc de xuat phu kien neu do khong phai thiet bi do.",
            "- Neu that su contexts khong co san pham phu hop, hay noi ro 'khong du du lieu trong kho tri thuc' va goi y tu khoa can them.",
            "- Tra loi tieng Viet co dau.",
        ]
    )


def build_shopping_retry_query(message: str, ranked: list[dict]) -> str:
    candidates = []
    for item in ranked[:8]:
        title = str(item.get("title") or "Untitled")
        uri = str(item.get("uri") or "")
        price = resolve_candidate_price_vnd(item)
        price_text = format_vnd(price) if price > 0 else "N/A"
        candidates.append(f"- {title} | gia={price_text} | uri={uri}")
    return "\n".join(
        [
            message,
            "",
            "RETRY_RULES:",
            "- Contexts da co ung vien. Khong duoc tu choi vi thieu boi canh neu da co san pham.",
            "- Moi san pham chi duoc xuat hien toi da 1 lan.",
            "- Bat buoc chon it nhat 1 san pham tu danh sach ung vien ben duoi neu danh sach khong rong.",
            "- Neu co tu 2 hoac 3 ung vien, hay tra ra dung so ung vien dang co.",
            "- Neu gia N/A, van co the de xuat va ghi ro 'chua co gia trong du lieu'.",
            "- Tra loi tieng Viet co dau.",
            "",
            "UNG_VIEN:",
            *candidates,
        ]
    )


def build_citations(rows: list[dict]) -> list[dict]:
    return [
        {
            "index": index,
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "uri": item["uri"],
            "score": float(item["rerank_score"]),
        }
        for index, item in enumerate(rows, start=1)
    ]
