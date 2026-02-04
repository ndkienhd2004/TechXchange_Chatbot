from __future__ import annotations

"""RAG orchestration layer with intent-based routing."""

import re
import unicodedata
from time import perf_counter

from app.config import settings
from app.providers.embedding import embed_text
from app.providers.generation import generate_answer
from app.query_rewrite import rewrite_query
from app.reranker import rerank
from app.retriever import retrieve
from app.source_sync import get_build_pc_candidates
from app.source_sync import get_top_selling_candidates

BUILD_PC_REQUIRED_ROLES = ("cpu", "gpu", "motherboard", "ram", "ssd", "psu", "case")
BUILD_PC_ROLE_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
    "motherboard": "Mainboard",
    "ram": "RAM",
    "ssd": "SSD",
    "psu": "PSU",
    "case": "Case",
}


def _strip_accents(text: str) -> str:
    """Remove accents to support matching with/without Vietnamese diacritics."""

    decomposed = unicodedata.normalize("NFD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    """Normalize text to lower alphanumeric tokens with single spaces."""

    lowered = _strip_accents(text).lower()
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered).split())


def _is_top_selling_intent(message: str) -> bool:
    """Detect user intent asking for best-selling products."""

    normalized = _normalize(message)
    if not normalized:
        return False
    intent_phrases = [
        "ban chay",
        "best seller",
        "bestseller",
        "top selling",
        "nhieu nguoi mua",
        "mua nhieu",
        "ban nhieu",
    ]
    return any(phrase in normalized for phrase in intent_phrases)


def _is_build_pc_intent(message: str) -> bool:
    """Detect user intent asking to build a PC configuration."""

    normalized = _normalize(message)
    if not normalized:
        return False
    return (
        ("build" in normalized and "pc" in normalized)
        or "cau hinh pc" in normalized
        or "lap pc" in normalized
    )


def _wants_multiple_builds(message: str) -> bool:
    """Detect whether user asks for multiple build options."""

    normalized = _normalize(message)
    if not normalized:
        return False
    phrases = [
        "mot vai bo",
        "vai bo",
        "vai cau hinh",
        "nhieu bo",
        "nhieu cau hinh",
        "vai option",
        "several",
        "multiple",
    ]
    return any(phrase in normalized for phrase in phrases)


def _is_policy_intent(message: str) -> bool:
    """Detect policy/FAQ intent to prioritize policy knowledge documents."""

    normalized = _normalize(message)
    if not normalized:
        return False
    intent_phrases = [
        "chinh sach",
        "bao hanh",
        "doi tra",
        "hoan tien",
        "thanh toan",
        "huy don",
        "giao hang",
        "van chuyen",
        "phuong thuc",
    ]
    return any(phrase in normalized for phrase in intent_phrases)


def _select_policy_candidates(candidates: list[dict]) -> list[dict]:
    """Keep only policy/faq candidates when policy intent is detected."""

    allowed_types = {"policy", "faq"}
    return [
        item
        for item in candidates
        if str((item.get("metadata") or {}).get("doc_type", "")).lower() in allowed_types
    ]


def _parse_budget_vnd(message: str) -> int | None:
    """Extract budget in VND from common formats (`tr`, `m`, plain digits)."""

    raw = _strip_accents(str(message or "")).lower().strip()
    if not raw:
        return None

    million_match = re.search(r"(?:^|[^\d])(\d+(?:[.,]\d+)?)\s*(trieu|tr|m)\b", raw)
    if million_match:
        value = float(million_match.group(1).replace(",", "."))
        return int(round(value * 1_000_000))

    billion_match = re.search(r"(?:^|[^\d])(\d+(?:[.,]\d+)?)\s*(ty|ti)\b", raw)
    if billion_match:
        value = float(billion_match.group(1).replace(",", "."))
        return int(round(value * 1_000_000_000))

    plain_match = re.search(r"(\d[\d.,]{5,})", raw)
    if plain_match:
        text = plain_match.group(1).replace(",", "").replace(".", "")
        if text.isdigit():
            parsed = int(text)
            if parsed >= 1_000_000:
                return parsed
    return None


def _parse_budget_request(message: str) -> dict | None:
    """Parse budget amount and derive budget mode/range from user phrasing."""

    budget_vnd = _parse_budget_vnd(message)
    if not budget_vnd:
        return None

    normalized = _normalize(message)
    upper_markers = [
        "duoi",
        "toi da",
        "khong qua",
        "under",
        "max",
        "den",
    ]
    lower_markers = [
        "tren",
        "toi thieu",
        "at least",
        "min",
    ]

    mode = "target"
    if any(marker in normalized for marker in upper_markers):
        mode = "upper"
    elif any(marker in normalized for marker in lower_markers):
        mode = "lower"

    if mode == "upper":
        min_budget = 0
        max_budget = budget_vnd
    elif mode == "lower":
        min_budget = budget_vnd
        max_budget = int(round(budget_vnd * 1.3))
    else:
        min_budget = int(round(budget_vnd * 0.95))
        max_budget = int(round(budget_vnd * 1.05))

    return {
        "budget_vnd": budget_vnd,
        "mode": mode,
        "min_budget": min_budget,
        "max_budget": max_budget,
    }


def _has_recent_build_pc_context(message: str, history: list[dict]) -> bool:
    """Check if recent user turns indicate the current turn is build-PC follow-up."""

    current = _normalize(message)
    checked = 0
    skipped_current = False

    for item in reversed(history):
        if item.get("role") != "user":
            continue
        text = _normalize(str(item.get("content") or ""))
        if not text:
            continue
        if not skipped_current and text == current:
            skipped_current = True
            continue
        checked += 1
        if (
            _is_build_pc_intent(text)
            or "linh kien" in text
            or "cpu" in text
            or "gpu" in text
        ):
            return True
        if checked >= 4:
            break

    return False


def _should_use_build_pc_route(message: str, history: list[dict]) -> tuple[bool, dict | None]:
    """Resolve whether the request should be handled by build-PC route."""

    budget_request = _parse_budget_request(message)
    if _is_build_pc_intent(message):
        return True, budget_request
    if budget_request and _has_recent_build_pc_context(message, history):
        return True, budget_request
    return False, budget_request


def _format_vnd(value: int) -> str:
    """Format integer VND with dot thousands separator."""

    return f"{int(value):,}".replace(",", ".")


def _candidate_score(item: dict) -> float:
    """Resolve stable candidate score for sorting mixed retrieval/rerank rows."""

    return float(item.get("rerank_score", item.get("retrieval_score", 0.0)) or 0.0)


def _ensure_build_pc_role_coverage(candidates: list[dict], limit: int) -> list[dict]:
    """Select contexts while ensuring at least one candidate for each PC role."""

    capped_limit = max(1, int(limit))
    sorted_rows = sorted(candidates, key=_candidate_score, reverse=True)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    # First pass: guarantee one row per required role whenever available.
    for role in BUILD_PC_REQUIRED_ROLES:
        for item in sorted_rows:
            role_name = str((item.get("metadata") or {}).get("pc_role", "")).lower()
            chunk_id = str(item.get("chunk_id") or "")
            if role_name != role or not chunk_id or chunk_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(chunk_id)
            break

    # Second pass: fill remaining slots by score.
    for item in sorted_rows:
        if len(selected) >= capped_limit:
            break
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(chunk_id)

    return selected[:capped_limit]


def _resolve_candidate_price_vnd(item: dict) -> int:
    """Extract candidate price in VND from metadata/content with safe fallback."""

    metadata = dict(item.get("metadata") or {})
    try:
        value = int(float(metadata.get("price_vnd") or 0))
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else 0


def _find_budget_valid_build(
    candidates: list[dict],
    min_budget: int,
    max_budget: int,
    per_role_limit: int = 4,
) -> tuple[list[dict], int] | None:
    """Find one 7-part build whose total is within budget range and best score."""

    pools: dict[str, list[dict]] = {role: [] for role in BUILD_PC_REQUIRED_ROLES}
    for item in candidates:
        role = str((item.get("metadata") or {}).get("pc_role", "")).lower()
        if role not in pools:
            continue
        if _resolve_candidate_price_vnd(item) <= 0:
            continue
        pools[role].append(item)

    if not all(pools[role] for role in BUILD_PC_REQUIRED_ROLES):
        return None

    for role in BUILD_PC_REQUIRED_ROLES:
        pools[role] = sorted(pools[role], key=_candidate_score, reverse=True)[: max(1, per_role_limit)]

    best_combo: list[dict] | None = None
    best_total = 0
    best_score = -1.0

    cpu_rows = pools["cpu"]
    gpu_rows = pools["gpu"]
    motherboard_rows = pools["motherboard"]
    ram_rows = pools["ram"]
    ssd_rows = pools["ssd"]
    psu_rows = pools["psu"]
    case_rows = pools["case"]

    for cpu in cpu_rows:
        for gpu in gpu_rows:
            for motherboard in motherboard_rows:
                for ram in ram_rows:
                    for ssd in ssd_rows:
                        for psu in psu_rows:
                            for case in case_rows:
                                combo = [cpu, gpu, motherboard, ram, ssd, psu, case]
                                total = sum(_resolve_candidate_price_vnd(item) for item in combo)
                                if total < min_budget or total > max_budget:
                                    continue
                                score = sum(_candidate_score(item) for item in combo)
                                if score > best_score:
                                    best_score = score
                                    best_combo = combo
                                    best_total = total

    if best_combo is None:
        return None
    return best_combo, best_total


async def run_rag_pipeline(message: str, locale: str, history: list[dict]) -> dict:
    """Execute full RAG flow for a user message."""

    started = perf_counter()

    # Stage 1: query rewrite to improve recall.
    rewrites = rewrite_query(message, history)

    # Stage 2A: intent route for build-pc with strict budget constraints.
    if _is_build_pc_intent(message):
        budget_vnd = _parse_budget_vnd(message)
        if budget_vnd:
            min_budget = int(round(budget_vnd * 0.95))
            max_budget = int(round(budget_vnd * 1.05))
            retrieve_started = perf_counter()
            retrieval = get_build_pc_candidates(
                query=message,
                budget_vnd=budget_vnd,
                per_role=4,
            )
            retrieve_ms = round((perf_counter() - retrieve_started) * 1000, 2)
            if not retrieval["candidates"]:
                return {
                    "answer": (
                        "Mình chưa đủ dữ liệu linh kiện thực tế để lên full bộ PC theo ngân sách này. "
                        "Hiện kho dữ liệu cần có đủ 7 nhóm: CPU, GPU, Mainboard, RAM, SSD, PSU, Case."
                    ),
                    "confidence": 0.2,
                    "citations": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "stage_latency_ms": {
                            "embed": 0.0,
                            "retrieve": retrieve_ms,
                            "rerank": 0.0,
                            "generate": 0.0,
                        },
                        "providers": {
                            "embedding": "intent_route",
                            "generation": "guardrail",
                            "retrieval": retrieval["retrieval_backend"],
                        },
                    },
                    "debug": {
                        "rewritten_queries": rewrites,
                        "vector_candidates": retrieval["vector_candidates"],
                        "keyword_candidates": retrieval["keyword_candidates"],
                        "retrieval_backend": retrieval["retrieval_backend"],
                        "intent_route": "build_pc_budget",
                        "budget_vnd": budget_vnd,
                        "budget_range": {
                            "min": min_budget,
                            "max": max_budget,
                        },
                        "selected_chunks": [],
                    },
                }
            rerank_started = perf_counter()
            rerank_limit = max(settings.rag_topk_rerank, 24)
            ranked_all = rerank(
                message,
                retrieval["candidates"],
                max(len(retrieval["candidates"]), rerank_limit),
            )
            ranked = _ensure_build_pc_role_coverage(ranked_all, rerank_limit)
            rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

            selected_roles = {
                str((item.get("metadata") or {}).get("pc_role", "")).lower()
                for item in ranked
            }
            if not all(role in selected_roles for role in BUILD_PC_REQUIRED_ROLES):
                return {
                    "answer": (
                        "Mình chưa đủ dữ liệu cân bằng giữa 7 nhóm linh kiện để đề xuất cấu hình ổn định "
                        "theo ngân sách này. Bạn thử tăng/giảm ngân sách hoặc nêu rõ ưu tiên CPU/GPU."
                    ),
                    "confidence": 0.25,
                    "citations": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "stage_latency_ms": {
                            "embed": 0.0,
                            "retrieve": retrieve_ms,
                            "rerank": rerank_ms,
                            "generate": 0.0,
                        },
                        "providers": {
                            "embedding": "intent_route",
                            "generation": "guardrail",
                            "retrieval": retrieval["retrieval_backend"],
                        },
                    },
                    "debug": {
                        "rewritten_queries": rewrites,
                        "vector_candidates": retrieval["vector_candidates"],
                        "keyword_candidates": retrieval["keyword_candidates"],
                        "retrieval_backend": retrieval["retrieval_backend"],
                        "intent_route": "build_pc_budget",
                        "budget_vnd": budget_vnd,
                        "budget_range": {
                            "min": min_budget,
                            "max": max_budget,
                        },
                        "selected_chunks": [item["chunk_id"] for item in ranked],
                        "selected_roles": sorted(selected_roles),
                    },
                }

            bundle = _find_budget_valid_build(
                ranked,
                min_budget=min_budget,
                max_budget=max_budget,
                per_role_limit=4,
            )
            if bundle is None:
                return {
                    "answer": (
                        "Mình chưa tìm được tổ hợp 7 linh kiện có tổng tiền nằm trong khoảng ngân sách "
                        f"{_format_vnd(min_budget)} - {_format_vnd(max_budget)} VND từ dữ liệu hiện có."
                    ),
                    "confidence": 0.3,
                    "citations": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "stage_latency_ms": {
                            "embed": 0.0,
                            "retrieve": retrieve_ms,
                            "rerank": rerank_ms,
                            "generate": 0.0,
                        },
                        "providers": {
                            "embedding": "intent_route",
                            "generation": "guardrail",
                            "retrieval": retrieval["retrieval_backend"],
                        },
                    },
                    "debug": {
                        "rewritten_queries": rewrites,
                        "vector_candidates": retrieval["vector_candidates"],
                        "keyword_candidates": retrieval["keyword_candidates"],
                        "retrieval_backend": retrieval["retrieval_backend"],
                        "intent_route": "build_pc_budget",
                        "budget_vnd": budget_vnd,
                        "budget_range": {
                            "min": min_budget,
                            "max": max_budget,
                        },
                        "selected_chunks": [item["chunk_id"] for item in ranked],
                        "selected_roles": sorted(selected_roles),
                        "bundle_found": False,
                    },
                }

            bundle_items, bundle_total = bundle

            generation_query = "\n".join(
                [
                    message,
                    "",
                    "BUILD_PC_RULES:",
                    "- Chi duoc chon linh kien trong contexts da cung cap.",
                    "- Bat buoc co du 7 nhom: CPU, GPU, Mainboard, RAM, SSD, PSU, Case.",
                    f"- Tong tien bat buoc trong khoang { _format_vnd(min_budget) } - { _format_vnd(max_budget) } VND (±5%).",
                    f"- Co san mot to hop hop le voi tong tien: {_format_vnd(bundle_total)} VND.",
                    "- Uu tien trinh bay bo cau hinh hop le tren.",
                    "- Neu khong du du lieu de dat khoang ngan sach, phai noi ro khong du du lieu.",
                    "- Tra loi bang tieng Viet co dau, ro tung linh kien va tong tien.",
                ]
            )

            generation_started = perf_counter()
            generation = await generate_answer(generation_query, locale, bundle_items)
            generate_ms = round((perf_counter() - generation_started) * 1000, 2)

            citations = [
                {
                    "index": index,
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "uri": item["uri"],
                    "score": float(item["rerank_score"]),
                }
                for index, item in enumerate(bundle_items, start=1)
            ]

            return {
                "answer": generation["answer"],
                "confidence": float(generation["confidence"]),
                "citations": citations,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "stage_latency_ms": {
                        "embed": 0.0,
                        "retrieve": retrieve_ms,
                        "rerank": rerank_ms,
                        "generate": generate_ms,
                    },
                    "providers": {
                        "embedding": "intent_route",
                        "generation": generation["provider"],
                        "retrieval": retrieval["retrieval_backend"],
                    },
                },
                "debug": {
                    "rewritten_queries": rewrites,
                    "vector_candidates": retrieval["vector_candidates"],
                    "keyword_candidates": retrieval["keyword_candidates"],
                    "retrieval_backend": retrieval["retrieval_backend"],
                    "intent_route": "build_pc_budget",
                    "budget_vnd": budget_vnd,
                    "budget_range": {
                        "min": min_budget,
                        "max": max_budget,
                    },
                    "selected_chunks": [item["chunk_id"] for item in ranked],
                    "selected_roles": sorted(selected_roles),
                    "bundle_found": True,
                    "bundle_total_vnd": bundle_total,
                    "bundle_chunks": [item["chunk_id"] for item in bundle_items],
                },
            }

    # Stage 2B: intent route for top-selling list requests.
    if _is_top_selling_intent(message):
        retrieve_started = perf_counter()
        retrieval = get_top_selling_candidates(
            query=message,
            limit=max(settings.rag_topk_rerank, 8),
        )
        retrieve_ms = round((perf_counter() - retrieve_started) * 1000, 2)
        if retrieval["candidates"]:
            rerank_started = perf_counter()
            ranked = rerank(message, retrieval["candidates"], settings.rag_topk_rerank)
            rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

            generation_started = perf_counter()
            generation = await generate_answer(message, locale, ranked)
            generate_ms = round((perf_counter() - generation_started) * 1000, 2)

            citations = [
                {
                    "index": index,
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "uri": item["uri"],
                    "score": float(item["rerank_score"]),
                }
                for index, item in enumerate(ranked, start=1)
            ]

            return {
                "answer": generation["answer"],
                "confidence": float(generation["confidence"]),
                "citations": citations,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "stage_latency_ms": {
                        "embed": 0.0,
                        "retrieve": retrieve_ms,
                        "rerank": rerank_ms,
                        "generate": generate_ms,
                    },
                    "providers": {
                        "embedding": "intent_route",
                        "generation": generation["provider"],
                        "retrieval": retrieval["retrieval_backend"],
                    },
                },
                "debug": {
                    "rewritten_queries": rewrites,
                    "vector_candidates": retrieval["vector_candidates"],
                    "keyword_candidates": retrieval["keyword_candidates"],
                    "retrieval_backend": retrieval["retrieval_backend"],
                    "intent_route": "top_selling",
                    "selected_chunks": [item["chunk_id"] for item in ranked],
                },
            }

    # Stage 3: default RAG flow (embed -> retrieve -> rerank -> generate).
    embed_started = perf_counter()
    embedding = await embed_text(message, task_type="RETRIEVAL_QUERY")
    embed_ms = round((perf_counter() - embed_started) * 1000, 2)

    retrieve_started = perf_counter()
    retrieval = retrieve(
        query=message,
        rewritten_queries=rewrites,
        query_embedding=embedding["vector"],
        topk_vector=settings.rag_topk_vector,
        topk_keyword=settings.rag_topk_keyword,
    )
    retrieve_ms = round((perf_counter() - retrieve_started) * 1000, 2)

    candidates_for_rerank = retrieval["candidates"]
    policy_priority_applied = False
    if _is_policy_intent(message):
        policy_candidates = _select_policy_candidates(retrieval["candidates"])
        if policy_candidates:
            candidates_for_rerank = policy_candidates
            policy_priority_applied = True

    rerank_started = perf_counter()
    ranked = rerank(message, candidates_for_rerank, settings.rag_topk_rerank)
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    generation_started = perf_counter()
    generation = await generate_answer(message, locale, ranked)
    generate_ms = round((perf_counter() - generation_started) * 1000, 2)

    citations = [
        {
            "index": index,
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "uri": item["uri"],
            "score": float(item["rerank_score"]),
        }
        for index, item in enumerate(ranked, start=1)
    ]

    return {
        "answer": generation["answer"],
        "confidence": float(generation["confidence"]),
        "citations": citations,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "stage_latency_ms": {
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "rerank": rerank_ms,
                "generate": generate_ms,
            },
            "providers": {
                "embedding": embedding["provider"],
                "generation": generation["provider"],
                "retrieval": retrieval["retrieval_backend"],
            },
        },
        "debug": {
            "rewritten_queries": rewrites,
            "vector_candidates": retrieval["vector_candidates"],
            "keyword_candidates": retrieval["keyword_candidates"],
            "retrieval_backend": retrieval["retrieval_backend"],
            "intent_route": "policy_priority" if policy_priority_applied else "default",
            "selected_chunks": [item["chunk_id"] for item in ranked],
        },
    }
