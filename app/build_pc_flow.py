from __future__ import annotations

from time import perf_counter

from app.config import settings
from app.budget_utils import format_vnd
from app.shopping_filters import (
    build_citations,
    candidate_product_id,
    candidate_score,
    resolve_candidate_price_vnd,
)
from app.providers.generation import generate_answer
from app.reranker import rerank
from app.source_sync import get_build_pc_candidates

BUILD_PC_REQUIRED_ROLES = ("cpu", "gpu", "motherboard", "ram", "ssd", "psu", "case")


def ensure_build_pc_role_coverage(candidates: list[dict], limit: int) -> list[dict]:
    capped_limit = max(1, int(limit))
    sorted_rows = sorted(candidates, key=candidate_score, reverse=True)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    for role in BUILD_PC_REQUIRED_ROLES:
        for item in sorted_rows:
            role_name = str((item.get("metadata") or {}).get("pc_role", "")).lower()
            chunk_id = str(item.get("chunk_id") or "")
            if role_name != role or not chunk_id or chunk_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(chunk_id)
            break

    for item in sorted_rows:
        if len(selected) >= capped_limit:
            break
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(chunk_id)

    return selected[:capped_limit]


def find_budget_valid_build(
    candidates: list[dict],
    min_budget: int,
    max_budget: int,
    per_role_limit: int = 4,
) -> tuple[list[dict], int] | None:
    pools: dict[str, list[dict]] = {role: [] for role in BUILD_PC_REQUIRED_ROLES}
    for item in candidates:
        role = str((item.get("metadata") or {}).get("pc_role", "")).lower()
        if role not in pools:
            continue
        if resolve_candidate_price_vnd(item) <= 0:
            continue
        pools[role].append(item)

    if not all(pools[role] for role in BUILD_PC_REQUIRED_ROLES):
        return None

    for role in BUILD_PC_REQUIRED_ROLES:
        limit = max(1, per_role_limit)
        top_by_score = sorted(pools[role], key=candidate_score, reverse=True)[:limit]
        top_by_price = sorted(pools[role], key=resolve_candidate_price_vnd, reverse=True)[: max(1, limit // 2)]
        merged: list[dict] = []
        seen: set[int] = set()
        for item in [*top_by_score, *top_by_price]:
            pid = candidate_product_id(item)
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(item)
            if len(merged) >= limit:
                break
        pools[role] = merged

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
                                total = sum(resolve_candidate_price_vnd(item) for item in combo)
                                if total < min_budget or total > max_budget:
                                    continue
                                score = sum(candidate_score(item) for item in combo)
                                if score > best_score:
                                    best_score = score
                                    best_combo = combo
                                    best_total = total

    if best_combo is None:
        return None
    return best_combo, best_total


async def handle_build_pc_route(
    *,
    message: str,
    locale: str,
    rewrites: list[str],
    route_debug: dict,
    budget_request: dict | None,
    started: float,
) -> dict:
    budget_vnd = int((budget_request or {}).get("budget_vnd") or 0)
    if budget_vnd <= 0:
        return {
            "answer": "Bạn muốn build PC theo ngân sách bao nhiêu? Ví dụ: 20 triệu, 25 củ hoặc dưới 30 triệu.",
            "confidence": 0.35,
            "citations": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "stage_latency_ms": {"embed": 0.0, "retrieve": 0.0, "rerank": 0.0, "generate": 0.0},
                "providers": {"embedding": "intent_route", "generation": "guardrail", "retrieval": "none"},
            },
            "debug": {
                "rewritten_queries": rewrites,
                "intent_route": "build_pc_budget",
                "intent_router": route_debug,
                "selected_chunks": [],
            },
        }

    min_budget = int((budget_request or {}).get("min_budget") or round(budget_vnd * 0.95))
    max_budget = int((budget_request or {}).get("max_budget") or round(budget_vnd * 1.05))

    retrieve_started = perf_counter()
    retrieval = get_build_pc_candidates(query=message, budget_vnd=budget_vnd, per_role=8)
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
                "stage_latency_ms": {"embed": 0.0, "retrieve": retrieve_ms, "rerank": 0.0, "generate": 0.0},
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
                "intent_router": route_debug,
                "budget_vnd": budget_vnd,
                "budget_range": {"min": min_budget, "max": max_budget},
                "selected_chunks": [],
            },
        }

    rerank_started = perf_counter()
    rerank_limit = max(settings.rag_topk_rerank, 24)
    ranked_all = rerank(message, retrieval["candidates"], max(len(retrieval["candidates"]), rerank_limit))
    ranked = ensure_build_pc_role_coverage(ranked_all, rerank_limit)
    bundle_pool = ensure_build_pc_role_coverage(ranked_all, max(len(ranked_all), 56))
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    selected_roles = {str((item.get("metadata") or {}).get("pc_role", "")).lower() for item in ranked}
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
                "stage_latency_ms": {"embed": 0.0, "retrieve": retrieve_ms, "rerank": rerank_ms, "generate": 0.0},
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
                "intent_router": route_debug,
                "budget_vnd": budget_vnd,
                "budget_range": {"min": min_budget, "max": max_budget},
                "selected_chunks": [item["chunk_id"] for item in ranked],
                "selected_roles": sorted(selected_roles),
            },
        }

    bundle = find_budget_valid_build(bundle_pool, min_budget=min_budget, max_budget=max_budget, per_role_limit=8)
    if bundle is None:
        return {
            "answer": (
                "Mình chưa tìm được tổ hợp 7 linh kiện có tổng tiền nằm trong khoảng ngân sách "
                f"{format_vnd(min_budget)} - {format_vnd(max_budget)} VND từ dữ liệu hiện có."
            ),
            "confidence": 0.3,
            "citations": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "stage_latency_ms": {"embed": 0.0, "retrieve": retrieve_ms, "rerank": rerank_ms, "generate": 0.0},
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
                "intent_router": route_debug,
                "budget_vnd": budget_vnd,
                "budget_range": {"min": min_budget, "max": max_budget},
                "selected_chunks": [item["chunk_id"] for item in ranked],
                "selected_roles": sorted(selected_roles),
                "bundle_pool_size": len(bundle_pool),
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
            f"- Tong tien bat buoc trong khoang {format_vnd(min_budget)} - {format_vnd(max_budget)} VND (±5%).",
            f"- Co san mot to hop hop le voi tong tien: {format_vnd(bundle_total)} VND.",
            "- Uu tien trinh bay bo cau hinh hop le tren.",
            "- Neu khong du du lieu de dat khoang ngan sach, phai noi ro khong du du lieu.",
            "- Tra loi bang tieng Viet co dau, ro tung linh kien va tong tien.",
        ]
    )

    generation_started = perf_counter()
    generation = await generate_answer(generation_query, locale, bundle_items)
    generate_ms = round((perf_counter() - generation_started) * 1000, 2)

    return {
        "answer": generation["answer"],
        "confidence": float(generation["confidence"]),
        "citations": build_citations(bundle_items),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "stage_latency_ms": {"embed": 0.0, "retrieve": retrieve_ms, "rerank": rerank_ms, "generate": generate_ms},
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
            "intent_router": route_debug,
            "budget_vnd": budget_vnd,
            "budget_range": {"min": min_budget, "max": max_budget},
            "selected_chunks": [item["chunk_id"] for item in ranked],
            "selected_roles": sorted(selected_roles),
            "bundle_pool_size": len(bundle_pool),
            "bundle_found": True,
            "bundle_total_vnd": bundle_total,
            "bundle_chunks": [item["chunk_id"] for item in bundle_items],
        },
    }
