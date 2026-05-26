from __future__ import annotations

from time import perf_counter

from app.config import settings
from app.shopping_filters import (
    build_citations,
    build_shopping_generation_query,
    build_shopping_retry_query,
    candidate_product_id,
    category_budget_fallback_candidates,
    dedupe_candidates_by_product,
    filter_product_candidates,
    looks_insufficient_answer,
    select_policy_candidates,
)
from app.providers.embedding import embed_text
from app.providers.generation import generate_answer
from app.product_search_candidates import get_product_search_candidates
from app.reranker import rerank
from app.retriever import retrieve
from app.text_normalization import is_policy_intent, is_top_selling_intent
from app.top_selling_candidates import get_top_selling_candidates


async def handle_product_search_route(
    *,
    message: str,
    locale: str,
    rewrites: list[str],
    route,
    route_debug: dict,
    started: float,
) -> dict | None:
    retrieve_started = perf_counter()
    retrieval = get_product_search_candidates(
        query=message,
        category=str(route.category or ""),
        budget_vnd=route.budget_vnd,
        budget_mode=route.budget_mode,
        limit=max(settings.rag_topk_rerank * 2, 12),
    )
    retrieve_ms = round((perf_counter() - retrieve_started) * 1000, 2)
    distinct_candidates = dedupe_candidates_by_product(
        retrieval["candidates"],
        limit=max(settings.rag_topk_rerank * 2, 12),
    )
    if not distinct_candidates:
        return None

    rerank_started = perf_counter()
    ranked = rerank(message, distinct_candidates, max(settings.rag_topk_rerank * 2, 12))
    ranked = dedupe_candidates_by_product(ranked, limit=max(settings.rag_topk_rerank, 6))
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    generation_started = perf_counter()
    generation = await generate_answer(build_shopping_generation_query(message), locale, ranked)
    if ranked and looks_insufficient_answer(str(generation.get("answer") or "")):
        generation = await generate_answer(build_shopping_retry_query(message, ranked), locale, ranked)
    generate_ms = round((perf_counter() - generation_started) * 1000, 2)

    return {
        "answer": generation["answer"],
        "confidence": float(generation["confidence"]),
        "citations": build_citations(ranked),
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
            "intent_route": "product_search",
            "intent_router": route_debug,
            "selected_chunks": [item["chunk_id"] for item in ranked],
            "distinct_products": [candidate_product_id(item) for item in ranked],
        },
    }


async def handle_top_selling_route(
    *,
    message: str,
    locale: str,
    rewrites: list[str],
    route,
    route_debug: dict,
    started: float,
) -> dict | None:
    if route.intent != "top_selling" and not is_top_selling_intent(message):
        return None

    retrieve_started = perf_counter()
    retrieval = get_top_selling_candidates(query=message, limit=max(settings.rag_topk_rerank, 8))
    retrieve_ms = round((perf_counter() - retrieve_started) * 1000, 2)
    distinct_candidates = dedupe_candidates_by_product(
        retrieval["candidates"],
        limit=max(settings.rag_topk_rerank * 2, 10),
    )
    if not distinct_candidates:
        return None

    rerank_started = perf_counter()
    ranked = rerank(message, distinct_candidates, max(settings.rag_topk_rerank * 2, 10))
    ranked = dedupe_candidates_by_product(ranked, limit=max(settings.rag_topk_rerank, 5))
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    generation_started = perf_counter()
    generation = await generate_answer(build_shopping_generation_query(message), locale, ranked)
    if ranked and looks_insufficient_answer(str(generation.get("answer") or "")):
        generation = await generate_answer(build_shopping_retry_query(message, ranked), locale, ranked)
    generate_ms = round((perf_counter() - generation_started) * 1000, 2)

    return {
        "answer": generation["answer"],
        "confidence": float(generation["confidence"]),
        "citations": build_citations(ranked),
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
            "intent_route": "top_selling",
            "intent_router": route_debug,
            "selected_chunks": [item["chunk_id"] for item in ranked],
            "distinct_products": [candidate_product_id(item) for item in ranked],
        },
    }


async def handle_default_rag_route(
    *,
    message: str,
    locale: str,
    rewrites: list[str],
    route,
    route_debug: dict,
    started: float,
) -> dict:
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
    if route.intent == "policy" or is_policy_intent(message):
        policy_candidates = select_policy_candidates(retrieval["candidates"])
        if policy_candidates:
            candidates_for_rerank = policy_candidates
            policy_priority_applied = True
    elif route.category and route.budget_vnd:
        filtered = filter_product_candidates(
            retrieval["candidates"],
            category=route.category,
            budget_vnd=route.budget_vnd,
            budget_mode=route.budget_mode,
        )
        if filtered:
            candidates_for_rerank = filtered
        else:
            candidates_for_rerank = category_budget_fallback_candidates(
                message,
                category=str(route.category),
                budget_vnd=int(route.budget_vnd),
                budget_mode=route.budget_mode,
                limit=max(settings.rag_topk_vector, settings.rag_topk_keyword, 60),
            )
        candidates_for_rerank = dedupe_candidates_by_product(
            candidates_for_rerank,
            limit=max(settings.rag_topk_rerank * 2, 12),
        )

    rerank_started = perf_counter()
    ranked = rerank(message, candidates_for_rerank, settings.rag_topk_rerank)
    if route.intent == "general" and route.category:
        ranked = dedupe_candidates_by_product(ranked, limit=settings.rag_topk_rerank)
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    generation_started = perf_counter()
    generation_query = message
    if route.intent == "general" and route.category and route.budget_vnd:
        generation_query = build_shopping_generation_query(message)
    generation = await generate_answer(generation_query, locale, ranked)
    if route.intent == "general" and route.category and route.budget_vnd and ranked and looks_insufficient_answer(
        str(generation.get("answer") or "")
    ):
        generation = await generate_answer(build_shopping_retry_query(message, ranked), locale, ranked)
    generate_ms = round((perf_counter() - generation_started) * 1000, 2)

    return {
        "answer": generation["answer"],
        "confidence": float(generation["confidence"]),
        "citations": build_citations(ranked),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "stage_latency_ms": {"embed": embed_ms, "retrieve": retrieve_ms, "rerank": rerank_ms, "generate": generate_ms},
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
            "intent_router": route_debug,
            "selected_chunks": [item["chunk_id"] for item in ranked],
            "distinct_products": [
                product_id
                for product_id in (candidate_product_id(item) for item in ranked)
                if product_id > 0
            ],
        },
    }
