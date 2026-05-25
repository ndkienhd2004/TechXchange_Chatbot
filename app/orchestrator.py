from __future__ import annotations

"""RAG orchestration layer with intent-based routing."""

from time import perf_counter

from app.budget_utils import parse_budget_request, should_use_build_pc_route
from app.build_pc_flow import handle_build_pc_route
from app.intent_router import build_budget_request
from app.intent_router import classify_intent
from app.intent_router import derive_shopping_category_from_message
from app.intent_router import message_blocks_build_pc_history_fallback
from app.query_rewrite import rewrite_query
from app.shopping_flow import (
    handle_default_rag_route,
    handle_product_search_route,
    handle_top_selling_route,
)


async def run_rag_pipeline(message: str, locale: str, history: list[dict]) -> dict:
    """Execute full RAG flow for a user message."""

    started = perf_counter()
    route = await classify_intent(message, history)
    if route.intent == "build_pc" and message_blocks_build_pc_history_fallback(message):
        route = route.model_copy(
            update={
                "intent": "general",
                "category": derive_shopping_category_from_message(message) or route.category,
                "confidence": min(route.confidence, 0.65),
                "source": "portable_device_guard",
                "rationale": (route.rationale or "build_pc") + "|portable_override",
            }
        )
    route_debug = route.model_dump()

    rewrites = rewrite_query(message, history, route)

    if route.budget_vnd:
        budget_request = build_budget_request(route.budget_vnd, route.budget_mode or "target")
    else:
        budget_request = parse_budget_request(message)

    use_build_route = route.intent == "build_pc"
    if not use_build_route and not message_blocks_build_pc_history_fallback(message):
        fallback_build_route, fallback_budget_request = should_use_build_pc_route(message, history)
        if fallback_build_route:
            use_build_route = True
            if fallback_budget_request:
                budget_request = fallback_budget_request

    if use_build_route:
        return await handle_build_pc_route(
            message=message,
            locale=locale,
            rewrites=rewrites,
            route_debug=route_debug,
            budget_request=budget_request,
            started=started,
        )

    if route.intent == "product_search":
        result = await handle_product_search_route(
            message=message,
            locale=locale,
            rewrites=rewrites,
            route=route,
            route_debug=route_debug,
            started=started,
        )
        if result is not None:
            return result

    result = await handle_top_selling_route(
        message=message,
        locale=locale,
        rewrites=rewrites,
        route=route,
        route_debug=route_debug,
        started=started,
    )
    if result is not None:
        return result

    return await handle_default_rag_route(
        message=message,
        locale=locale,
        rewrites=rewrites,
        route=route,
        route_debug=route_debug,
        started=started,
    )
