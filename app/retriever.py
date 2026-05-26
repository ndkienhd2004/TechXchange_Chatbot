from __future__ import annotations

"""Hybrid retriever: vector similarity + SQL keyword matching."""

from typing import Any

from app.repository import chatbot_repository
from app.text_normalization import normalize_preserve_accents
from app.text_normalization import tokenize_text_preserve_accents


def _tokenize_text(text: str) -> list[str]:
    """Backward-compatible wrapper around shared tokenizer."""

    return tokenize_text_preserve_accents(text)


def _compute_keyword_overlap(query: str, content: str) -> float:
    """Return simple keyword-hit ratio between query and content."""

    query_terms = set(_tokenize_text(query))
    if not query_terms:
        return 0.0
    haystack = normalize_preserve_accents(content)
    hit_count = sum(1 for term in query_terms if term in haystack)
    return hit_count / len(query_terms)


def _compute_cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve(
    query: str,
    rewritten_queries: list[str],
    query_embedding: list[float],
    topk_vector: int,
    topk_keyword: int,
) -> dict[str, Any]:
    """Retrieve candidate chunks using vector + keyword blend scoring."""

    capabilities = chatbot_repository.get_storage_capabilities()
    all_queries = [query, *rewritten_queries]
    candidates: dict[str, dict[str, Any]] = {}

    if capabilities["vector_search_enabled"]:
        vector_rows = chatbot_repository.vector_search(query_embedding, topk_vector)
        retrieval_backend = "pgvector"
    else:
        chunks = chatbot_repository.get_chunks(include_embeddings=True)
        vector_rows = []
        for chunk in chunks:
            vector_rows.append(
                {
                    **chunk,
                    "semantic_score": _compute_cosine_similarity(
                        query_embedding,
                        list(chunk.get("embedding") or []),
                    ),
                }
            )
        vector_rows = sorted(
            vector_rows,
            key=lambda item: item["semantic_score"],
            reverse=True,
        )[: max(1, topk_vector)]
        retrieval_backend = "python_cosine"

    keyword_seed = chatbot_repository.keyword_search(all_queries, topk_keyword)
    if keyword_seed:
        keyword_rows = []
        for chunk in keyword_seed:
            best_keyword = 0.0
            for candidate_query in all_queries:
                best_keyword = max(
                    best_keyword,
                    _compute_keyword_overlap(candidate_query, chunk["content"]),
                )
            keyword_rows.append({**chunk, "keyword_score": best_keyword})
        keyword_rows = sorted(
            keyword_rows,
            key=lambda item: item["keyword_score"],
            reverse=True,
        )[: max(1, topk_keyword)]
    else:
        keyword_rows = []

    for item in vector_rows:
        candidates[item["chunk_id"]] = {
            **item,
            "semantic_score": float(item.get("semantic_score", 0.0)),
            "keyword_score": 0.0,
            "retrieval_score": float(item.get("semantic_score", 0.0)) * 0.7,
        }

    for item in keyword_rows:
        current = candidates.get(item["chunk_id"])
        if not current:
            candidates[item["chunk_id"]] = {
                **item,
                "semantic_score": 0.0,
                "keyword_score": float(item.get("keyword_score", 0.0)),
                "retrieval_score": float(item.get("keyword_score", 0.0)) * 0.3,
            }
            continue

        semantic_score = float(current["semantic_score"])
        keyword_score = float(item.get("keyword_score", 0.0))
        current["keyword_score"] = keyword_score
        current["retrieval_score"] = semantic_score * 0.7 + keyword_score * 0.3

    return {
        "vector_candidates": len(vector_rows),
        "keyword_candidates": len(keyword_rows),
        "retrieval_backend": retrieval_backend,
        "candidates": sorted(
            candidates.values(),
            key=lambda item: item["retrieval_score"],
            reverse=True,
        ),
    }
