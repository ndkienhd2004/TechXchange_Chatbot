from __future__ import annotations

import unicodedata


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def normalize(text: str) -> str:
    lowered = strip_accents(text).lower()
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered).split())


def normalize_preserve_accents(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "")).split())


def tokenize_text(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token]


def tokenize_text_preserve_accents(text: str) -> list[str]:
    return [token for token in normalize_preserve_accents(text).split() if token]


def is_top_selling_intent(message: str) -> bool:
    normalized = normalize(message)
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


def is_build_pc_intent(message: str) -> bool:
    normalized = normalize(message)
    if not normalized:
        return False
    return (
        ("build" in normalized and "pc" in normalized)
        or "cau hinh pc" in normalized
        or "lap pc" in normalized
    )


def wants_multiple_builds(message: str) -> bool:
    normalized = normalize(message)
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


def is_policy_intent(message: str) -> bool:
    normalized = normalize(message)
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
