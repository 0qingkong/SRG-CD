from __future__ import annotations

import re
from dataclasses import dataclass

from srg.relations import normalize_relation, get_supported_relations_sorted


ARTICLES = {"the", "a", "an"}


@dataclass
class SROResult:
    subject: str | None
    relation: str | None
    object: str | None
    success: bool
    error: str | None = None


def clean_phrase(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.,!?;:]+$", "", text)
    text = " ".join(text.split())

    parts = text.split()

    while parts and parts[0] in ARTICLES:
        parts = parts[1:]

    if parts and parts[-1] in {"is", "are", "was", "were", "be", "being"}:
        parts = parts[:-1]

    return " ".join(parts).strip()


def remove_copula_from_subject(text: str) -> str:
    """
    处理：
    "the cup is" -> "cup"
    "a dog is" -> "dog"
    """
    text = text.strip().lower()
    text = re.sub(r"\b(is|are|was|were)\s*$", "", text).strip()
    return clean_phrase(text)


def parse_sro_from_caption(caption: str) -> SROResult:
    if not caption or not isinstance(caption, str):
        return SROResult(None, None, None, False, "empty_caption")

    normalized_caption = caption.strip().lower()
    normalized_caption = normalized_caption.replace("’", "'")
    normalized_caption = " ".join(normalized_caption.split())

    candidate_relations = get_supported_relations_sorted()

    matched_relation = None
    matched_span = None

    for relation in candidate_relations:
        pattern = r"\b" + re.escape(relation) + r"\b"
        match = re.search(pattern, normalized_caption)
        if match:
            matched_relation = relation
            matched_span = match.span()
            break

    if matched_relation is None or matched_span is None:
        return SROResult(None, None, None, False, "unsupported_or_missing_relation")

    left_part = normalized_caption[: matched_span[0]].strip()
    right_part = normalized_caption[matched_span[1] :].strip()

    subject = remove_copula_from_subject(left_part)
    obj = clean_phrase(right_part)
    relation = normalize_relation(matched_relation)

    if not subject:
        return SROResult(None, relation, obj, False, "missing_subject")

    if not obj:
        return SROResult(subject, relation, None, False, "missing_object")

    if relation is None:
        return SROResult(subject, None, obj, False, "unsupported_relation")

    return SROResult(subject, relation, obj, True, None)


def build_caption(subject: str, relation: str, obj: str) -> str:
    subject = clean_phrase(subject)
    obj = clean_phrase(obj)
    relation = normalize_relation(relation) or relation

    return f"The {subject} is {relation} the {obj}."