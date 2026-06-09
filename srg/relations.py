from __future__ import annotations


SUPPORTED_RELATIONS = {
    "left of",
    "right of",
    "above",
    "below",
    "under",
    "over",
    "inside",
    "outside",
    "near",
    "far from",
    "next to",
    "on top of",
    "beneath",
}


RELATION_ALIASES = {
    "left": "left of",
    "to the left of": "left of",
    "right": "right of",
    "to the right of": "right of",
    "underneath": "under",
    "beneath": "under",
    "on top of": "over",
    "close to": "near",
    "near to": "near",
    "far away from": "far from",
    "next to": "near",
}


INVERSE_RELATIONS = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
    "under": "over",
    "over": "under",
    "inside": "outside",
    "outside": "inside",
    "near": "far from",
    "far from": "near",
    "next to": "far from",
    "on top of": "under",
    "beneath": "over",
}


RELATION_GROUPS = {
    "left of": "horizontal",
    "right of": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "under": "vertical",
    "over": "vertical",
    "inside": "containment",
    "outside": "containment",
    "near": "distance",
    "far from": "distance",
    "next to": "distance",
    "on top of": "vertical",
    "beneath": "vertical",
}


def normalize_relation(relation: str | None) -> str | None:
    if relation is None:
        return None

    r = relation.strip().lower()
    r = r.replace("_", " ")
    r = " ".join(r.split())

    if r in RELATION_ALIASES:
        r = RELATION_ALIASES[r]

    if r in SUPPORTED_RELATIONS:
        return r

    return None


def relation_to_key(relation: str) -> str:
    return relation.strip().lower().replace(" ", "_")


def get_inverse_relation(relation: str) -> str | None:
    relation = normalize_relation(relation)
    if relation is None:
        return None
    return INVERSE_RELATIONS.get(relation)


def get_relation_group(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation is None:
        return "unknown"
    return RELATION_GROUPS.get(relation, "unknown")


def get_supported_relations_sorted() -> list[str]:
    return sorted(SUPPORTED_RELATIONS, key=len, reverse=True)