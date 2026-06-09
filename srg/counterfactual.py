from __future__ import annotations

from srg.relations import get_inverse_relation, relation_to_key, get_relation_group
from srg.sro_parser import build_caption


def build_caption_level_srg(subject: str, relation: str, obj: str) -> dict:
    normalized_relation = relation_to_key(relation)

    return {
        "nodes": [
            {
                "id": "subject",
                "name": subject,
                "role": "subject",
            },
            {
                "id": "object",
                "name": obj,
                "role": "object",
            },
        ],
        "edges": [
            {
                "source": "subject",
                "target": "object",
                "relation": normalized_relation,
                "evidence_type": "caption_level",
                "confidence": 1.0,
            }
        ],
    }


def generate_counterfactuals(
    subject: str,
    relation: str,
    obj: str,
    original_label: bool,
) -> dict:
    inverse_relation = get_inverse_relation(relation)

    if inverse_relation is None:
        return {
            "relation_flip": None,
            "object_swap": None,
            "wrong_srg": None,
        }

    relation_flip_caption = build_caption(subject, inverse_relation, obj)

    # object swap 的语义：
    # A left of B 等价于 B right of A
    object_swap_caption = build_caption(obj, inverse_relation, subject)

    wrong_srg = {
        "edge": ["subject", relation_to_key(inverse_relation), "object"],
        "relation": inverse_relation,
        "normalized_relation": relation_to_key(inverse_relation),
        "conflict_with_caption": True,
    }

    return {
        "relation_flip": {
            "caption": relation_flip_caption,
            "relation": inverse_relation,
            "normalized_relation": relation_to_key(inverse_relation),
            "expected_label": not original_label,
            "type": "relation_flip",
        },
        "object_swap": {
            "caption": object_swap_caption,
            "subject": obj,
            "relation": inverse_relation,
            "object": subject,
            "normalized_relation": relation_to_key(inverse_relation),
            "expected_label": original_label,
            "type": "object_swap",
        },
        "wrong_srg": wrong_srg,
    }


def build_srg_bench_record(
    sample_id: str,
    source: str,
    split: str,
    image: str | None,
    image_path: str | None,
    caption: str,
    label: bool,
    subject: str,
    relation: str,
    obj: str,
) -> dict:
    return {
        "id": sample_id,
        "source": source,
        "split": split,
        "image": image,
        "image_path": image_path,
        "caption": caption,
        "label": bool(label),
        "subject": subject,
        "relation": relation,
        "object": obj,
        "relation_group": get_relation_group(relation),
        "normalized_relation": relation_to_key(relation),
        "srg": build_caption_level_srg(subject, relation, obj),
        "counterfactuals": generate_counterfactuals(subject, relation, obj, bool(label)),
        "quality_flags": {
            "sro_parsed": True,
            "relation_supported": True,
            "needs_manual_check": False,
        },
    }