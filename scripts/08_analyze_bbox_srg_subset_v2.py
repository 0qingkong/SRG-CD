from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


SUCCESS_PATH = Path("data/srg_bench_v01/bbox_srg_bench_v01_subset.jsonl")
FAILED_PATH = Path("data/srg_bench_v01/bbox_srg_failed_subset.jsonl")


CANONICAL_RELATION_MAP = {
    "left_of": "left_of",
    "right_of": "right_of",

    "above": "above",
    "over": "above",
    "on_top_of": "above",

    "below": "below",
    "under": "below",
    "beneath": "below",

    # 这些关系当前不能仅靠中心点 left/right/above/below 稳定判断
    "inside": "containment",
    "outside": "containment",
    "near": "distance",
    "next_to": "distance",
    "far_from": "distance",
}


DIRECTIONAL_RELATIONS = {"left_of", "right_of", "above", "below"}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def normalize_relation(relation: str) -> str:
    key = relation.strip().lower().replace(" ", "_")
    return CANONICAL_RELATION_MAP.get(key, key)


def is_directional_relation(relation: str) -> bool:
    return normalize_relation(relation) in DIRECTIONAL_RELATIONS


def main() -> None:
    success_records = read_jsonl(SUCCESS_PATH)
    failed_records = read_jsonl(FAILED_PATH)

    total = len(success_records) + len(failed_records)

    print("=" * 80)
    print("BBox-SRG Subset Analysis v2")
    print("=" * 80)

    print(f"Total records: {total}")
    print(f"Success records: {len(success_records)}")
    print(f"Failed records: {len(failed_records)}")
    print(f"Detection success rate: {safe_div(len(success_records), total):.4f}")

    print("\n" + "-" * 80)
    print("Failure reason distribution")
    print("-" * 80)

    failure_counter = Counter(item.get("bbox_error", "unknown") for item in failed_records)
    for key, value in failure_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("Canonical caption-bbox relation match")
    print("-" * 80)

    match_counter = Counter()
    diagnosis_counter = Counter()
    label_stats = defaultdict(lambda: Counter())
    group_stats = defaultdict(lambda: Counter())

    directional_success_records = []

    for item in success_records:
        caption_raw = item["bbox_evidence"].get("caption_relation", item.get("relation", ""))
        bbox_raw = item["bbox_evidence"].get("bbox_relation", "")

        caption_rel = normalize_relation(caption_raw)
        bbox_rel = normalize_relation(bbox_raw)

        relation_group = item.get("relation_group", "unknown")
        label = bool(item.get("label"))

        is_directional = caption_rel in DIRECTIONAL_RELATIONS and bbox_rel in DIRECTIONAL_RELATIONS

        if is_directional:
            directional_success_records.append(item)

            matched = caption_rel == bbox_rel

            # label=True：caption 应该被 bbox 支持
            # label=False：caption 应该被 bbox 反驳
            diagnostically_correct = matched if label else not matched

            match_counter[str(matched)] += 1
            diagnosis_counter[str(diagnostically_correct)] += 1

            label_stats[str(label)]["total"] += 1
            label_stats[str(label)]["match"] += int(matched)
            label_stats[str(label)]["mismatch"] += int(not matched)
            label_stats[str(label)]["diagnostic_correct"] += int(diagnostically_correct)

            group_stats[relation_group]["total"] += 1
            group_stats[relation_group]["match"] += int(matched)
            group_stats[relation_group]["diagnostic_correct"] += int(diagnostically_correct)
        else:
            group_stats[relation_group]["non_directional_skipped"] += 1

    directional_total = len(directional_success_records)

    print(f"Directional success records: {directional_total}")
    print(f"Canonical match true: {match_counter.get('True', 0)}")
    print(f"Canonical match false: {match_counter.get('False', 0)}")
    print(
        f"Canonical match rate: "
        f"{safe_div(match_counter.get('True', 0), directional_total):.4f}"
    )

    print("\n" + "-" * 80)
    print("Diagnostic consistency")
    print("-" * 80)

    diagnostic_true = diagnosis_counter.get("True", 0)
    diagnostic_false = diagnosis_counter.get("False", 0)

    print(f"Diagnostic correct: {diagnostic_true}")
    print(f"Diagnostic incorrect: {diagnostic_false}")
    print(f"Diagnostic consistency rate: {safe_div(diagnostic_true, directional_total):.4f}")

    print("\n" + "-" * 80)
    print("Diagnostic distribution by original label")
    print("-" * 80)

    for label, counter in sorted(label_stats.items()):
        total_label = counter["total"]
        print(
            f"label={label}: "
            f"total={total_label}, "
            f"match={counter['match']}, "
            f"mismatch={counter['mismatch']}, "
            f"match_rate={safe_div(counter['match'], total_label):.4f}, "
            f"diagnostic_correct={counter['diagnostic_correct']}, "
            f"diagnostic_rate={safe_div(counter['diagnostic_correct'], total_label):.4f}"
        )

    print("\n" + "-" * 80)
    print("Detection success distribution by relation_group")
    print("-" * 80)

    success_group_counter = Counter(item.get("relation_group", "unknown") for item in success_records)
    failed_group_counter = Counter(item.get("relation_group", "unknown") for item in failed_records)

    all_groups = sorted(set(success_group_counter) | set(failed_group_counter))

    for group in all_groups:
        s = success_group_counter.get(group, 0)
        f = failed_group_counter.get(group, 0)
        group_total = s + f
        print(
            f"{group}: total={group_total}, success={s}, failed={f}, "
            f"success_rate={safe_div(s, group_total):.4f}"
        )

    print("\n" + "-" * 80)
    print("Diagnostic consistency by relation_group")
    print("-" * 80)

    for group, counter in sorted(group_stats.items()):
        total_group = counter["total"]
        if total_group == 0:
            print(
                f"{group}: directional_total=0, "
                f"non_directional_skipped={counter['non_directional_skipped']}"
            )
            continue

        print(
            f"{group}: "
            f"directional_total={total_group}, "
            f"match={counter['match']}, "
            f"diagnostic_correct={counter['diagnostic_correct']}, "
            f"diagnostic_rate={safe_div(counter['diagnostic_correct'], total_group):.4f}, "
            f"non_directional_skipped={counter['non_directional_skipped']}"
        )

    print("\n" + "-" * 80)
    print("BBox relation distribution")
    print("-" * 80)

    bbox_relation_counter = Counter(
        normalize_relation(item["bbox_evidence"].get("bbox_relation", "unknown"))
        for item in success_records
    )

    for key, value in bbox_relation_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("Analysis v2 finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
