from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_RELATION_MAP = {
    "left_of": "left_of",
    "right_of": "right_of",

    "above": "above",
    "over": "above",
    "on_top_of": "above",

    "below": "below",
    "under": "below",
    "beneath": "below",

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--success",
        type=str,
        default="data/srg_bench_v01/bbox_srg_bench_v01_subset.jsonl",
    )
    parser.add_argument(
        "--failed",
        type=str,
        default="data/srg_bench_v01/bbox_srg_failed_subset.jsonl",
    )
    args = parser.parse_args()

    success_records = read_jsonl(Path(args.success))
    failed_records = read_jsonl(Path(args.failed))

    total = len(success_records) + len(failed_records)

    print("=" * 80)
    print("BBox-SRG Analysis")
    print("=" * 80)

    print(f"Success file: {args.success}")
    print(f"Failed file: {args.failed}")
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

    directional_records = []
    diagnosis_counter = Counter()
    label_stats = defaultdict(lambda: Counter())
    group_detect_stats = defaultdict(lambda: Counter())
    group_diag_stats = defaultdict(lambda: Counter())

    for item in success_records:
        caption_raw = item["bbox_evidence"].get("caption_relation", item.get("relation", ""))
        bbox_raw = item["bbox_evidence"].get("bbox_relation", "")

        caption_rel = normalize_relation(caption_raw)
        bbox_rel = normalize_relation(bbox_raw)

        relation_group = item.get("relation_group", "unknown")
        label = bool(item.get("label"))

        is_directional = caption_rel in DIRECTIONAL_RELATIONS and bbox_rel in DIRECTIONAL_RELATIONS

        if is_directional:
            directional_records.append(item)

            matched = caption_rel == bbox_rel
            diagnostic_correct = matched if label else not matched

            diagnosis_counter[str(diagnostic_correct)] += 1

            label_stats[str(label)]["total"] += 1
            label_stats[str(label)]["match"] += int(matched)
            label_stats[str(label)]["mismatch"] += int(not matched)
            label_stats[str(label)]["diagnostic_correct"] += int(diagnostic_correct)

            group_diag_stats[relation_group]["total"] += 1
            group_diag_stats[relation_group]["diagnostic_correct"] += int(diagnostic_correct)

    success_group_counter = Counter(item.get("relation_group", "unknown") for item in success_records)
    failed_group_counter = Counter(item.get("relation_group", "unknown") for item in failed_records)

    print("\n" + "-" * 80)
    print("Detection success by relation_group")
    print("-" * 80)

    all_groups = sorted(set(success_group_counter) | set(failed_group_counter))

    for group in all_groups:
        s = success_group_counter.get(group, 0)
        f = failed_group_counter.get(group, 0)
        group_total = s + f
        group_detect_stats[group]["total"] = group_total
        group_detect_stats[group]["success"] = s
        group_detect_stats[group]["failed"] = f

        print(
            f"{group}: total={group_total}, success={s}, failed={f}, "
            f"success_rate={safe_div(s, group_total):.4f}"
        )

    directional_total = len(directional_records)
    diagnostic_true = diagnosis_counter.get("True", 0)
    diagnostic_false = diagnosis_counter.get("False", 0)

    print("\n" + "-" * 80)
    print("Directional diagnostic consistency")
    print("-" * 80)

    print(f"Directional success records: {directional_total}")
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
            f"diagnostic_correct={counter['diagnostic_correct']}, "
            f"diagnostic_rate={safe_div(counter['diagnostic_correct'], total_label):.4f}"
        )

    print("\n" + "-" * 80)
    print("Diagnostic consistency by relation_group")
    print("-" * 80)

    for group, counter in sorted(group_diag_stats.items()):
        total_group = counter["total"]
        print(
            f"{group}: directional_total={total_group}, "
            f"diagnostic_correct={counter['diagnostic_correct']}, "
            f"diagnostic_rate={safe_div(counter['diagnostic_correct'], total_group):.4f}"
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
    print("Analysis finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
