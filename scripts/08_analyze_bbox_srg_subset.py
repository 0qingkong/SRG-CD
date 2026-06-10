from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


SUCCESS_PATH = Path("data/srg_bench_v01/bbox_srg_bench_v01_subset.jsonl")
FAILED_PATH = Path("data/srg_bench_v01/bbox_srg_failed_subset.jsonl")


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


def main() -> None:
    success_records = read_jsonl(SUCCESS_PATH)
    failed_records = read_jsonl(FAILED_PATH)

    total = len(success_records) + len(failed_records)

    print("=" * 80)
    print("BBox-SRG Subset Analysis")
    print("=" * 80)

    print(f"Total records: {total}")
    print(f"Success records: {len(success_records)}")
    print(f"Failed records: {len(failed_records)}")
    print(f"Success rate: {safe_div(len(success_records), total):.4f}")

    print("\n" + "-" * 80)
    print("Failure reason distribution")
    print("-" * 80)

    failure_counter = Counter(item.get("bbox_error", "unknown") for item in failed_records)
    for key, value in failure_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("Caption-BBox relation match")
    print("-" * 80)

    match_counter = Counter(
        item["bbox_evidence"].get("caption_bbox_relation_match", False)
        for item in success_records
    )

    match_true = match_counter.get(True, 0)
    match_false = match_counter.get(False, 0)

    print(f"Match true: {match_true}")
    print(f"Match false: {match_false}")
    print(f"Match rate among success: {safe_div(match_true, len(success_records)):.4f}")

    print("\n" + "-" * 80)
    print("Match distribution by original label")
    print("-" * 80)

    label_stats = defaultdict(lambda: Counter())

    for item in success_records:
        label = item.get("label")
        match = item["bbox_evidence"].get("caption_bbox_relation_match", False)
        label_stats[str(label)][str(match)] += 1

    for label, counter in label_stats.items():
        total_label = sum(counter.values())
        true_count = counter.get("True", 0)
        false_count = counter.get("False", 0)
        print(
            f"label={label}: total={total_label}, "
            f"match={true_count}, mismatch={false_count}, "
            f"match_rate={safe_div(true_count, total_label):.4f}"
        )

    print("\n" + "-" * 80)
    print("Success distribution by relation_group")
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
    print("BBox relation distribution")
    print("-" * 80)

    bbox_relation_counter = Counter(
        item["bbox_evidence"].get("bbox_relation", "unknown")
        for item in success_records
    )

    for key, value in bbox_relation_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("Analysis finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
