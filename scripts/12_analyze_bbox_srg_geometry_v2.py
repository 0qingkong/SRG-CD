from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/srg_bench_v01/bbox_srg_bench_v01_500_v2.jsonl",
    )
    parser.add_argument(
        "--failed",
        type=str,
        default="data/srg_bench_v01/bbox_srg_failed_500.jsonl",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    failed_records = read_jsonl(Path(args.failed)) if Path(args.failed).exists() else []

    total_attempted = len(records) + len(failed_records)

    print("=" * 80)
    print("BBox-SRG Geometry v2 Analysis")
    print("=" * 80)
    print(f"Input success records: {len(records)}")
    print(f"Failed detection records: {len(failed_records)}")
    print(f"Total attempted records: {total_attempted}")
    print(f"Detection success rate: {safe_div(len(records), total_attempted):.4f}")

    family_counter = Counter()
    relation_counter = Counter()
    supported_counter = Counter()
    diagnostic_counter = Counter()
    label_stats = defaultdict(lambda: Counter())
    group_stats = defaultdict(lambda: Counter())
    family_stats = defaultdict(lambda: Counter())

    for item in records:
        ev = item["bbox_evidence_v2"]

        family = ev["relation_family_v2"]
        bbox_relation = ev["bbox_relation_v2"]
        supported = bool(ev["relation_supported_v2"])
        diag = ev["diagnostic_correct_v2"]
        label = str(bool(item["label"]))
        group = item.get("relation_group", "unknown")

        family_counter[family] += 1
        relation_counter[bbox_relation] += 1
        supported_counter[str(supported)] += 1

        if diag is not None:
            diagnostic_counter[str(diag)] += 1

            label_stats[label]["total"] += 1
            label_stats[label]["diagnostic_correct"] += int(bool(diag))

            group_stats[group]["total"] += 1
            group_stats[group]["diagnostic_correct"] += int(bool(diag))

            family_stats[family]["total"] += 1
            family_stats[family]["diagnostic_correct"] += int(bool(diag))

    supported_total = diagnostic_counter.get("True", 0) + diagnostic_counter.get("False", 0)

    print("\n" + "-" * 80)
    print("Relation family distribution v2")
    print("-" * 80)
    for key, value in family_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("BBox relation distribution v2")
    print("-" * 80)
    for key, value in relation_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("Supported relation records v2")
    print("-" * 80)
    for key, value in supported_counter.most_common():
        print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("Diagnostic consistency v2")
    print("-" * 80)
    print(f"Supported diagnostic records: {supported_total}")
    print(f"Diagnostic correct: {diagnostic_counter.get('True', 0)}")
    print(f"Diagnostic incorrect: {diagnostic_counter.get('False', 0)}")
    print(
        f"Diagnostic consistency rate: "
        f"{safe_div(diagnostic_counter.get('True', 0), supported_total):.4f}"
    )

    print("\n" + "-" * 80)
    print("Diagnostic by original label v2")
    print("-" * 80)
    for label, counter in sorted(label_stats.items()):
        total = counter["total"]
        correct = counter["diagnostic_correct"]
        print(
            f"label={label}: total={total}, diagnostic_correct={correct}, "
            f"diagnostic_rate={safe_div(correct, total):.4f}"
        )

    print("\n" + "-" * 80)
    print("Diagnostic by relation_group v2")
    print("-" * 80)
    for group, counter in sorted(group_stats.items()):
        total = counter["total"]
        correct = counter["diagnostic_correct"]
        print(
            f"{group}: total={total}, diagnostic_correct={correct}, "
            f"diagnostic_rate={safe_div(correct, total):.4f}"
        )

    print("\n" + "-" * 80)
    print("Diagnostic by relation_family v2")
    print("-" * 80)
    for family, counter in sorted(family_stats.items()):
        total = counter["total"]
        correct = counter["diagnostic_correct"]
        print(
            f"{family}: total={total}, diagnostic_correct={correct}, "
            f"diagnostic_rate={safe_div(correct, total):.4f}"
        )

    print("\n" + "=" * 80)
    print("Geometry v2 analysis finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
