from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/srg_bench_v01/srg_bench_v01.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tables",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_counter = Counter(item["relation"] for item in records)
    group_counter = Counter(item["relation_group"] for item in records)
    label_counter = Counter(item["label"] for item in records)

    cf_counter = Counter()
    for item in records:
        cfs = item.get("counterfactuals", {})
        for key in ["relation_flip", "object_swap", "wrong_srg"]:
            if cfs.get(key) is not None:
                cf_counter[key] += 1

    print("=" * 80)
    print("SRG-Bench v0.1 Summary")
    print("=" * 80)
    print(f"Total records: {len(records)}")

    print("\nLabel distribution:")
    for k, v in label_counter.items():
        print(f"  {k}: {v}")

    print("\nRelation group distribution:")
    for k, v in group_counter.most_common():
        print(f"  {k}: {v}")

    print("\nRelation distribution:")
    for k, v in relation_counter.most_common():
        print(f"  {k}: {v}")

    print("\nCounterfactual coverage:")
    for k, v in cf_counter.items():
        print(f"  {k}: {v}")

    pd.DataFrame(
        [{"relation": k, "count": v} for k, v in relation_counter.most_common()]
    ).to_csv(output_dir / "srg_relation_counts.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [{"relation_group": k, "count": v} for k, v in group_counter.most_common()]
    ).to_csv(output_dir / "srg_relation_group_counts.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [{"label": str(k), "count": v} for k, v in label_counter.items()]
    ).to_csv(output_dir / "srg_label_counts.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [{"counterfactual_type": k, "count": v} for k, v in cf_counter.items()]
    ).to_csv(output_dir / "srg_counterfactual_coverage.csv", index=False, encoding="utf-8-sig")

    print("\nSaved analysis tables to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
