from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from srg.counterfactual import build_srg_bench_record


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/vsr_sro_parsed.jsonl")
    parser.add_argument(
        "--output",
        type=str,
        default="data/srg_bench_v01/srg_bench_v01.jsonl",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    output_records = []

    skipped = 0

    for item in tqdm(records, desc="Generating SRG-Bench v0.1"):
        try:
            record = build_srg_bench_record(
                sample_id=item["id"],
                source=item.get("source", "VSR"),
                split=item.get("split", "unknown"),
                image=item.get("image"),
                image_path=item.get("image_path"),
                caption=item["caption"],
                label=bool(item["label"]),
                subject=item["subject"],
                relation=item["relation"],
                obj=item["object"],
            )

            record["metadata"] = {
                "raw_relation": item.get("relation"),
                "raw_id": item.get("id"),
            }

            output_records.append(record)

        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {item.get('id')} because {exc}")

    write_jsonl(output_records, Path(args.output))

    print(f"Input records: {len(records)}")
    print(f"Output records: {len(output_records)}")
    print(f"Skipped records: {skipped}")
    print(f"Saved SRG-Bench v0.1 to: {args.output}")


if __name__ == "__main__":
    main()