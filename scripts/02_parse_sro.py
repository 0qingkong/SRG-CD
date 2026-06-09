from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from srg.sro_parser import parse_sro_from_caption


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
    parser.add_argument("--input", type=str, default="data/processed/vsr_filtered.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/vsr_sro_parsed.jsonl")
    parser.add_argument(
        "--error_output",
        type=str,
        default="data/processed/vsr_sro_parse_errors.jsonl",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))

    parsed_records = []
    error_records = []

    for item in tqdm(records, desc="Parsing SRO"):
        caption = item.get("caption", "")
        result = parse_sro_from_caption(caption)

        if result.success:
            new_item = dict(item)
            new_item["subject"] = result.subject
            new_item["relation"] = result.relation
            new_item["object"] = result.object
            new_item["quality_flags"] = {
                "sro_parsed": True,
                "relation_supported": True,
                "needs_manual_check": False,
            }
            parsed_records.append(new_item)
        else:
            error_item = dict(item)
            error_item["parse_error"] = result.error
            error_item["quality_flags"] = {
                "sro_parsed": False,
                "relation_supported": False,
                "needs_manual_check": True,
            }
            error_records.append(error_item)

    write_jsonl(parsed_records, Path(args.output))
    write_jsonl(error_records, Path(args.error_output))

    print(f"Input records: {len(records)}")
    print(f"Parsed records: {len(parsed_records)}")
    print(f"Parse errors: {len(error_records)}")
    print(f"Saved parsed records to: {args.output}")
    print(f"Saved error records to: {args.error_output}")


if __name__ == "__main__":
    main()