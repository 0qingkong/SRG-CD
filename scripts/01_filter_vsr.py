from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from srg.sro_parser import parse_sro_from_caption
from srg.relations import normalize_relation


DEFAULT_TARGET_RELATIONS = {
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
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_vsr_from_huggingface(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("请先安装 datasets：pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    return [dict(item) for item in dataset]


def get_first_existing_key(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def normalize_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return bool(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False

    raise ValueError(f"无法解析 label：{value}")


def normalize_vsr_item(
    item: dict[str, Any],
    index: int,
    split: str,
    image_root: str,
) -> dict[str, Any] | None:
    caption = get_first_existing_key(item, ["caption", "sentence", "text"])
    if not caption:
        return None

    label_raw = get_first_existing_key(item, ["label", "answer", "truth", "is_true"])
    if label_raw is None:
        return None

    try:
        label = normalize_label(label_raw)
    except ValueError:
        return None

    image = get_first_existing_key(
        item,
        ["image", "image_id", "image_file", "file_name", "filename"],
    )

    relation_raw = get_first_existing_key(item, ["relation", "rel"])
    relation = normalize_relation(relation_raw) if relation_raw else None

    if relation is None:
        parsed = parse_sro_from_caption(caption)
        relation = parsed.relation if parsed.success else None

    if relation is None:
        return None

    image_path = None
    if image:
        image_path = str(Path(image_root) / str(image))

    return {
        "id": f"vsr_{split}_{index:06d}",
        "source": "VSR",
        "split": split,
        "image": image,
        "image_path": image_path,
        "caption": caption,
        "label": label,
        "relation": relation,
        "raw": item,
    }


def balanced_sample(
    records: list[dict[str, Any]],
    max_per_relation: int | None,
    max_total: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    random.seed(seed)

    buckets = defaultdict(list)
    for item in records:
        key = (item["relation"], item["label"])
        buckets[key].append(item)

    sampled = []

    relations = sorted({item["relation"] for item in records})
    for relation in relations:
        pos = buckets[(relation, True)]
        neg = buckets[(relation, False)]

        random.shuffle(pos)
        random.shuffle(neg)

        if max_per_relation is None:
            n = min(len(pos), len(neg))
        else:
            n = min(len(pos), len(neg), max_per_relation // 2)

        sampled.extend(pos[:n])
        sampled.extend(neg[:n])

    random.shuffle(sampled)

    if max_total is not None:
        sampled = sampled[:max_total]

    return sampled


def save_relation_stats(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    counter = Counter((item["relation"], item["label"]) for item in records)

    relations = sorted({item["relation"] for item in records})
    for relation in relations:
        true_count = counter[(relation, True)]
        false_count = counter[(relation, False)]
        rows.append(
            {
                "relation": relation,
                "true_count": true_count,
                "false_count": false_count,
                "total": true_count + false_count,
            }
        )

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--use_hf", action="store_true")
    parser.add_argument("--hf_dataset", type=str, default="cambridgeltl/vsr_random")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--image_root", type=str, default="data/raw/vsr/images")
    parser.add_argument("--output", type=str, default="data/processed/vsr_filtered.jsonl")
    parser.add_argument(
        "--stats_output",
        type=str,
        default="results/tables/relation_distribution.csv",
    )

    parser.add_argument("--max_per_relation", type=int, default=200)
    parser.add_argument("--max_total", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.use_hf:
        raw_data = load_vsr_from_huggingface(args.hf_dataset, args.split)
    else:
        if args.input is None:
            raise ValueError("请提供 --input 或使用 --use_hf")
        raw_data = read_jsonl(Path(args.input))

    normalized_records = []

    for idx, item in enumerate(tqdm(raw_data, desc="Normalizing VSR")):
        record = normalize_vsr_item(
            item=item,
            index=idx,
            split=args.split,
            image_root=args.image_root,
        )
        if record is None:
            continue

        if record["relation"] not in DEFAULT_TARGET_RELATIONS:
            continue

        normalized_records.append(record)

    sampled_records = balanced_sample(
        records=normalized_records,
        max_per_relation=args.max_per_relation,
        max_total=args.max_total,
        seed=args.seed,
    )

    write_jsonl(sampled_records, Path(args.output))
    save_relation_stats(sampled_records, Path(args.stats_output))

    print(f"Raw samples: {len(raw_data)}")
    print(f"Filtered samples: {len(normalized_records)}")
    print(f"Sampled samples: {len(sampled_records)}")
    print(f"Saved to: {args.output}")
    print(f"Stats saved to: {args.stats_output}")


if __name__ == "__main__":
    main()