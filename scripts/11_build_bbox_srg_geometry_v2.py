from __future__ import annotations

import argparse
import json
from pathlib import Path

from srg.bbox_geometry import infer_relation_v2


CANONICAL_RELATION_MAP = {
    "left_of": "left_of",
    "right_of": "right_of",

    "above": "above",
    "over": "above",
    "on_top_of": "above",

    "below": "below",
    "under": "below",
    "beneath": "below",

    "inside": "inside",
    "outside": "outside",

    "near": "near",
    "next_to": "near",
    "far_from": "far_from",
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_relation(relation: str) -> str:
    key = relation.strip().lower().replace(" ", "_")
    return CANONICAL_RELATION_MAP.get(key, key)


def is_supported_v2_relation(relation: str) -> bool:
    return normalize_relation(relation) in {
        "left_of",
        "right_of",
        "above",
        "below",
        "inside",
        "outside",
        "near",
        "far_from",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/srg_bench_v01/bbox_srg_bench_v01_500.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/srg_bench_v01/bbox_srg_bench_v01_500_v2.jsonl",
    )
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))

    output_records = []

    for item in records:
        bbox_evidence = item["bbox_evidence"]

        width, height = bbox_evidence["image_size"]

        subject_det = bbox_evidence["subject_detection"]
        object_det = bbox_evidence["object_detection"]

        subject_box = subject_det["box"]
        object_box = object_det["box"]

        caption_relation_raw = item["relation"]
        caption_relation = normalize_relation(caption_relation_raw)

        relation_v2 = infer_relation_v2(
            subject_box=subject_box,
            object_box=object_box,
            width=width,
            height=height,
            caption_relation=caption_relation,
        )

        bbox_relation_v2 = normalize_relation(relation_v2["computed_relation"])

        label = bool(item["label"])
        relation_supported_v2 = is_supported_v2_relation(caption_relation)

        caption_bbox_match_v2 = caption_relation == bbox_relation_v2

        if relation_supported_v2:
            diagnostic_correct_v2 = (
                caption_bbox_match_v2 if label else not caption_bbox_match_v2
            )
        else:
            diagnostic_correct_v2 = None

        geometry_features = relation_v2["geometry_features"]

        new_item = dict(item)

        # 保留原始 bbox_evidence，同时新增 v2 结果，不覆盖旧结果
        new_item["bbox_evidence_v2"] = {
            "detector": bbox_evidence.get("subject_detection", {}).get(
                "detector",
                "google/owlvit-base-patch32",
            ),
            "image_size": [width, height],
            "caption_relation_raw": caption_relation_raw,
            "caption_relation_canonical": caption_relation,
            "bbox_relation_v1": bbox_evidence.get("bbox_relation"),
            "bbox_relation_v2": bbox_relation_v2,
            "relation_family_v2": relation_v2["relation_family"],
            "relation_supported_v2": relation_supported_v2,
            "caption_bbox_match_v2": caption_bbox_match_v2,
            "diagnostic_correct_v2": diagnostic_correct_v2,
            "geometry_features": geometry_features,
        }

        # 构建 v2 bbox_srg
        new_item["bbox_srg_v2"] = {
            "nodes": [
                {
                    "id": "subject",
                    "name": item["subject"],
                    "role": "subject",
                    "bbox": geometry_features["subject_box_clipped"],
                    "score": subject_det["score"],
                },
                {
                    "id": "object",
                    "name": item["object"],
                    "role": "object",
                    "bbox": geometry_features["object_box_clipped"],
                    "score": object_det["score"],
                },
            ],
            "edges": [
                {
                    "source": "subject",
                    "target": "object",
                    "relation": bbox_relation_v2,
                    "relation_family": relation_v2["relation_family"],
                    "evidence_type": "bbox_level_geometry_v2",
                    "geometry_features": {
                        "dx": geometry_features["dx"],
                        "dy": geometry_features["dy"],
                        "iou": geometry_features["iou"],
                        "subject_in_object_ratio": geometry_features[
                            "subject_in_object_ratio"
                        ],
                        "object_in_subject_ratio": geometry_features[
                            "object_in_subject_ratio"
                        ],
                        "normalized_center_distance": geometry_features[
                            "normalized_center_distance"
                        ],
                        "x_overlap_ratio": geometry_features["x_overlap_ratio"],
                        "y_overlap_ratio": geometry_features["y_overlap_ratio"],
                    },
                }
            ],
        }

        output_records.append(new_item)

    write_jsonl(Path(args.output), output_records)

    print("=" * 80)
    print("BBox-SRG Geometry v2 finished")
    print("=" * 80)
    print(f"Input records: {len(records)}")
    print(f"Output records: {len(output_records)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
