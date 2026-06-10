from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2


SUCCESS_PATH = Path("data/srg_bench_v01/bbox_srg_bench_v01_subset.jsonl")
FAILED_PATH = Path("data/srg_bench_v01/bbox_srg_failed_subset.jsonl")

OUTPUT_DIR = Path("results/cases/bbox_srg_visualizations")
TABLE_DIR = Path("results/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


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


def normalize_relation(relation: str) -> str:
    key = relation.strip().lower().replace(" ", "_")
    return CANONICAL_RELATION_MAP.get(key, key)


def draw_one_case(item: dict, output_path: Path) -> bool:
    image_path = Path(item["image_path"])
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    bbox_evidence = item["bbox_evidence"]
    subject_det = bbox_evidence["subject_detection"]
    object_det = bbox_evidence["object_detection"]

    subject = item["subject"]
    obj = item["object"]

    # subject box
    sx1, sy1, sx2, sy2 = [int(v) for v in subject_det["box"]]
    cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"S: {subject} {subject_det['score']:.2f}",
        (sx1, max(sy1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    # object box
    ox1, oy1, ox2, oy2 = [int(v) for v in object_det["box"]]
    cv2.rectangle(image, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"O: {obj} {object_det['score']:.2f}",
        (ox1, max(oy1 - 8, 38)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    # centers
    s_cx, s_cy = bbox_evidence["spatial_measurement"]["subject_center"]
    o_cx, o_cy = bbox_evidence["spatial_measurement"]["object_center"]

    s_center = (int(s_cx), int(s_cy))
    o_center = (int(o_cx), int(o_cy))

    cv2.circle(image, s_center, 5, (0, 255, 0), -1)
    cv2.circle(image, o_center, 5, (255, 0, 0), -1)
    cv2.line(image, s_center, o_center, (0, 255, 255), 2)

    caption_rel = normalize_relation(item["relation"])
    bbox_rel = normalize_relation(bbox_evidence["bbox_relation"])

    label = bool(item["label"])
    matched = caption_rel == bbox_rel
    diagnostic_correct = matched if label else not matched

    title_1 = f"caption: {item['caption']}"
    title_2 = (
        f"label={label} | caption_rel={caption_rel} | "
        f"bbox_rel={bbox_rel} | diag_correct={diagnostic_correct}"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        image,
        title_1[:95],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        title_2[:95],
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    cv2.imwrite(str(output_path), image)
    return True


def main() -> None:
    success_records = read_jsonl(SUCCESS_PATH)
    failed_records = read_jsonl(FAILED_PATH)

    success_csv = TABLE_DIR / "bbox_srg_success_cases.csv"
    failed_csv = TABLE_DIR / "bbox_srg_failed_cases.csv"
    summary_csv = TABLE_DIR / "bbox_srg_summary.csv"

    success_rows = []
    failed_rows = []

    diagnostic_correct_count = 0
    directional_count = 0

    visualized = 0
    max_visualize = 20

    for item in success_records:
        bbox_evidence = item["bbox_evidence"]

        caption_rel = normalize_relation(item["relation"])
        bbox_rel = normalize_relation(bbox_evidence["bbox_relation"])

        is_directional = (
            caption_rel in DIRECTIONAL_RELATIONS
            and bbox_rel in DIRECTIONAL_RELATIONS
        )

        label = bool(item["label"])
        matched = caption_rel == bbox_rel
        diagnostic_correct = matched if label else not matched

        if is_directional:
            directional_count += 1
            diagnostic_correct_count += int(diagnostic_correct)

        output_img = ""

        if visualized < max_visualize and is_directional:
            output_path = OUTPUT_DIR / f"{item['id']}_bbox_srg.jpg"
            ok = draw_one_case(item, output_path)
            if ok:
                output_img = str(output_path)
                visualized += 1

        success_rows.append(
            {
                "id": item["id"],
                "image": item["image"],
                "caption": item["caption"],
                "label": str(label),
                "subject": item["subject"],
                "object": item["object"],
                "relation": item["relation"],
                "caption_relation_canonical": caption_rel,
                "bbox_relation": bbox_rel,
                "relation_group": item.get("relation_group", ""),
                "is_directional": str(is_directional),
                "caption_bbox_match": str(matched),
                "diagnostic_correct": str(diagnostic_correct),
                "subject_score": bbox_evidence["subject_detection"]["score"],
                "object_score": bbox_evidence["object_detection"]["score"],
                "dx": bbox_evidence["spatial_measurement"]["dx"],
                "dy": bbox_evidence["spatial_measurement"]["dy"],
                "visualization": output_img,
            }
        )

    for item in failed_records:
        failed_rows.append(
            {
                "id": item["id"],
                "image": item.get("image", ""),
                "caption": item.get("caption", ""),
                "label": str(item.get("label", "")),
                "subject": item.get("subject", ""),
                "object": item.get("object", ""),
                "relation": item.get("relation", ""),
                "relation_group": item.get("relation_group", ""),
                "bbox_error": item.get("bbox_error", ""),
                "has_subject": str(item.get("has_subject", "")),
                "has_object": str(item.get("has_object", "")),
            }
        )

    with success_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(success_rows[0].keys()))
        writer.writeheader()
        writer.writerows(success_rows)

    with failed_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(failed_rows)

    total = len(success_records) + len(failed_records)
    detection_success_rate = len(success_records) / total if total else 0.0
    diagnostic_rate = (
        diagnostic_correct_count / directional_count if directional_count else 0.0
    )

    summary_rows = [
        {
            "metric": "total_records",
            "value": total,
        },
        {
            "metric": "success_records",
            "value": len(success_records),
        },
        {
            "metric": "failed_records",
            "value": len(failed_records),
        },
        {
            "metric": "detection_success_rate",
            "value": detection_success_rate,
        },
        {
            "metric": "directional_success_records",
            "value": directional_count,
        },
        {
            "metric": "diagnostic_correct_records",
            "value": diagnostic_correct_count,
        },
        {
            "metric": "diagnostic_consistency_rate",
            "value": diagnostic_rate,
        },
        {
            "metric": "visualized_cases",
            "value": visualized,
        },
    ]

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 80)
    print("BBox-SRG export finished")
    print("=" * 80)
    print(f"Success CSV: {success_csv}")
    print(f"Failed CSV: {failed_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Visualization dir: {OUTPUT_DIR}")
    print(f"Visualized cases: {visualized}")


if __name__ == "__main__":
    main()
