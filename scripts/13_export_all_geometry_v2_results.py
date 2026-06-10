from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


SUCCESS_PATH = Path("data/srg_bench_v01/bbox_srg_bench_v01_all_v2.jsonl")
FAILED_PATH = Path("data/srg_bench_v01/bbox_srg_failed_all.jsonl")

TABLE_DIR = Path("results/tables")
VIS_DIR = Path("results/cases/all_geometry_v2_visualizations")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


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


def draw_case(item: dict, output_path: Path) -> bool:
    image_path = Path(item["image_path"])
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    ev = item["bbox_evidence_v2"]
    nodes = item["bbox_srg_v2"]["nodes"]

    subject_node = nodes[0]
    object_node = nodes[1]

    subject = subject_node["name"]
    obj = object_node["name"]

    sx1, sy1, sx2, sy2 = [int(v) for v in subject_node["bbox"]]
    ox1, oy1, ox2, oy2 = [int(v) for v in object_node["bbox"]]

    # subject: green
    cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"S: {subject} {subject_node['score']:.2f}",
        (sx1, max(sy1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    # object: blue
    cv2.rectangle(image, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"O: {obj} {object_node['score']:.2f}",
        (ox1, max(oy1 - 8, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    gf = ev["geometry_features"]

    s_cx, s_cy = gf["subject_center"]
    o_cx, o_cy = gf["object_center"]

    s_center = (int(s_cx), int(s_cy))
    o_center = (int(o_cx), int(o_cy))

    cv2.circle(image, s_center, 5, (0, 255, 0), -1)
    cv2.circle(image, o_center, 5, (255, 0, 0), -1)
    cv2.line(image, s_center, o_center, (0, 255, 255), 2)

    label = bool(item["label"])
    diag = ev["diagnostic_correct_v2"]

    title_1 = f"caption: {item['caption']}"
    title_2 = (
        f"label={label} | caption={ev['caption_relation_canonical']} | "
        f"bbox={ev['bbox_relation_v2']} | family={ev['relation_family_v2']} | "
        f"diag={diag}"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(
        image,
        title_1[:100],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        title_2[:100],
        (8, 50),
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

    total = len(success_records) + len(failed_records)

    diagnostic_counter = Counter()
    label_stats = defaultdict(lambda: Counter())
    group_stats = defaultdict(lambda: Counter())
    family_stats = defaultdict(lambda: Counter())
    relation_counter = Counter()
    failure_counter = Counter()

    success_rows = []
    failed_rows = []

    # 每类抽一些可视化案例
    visualized = 0
    max_visualize = 60
    family_visual_count = Counter()

    for item in success_records:
        ev = item["bbox_evidence_v2"]
        label = bool(item["label"])
        diag = ev["diagnostic_correct_v2"]
        group = item.get("relation_group", "unknown")
        family = ev["relation_family_v2"]
        bbox_relation = ev["bbox_relation_v2"]

        diagnostic_counter[str(diag)] += 1
        label_stats[str(label)]["total"] += 1
        label_stats[str(label)]["correct"] += int(bool(diag))
        group_stats[group]["total"] += 1
        group_stats[group]["correct"] += int(bool(diag))
        family_stats[family]["total"] += 1
        family_stats[family]["correct"] += int(bool(diag))
        relation_counter[bbox_relation] += 1

        vis_path = ""

        if visualized < max_visualize and family_visual_count[family] < 15:
            out = VIS_DIR / f"{item['id']}_geometry_v2.jpg"
            if draw_case(item, out):
                vis_path = str(out)
                visualized += 1
                family_visual_count[family] += 1

        gf = ev["geometry_features"]

        success_rows.append(
            {
                "id": item["id"],
                "image": item["image"],
                "caption": item["caption"],
                "label": str(label),
                "subject": item["subject"],
                "object": item["object"],
                "caption_relation": item["relation"],
                "caption_relation_canonical": ev["caption_relation_canonical"],
                "bbox_relation_v2": ev["bbox_relation_v2"],
                "relation_group": group,
                "relation_family_v2": family,
                "diagnostic_correct_v2": str(diag),
                "subject_score": item["bbox_srg_v2"]["nodes"][0]["score"],
                "object_score": item["bbox_srg_v2"]["nodes"][1]["score"],
                "dx": gf["dx"],
                "dy": gf["dy"],
                "iou": gf["iou"],
                "subject_in_object_ratio": gf["subject_in_object_ratio"],
                "object_in_subject_ratio": gf["object_in_subject_ratio"],
                "normalized_center_distance": gf["normalized_center_distance"],
                "x_overlap_ratio": gf["x_overlap_ratio"],
                "y_overlap_ratio": gf["y_overlap_ratio"],
                "visualization": vis_path,
            }
        )

    for item in failed_records:
        failure_counter[item.get("bbox_error", "unknown")] += 1

        failed_rows.append(
            {
                "id": item.get("id", ""),
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

    summary_rows = [
        {"metric": "total_records", "value": total},
        {"metric": "success_records", "value": len(success_records)},
        {"metric": "failed_records", "value": len(failed_records)},
        {
            "metric": "detection_success_rate",
            "value": safe_div(len(success_records), total),
        },
        {
            "metric": "diagnostic_correct_records",
            "value": diagnostic_counter.get("True", 0),
        },
        {
            "metric": "diagnostic_incorrect_records",
            "value": diagnostic_counter.get("False", 0),
        },
        {
            "metric": "diagnostic_consistency_rate",
            "value": safe_div(diagnostic_counter.get("True", 0), len(success_records)),
        },
        {"metric": "visualized_cases", "value": visualized},
    ]

    group_rows = []

    for group, counter in sorted(group_stats.items()):
        group_rows.append(
            {
                "type": "relation_group",
                "name": group,
                "total": counter["total"],
                "correct": counter["correct"],
                "rate": safe_div(counter["correct"], counter["total"]),
            }
        )

    for family, counter in sorted(family_stats.items()):
        group_rows.append(
            {
                "type": "relation_family_v2",
                "name": family,
                "total": counter["total"],
                "correct": counter["correct"],
                "rate": safe_div(counter["correct"], counter["total"]),
            }
        )

    for label, counter in sorted(label_stats.items()):
        group_rows.append(
            {
                "type": "label",
                "name": label,
                "total": counter["total"],
                "correct": counter["correct"],
                "rate": safe_div(counter["correct"], counter["total"]),
            }
        )

    for relation, count in relation_counter.most_common():
        group_rows.append(
            {
                "type": "bbox_relation_v2",
                "name": relation,
                "total": count,
                "correct": "",
                "rate": "",
            }
        )

    for reason, count in failure_counter.most_common():
        group_rows.append(
            {
                "type": "failure_reason",
                "name": reason,
                "total": count,
                "correct": "",
                "rate": "",
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(TABLE_DIR / "all_geometry_v2_summary.csv", summary_rows)
    write_csv(TABLE_DIR / "all_geometry_v2_success_cases.csv", success_rows)
    write_csv(TABLE_DIR / "all_geometry_v2_failed_cases.csv", failed_rows)
    write_csv(TABLE_DIR / "all_geometry_v2_group_metrics.csv", group_rows)

    print("=" * 80)
    print("Export all Geometry v2 results finished")
    print("=" * 80)
    print(f"Summary: {TABLE_DIR / 'all_geometry_v2_summary.csv'}")
    print(f"Success cases: {TABLE_DIR / 'all_geometry_v2_success_cases.csv'}")
    print(f"Failed cases: {TABLE_DIR / 'all_geometry_v2_failed_cases.csv'}")
    print(f"Group metrics: {TABLE_DIR / 'all_geometry_v2_group_metrics.csv'}")
    print(f"Visualization dir: {VIS_DIR}")
    print(f"Visualized cases: {visualized}")


if __name__ == "__main__":
    main()
