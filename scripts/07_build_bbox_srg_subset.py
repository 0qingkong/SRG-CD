from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor


MODEL_NAME = "google/owlvit-base-patch32"


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


def center_of_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_bbox_relation(subject_box: list[float], object_box: list[float]) -> dict:
    s_cx, s_cy = center_of_box(subject_box)
    o_cx, o_cy = center_of_box(object_box)

    dx = s_cx - o_cx
    dy = s_cy - o_cy

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx >= abs_dy:
        computed_relation = "right_of" if dx > 0 else "left_of"
        axis = "horizontal"
    else:
        computed_relation = "below" if dy > 0 else "above"
        axis = "vertical"

    return {
        "subject_center": [s_cx, s_cy],
        "object_center": [o_cx, o_cy],
        "dx": dx,
        "dy": dy,
        "dominant_axis": axis,
        "computed_relation": computed_relation,
    }


def normalize_relation_key(relation: str) -> str:
    return relation.strip().lower().replace(" ", "_")


def pick_best_detection(
    detections: list[dict],
    label: str,
    image_width: int,
    image_height: int,
    min_score: float,
    max_area_ratio: float,
) -> dict | None:
    candidates = []

    image_area = float(image_width * image_height)

    for det in detections:
        if det["label"] != label:
            continue

        if det["score"] < min_score:
            continue

        area_ratio = box_area(det["box"]) / image_area
        if area_ratio > max_area_ratio:
            continue

        candidates.append(det)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


def detect_one(
    image_path: Path,
    subject: str,
    obj: str,
    processor: OwlViTProcessor,
    model: OwlViTForObjectDetection,
    device: str,
    threshold: float,
) -> tuple[list[dict], tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    texts = [[subject, obj]]

    inputs = processor(text=texts, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )

    result = results[0]

    detections = []
    for box, score, label_idx in zip(
        result["boxes"],
        result["scores"],
        result["labels"],
    ):
        label = texts[0][int(label_idx)]
        detections.append(
            {
                "label": label,
                "score": float(score.detach().cpu()),
                "box": [float(x) for x in box.detach().cpu().tolist()],
            }
        )

    detections.sort(key=lambda x: x["score"], reverse=True)

    return detections, (width, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/srg_bench_v01/srg_bench_v01.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/srg_bench_v01/bbox_srg_bench_v01_subset.jsonl",
    )
    parser.add_argument(
        "--failed_output",
        type=str,
        default="data/srg_bench_v01/bbox_srg_failed_subset.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min_score", type=float, default=0.05)
    parser.add_argument("--max_area_ratio", type=float, default=0.85)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Build BBox-SRG subset with OWL-ViT")
    print("=" * 80)
    print("device:", device)
    print("model:", MODEL_NAME)
    print("input:", args.input)
    print("max_samples:", args.max_samples)
    print("threshold:", args.threshold)
    print("min_score:", args.min_score)
    print("max_area_ratio:", args.max_area_ratio)

    records = read_jsonl(Path(args.input))
    records = records[: args.max_samples]

    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    output_records = []
    failed_records = []

    for item in tqdm(records, desc="Building BBox-SRG"):
        image_path = Path(item["image_path"])
        subject = item["subject"]
        obj = item["object"]

        if not image_path.exists():
            failed = dict(item)
            failed["bbox_error"] = "image_not_found"
            failed_records.append(failed)
            continue

        try:
            detections, (width, height) = detect_one(
                image_path=image_path,
                subject=subject,
                obj=obj,
                processor=processor,
                model=model,
                device=device,
                threshold=args.threshold,
            )

            subject_det = pick_best_detection(
                detections=detections,
                label=subject,
                image_width=width,
                image_height=height,
                min_score=args.min_score,
                max_area_ratio=args.max_area_ratio,
            )

            object_det = pick_best_detection(
                detections=detections,
                label=obj,
                image_width=width,
                image_height=height,
                min_score=args.min_score,
                max_area_ratio=args.max_area_ratio,
            )

            if subject_det is None or object_det is None:
                failed = dict(item)
                failed["bbox_error"] = "missing_subject_or_object"
                failed["all_detections"] = detections
                failed["has_subject"] = subject_det is not None
                failed["has_object"] = object_det is not None
                failed_records.append(failed)
                continue

            bbox_relation = compute_bbox_relation(
                subject_box=subject_det["box"],
                object_box=object_det["box"],
            )

            caption_relation = normalize_relation_key(item["relation"])

            bbox_srg = {
                "nodes": [
                    {
                        "id": "subject",
                        "name": subject,
                        "role": "subject",
                        "bbox": subject_det["box"],
                        "score": subject_det["score"],
                    },
                    {
                        "id": "object",
                        "name": obj,
                        "role": "object",
                        "bbox": object_det["box"],
                        "score": object_det["score"],
                    },
                ],
                "edges": [
                    {
                        "source": "subject",
                        "target": "object",
                        "relation": bbox_relation["computed_relation"],
                        "evidence_type": "bbox_level",
                        "detector": MODEL_NAME,
                    }
                ],
            }

            new_item = dict(item)
            new_item["bbox_evidence"] = {
                "image_size": [width, height],
                "subject_detection": subject_det,
                "object_detection": object_det,
                "spatial_measurement": bbox_relation,
                "caption_relation": caption_relation,
                "bbox_relation": bbox_relation["computed_relation"],
                "caption_bbox_relation_match": caption_relation
                == bbox_relation["computed_relation"],
            }
            new_item["bbox_srg"] = bbox_srg
            new_item["all_detections"] = detections

            output_records.append(new_item)

        except Exception as e:
            failed = dict(item)
            failed["bbox_error"] = repr(e)
            failed_records.append(failed)

    write_jsonl(Path(args.output), output_records)
    write_jsonl(Path(args.failed_output), failed_records)

    print("\n" + "=" * 80)
    print("BBox-SRG subset summary")
    print("=" * 80)
    print(f"Input records: {len(records)}")
    print(f"Success records: {len(output_records)}")
    print(f"Failed records: {len(failed_records)}")
    print(f"Saved output to: {args.output}")
    print(f"Saved failed records to: {args.failed_output}")


if __name__ == "__main__":
    main()
