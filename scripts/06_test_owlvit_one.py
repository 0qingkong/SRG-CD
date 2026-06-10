from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


BENCH_PATH = Path("data/srg_bench_v01/srg_bench_v01.jsonl")
OUTPUT_DIR = Path("results/cases/owlvit_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/owlvit-base-patch32"


def read_first_valid_record() -> dict:
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            image_path = Path(item["image_path"])
            if image_path.exists():
                return item

    raise RuntimeError("No valid record with existing image found.")


def draw_boxes(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["label"]
        score = det["score"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_path), image)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    record = read_first_valid_record()

    image_path = Path(record["image_path"])
    subject = record["subject"]
    obj = record["object"]

    print("=" * 80)
    print("OWL-ViT one-image test")
    print("=" * 80)
    print("device:", device)
    print("id:", record["id"])
    print("image:", image_path)
    print("caption:", record["caption"])
    print("subject:", subject)
    print("relation:", record["relation"])
    print("object:", obj)

    image = Image.open(image_path).convert("RGB")

    # OWL-ViT 的文本输入是候选类别列表
    texts = [[subject, obj]]

    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    inputs = processor(text=texts, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.10,
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

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    print("\nDetection results:")
    for i, det in enumerate(detections):
        print(
            f"[{i}] label={det['label']}, "
            f"score={det['score']:.4f}, "
            f"box={det['box']}"
        )

    output_img = OUTPUT_DIR / f"{record['id']}_owlvit.jpg"
    output_json = OUTPUT_DIR / f"{record['id']}_owlvit.json"

    draw_boxes(image_path, detections, output_img)

    payload = {
        "id": record["id"],
        "image": record["image"],
        "caption": record["caption"],
        "subject": subject,
        "relation": record["relation"],
        "object": obj,
        "model": MODEL_NAME,
        "detections": detections,
        "annotated_image": str(output_img),
    }

    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved annotated image to:", output_img)
    print("Saved detection json to:", output_json)


if __name__ == "__main__":
    main()
