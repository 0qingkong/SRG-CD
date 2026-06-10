from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm


COCO_URL_TEMPLATES = [
    "https://images.cocodataset.org/val2017/{image}",
    "https://images.cocodataset.org/train2017/{image}",
    "https://images.cocodataset.org/test2017/{image}",
]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def download_one(image: str, output_path: Path, timeout: int = 20) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    for template in COCO_URL_TEMPLATES:
        url = template.format(image=image)

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    continue

                data = response.read()
                if len(data) < 100:
                    continue

                output_path.write_bytes(data)
                return True

        except Exception:
            continue

    return False


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
        default="data/raw/vsr/images",
    )
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))

    image_names = []
    seen = set()

    for item in records:
        image = item.get("image")
        if not image:
            continue
        if image in seen:
            continue
        seen.add(image)
        image_names.append(image)

    if args.max_images is not None and args.max_images > 0:
        image_names = image_names[: args.max_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for image in tqdm(image_names, desc="Downloading VSR images"):
        output_path = output_dir / image

        ok = download_one(image, output_path)
        if ok:
            success += 1
        else:
            failed.append(image)

        if args.sleep > 0:
            time.sleep(args.sleep)

    failed_path = output_dir.parent / "download_failed_images.txt"
    failed_path.write_text("\n".join(failed), encoding="utf-8")

    print("=" * 80)
    print("VSR image download summary")
    print("=" * 80)
    print(f"Target images: {len(image_names)}")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Image dir: {output_dir}")
    print(f"Failed list: {failed_path}")


if __name__ == "__main__":
    main()
