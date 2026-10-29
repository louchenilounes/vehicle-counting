"""Check a YOLO dataset and print statistics.

- every image has a label file, every label has valid values in [0, 1]
- number of images and boxes per class (and per weather if metadata.csv exists)
- saves a grid of N random images with their boxes drawn (visual check)

Usage:
    python scripts/dataset_stats.py --data-dir OUT/ua_detrac --n-preview 20 \
        --preview outputs/preview_ua_detrac.jpg
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.data.classes import class_ids, load_config

COLORS = [(0, 200, 0), (0, 0, 255), (255, 120, 0), (0, 200, 255)]  # BGR per class
PREVIEW_W = 480


def read_labels(path: Path, n_classes: int) -> tuple[list[tuple], list[str]]:
    boxes, errors = [], []
    for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path.name}:{k} expected 5 values")
            continue
        cid, *vals = parts
        cid, vals = int(cid), [float(v) for v in vals]
        if not 0 <= cid < n_classes:
            errors.append(f"{path.name}:{k} bad class id {cid}")
        if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            errors.append(f"{path.name}:{k} values out of [0, 1]")
        boxes.append((cid, *vals))
    return boxes, errors


def draw(image: np.ndarray, boxes: list[tuple], names: dict[int, str]) -> np.ndarray:
    h, w = image.shape[:2]
    for cid, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, names[cid], (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    return image


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="folder with images/ labels/")
    p.add_argument("--n-preview", type=int, default=20)
    p.add_argument("--preview", default="outputs/preview.jpg")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = {i: n for n, i in class_ids(load_config()).items()}
    data = Path(args.data_dir)
    images = sorted((data / "images").glob("*.jpg"))
    labels_dir = data / "labels"

    missing, errors = [], []
    boxes_per_class, images_per_class = Counter(), Counter()
    empty_images = 0
    all_boxes = {}
    for img in images:
        lab = labels_dir / f"{img.stem}.txt"
        if not lab.exists():
            missing.append(img.name)
            continue
        boxes, errs = read_labels(lab, len(names))
        errors += errs
        all_boxes[img] = boxes
        if not boxes:
            empty_images += 1
        classes = Counter(b[0] for b in boxes)
        boxes_per_class.update(classes)
        images_per_class.update(classes.keys())

    total = sum(boxes_per_class.values())
    report = {
        "images": len(images),
        "images_without_label_file": len(missing),
        "images_with_no_box": empty_images,
        "label_errors": len(errors),
        "boxes_total": total,
        "per_class": {
            names[c]: {"boxes": boxes_per_class[c],
                       "share_%": round(100 * boxes_per_class[c] / total, 2) if total else 0,
                       "images": images_per_class[c]}
            for c in sorted(names)
        },
    }

    meta = data / "metadata.csv"
    if meta.exists():
        per_weather = defaultdict(lambda: {"images": 0, "sequences": set()})
        with open(meta, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wtr = row["weather"] or "unknown"
                per_weather[wtr]["images"] += 1
                per_weather[wtr]["sequences"].add(row["sequence"])
        report["per_weather"] = {k: {"images": v["images"], "sequences": len(v["sequences"])}
                                 for k, v in sorted(per_weather.items())}

    print(json.dumps(report, indent=2))
    for e in errors[:10]:
        print("LABEL ERROR:", e)
    for m in missing[:10]:
        print("MISSING LABEL:", m)

    # Preview grid
    random.seed(args.seed)
    picks = random.sample(list(all_boxes), min(args.n_preview, len(all_boxes)))
    tiles = []
    for img_path in picks:
        img = cv2.imread(str(img_path))
        img = draw(img, all_boxes[img_path], names)
        scale = PREVIEW_W / img.shape[1]
        tiles.append(cv2.resize(img, (PREVIEW_W, int(img.shape[0] * scale))))
    if tiles:
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, np.vstack(rows))
        print(f"Preview saved: {args.preview}")

    ok = not missing and not errors and len(images) > 0
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
