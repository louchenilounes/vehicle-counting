"""Save a grid of random crops of one UA-DETRAC vehicle type (visual check).

Usage:
    python scripts/sample_class_crops.py --xml-dir RAW/annotations --images-dir RAW/images \
        --label others --n 100 --out outputs/crops_others.jpg
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.data.detrac import (clip_box, find_sequence_dirs, frame_image_path,
                                         parse_sequence)

TILE = 128   # tile size in pixels
COLS = 10


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--label", default="others", help="UA-DETRAC vehicle_type")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-size", type=int, default=40, help="min box side in pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    seq_dirs = find_sequence_dirs(args.images_dir)
    candidates = []        # (seq_dir, frame_num, box)
    per_sequence = {}
    for xml_path in sorted(Path(args.xml_dir).rglob("*.xml")):
        seq = parse_sequence(xml_path)
        if seq.name not in seq_dirs:
            continue
        for fr in seq.frames:
            for b in fr.boxes:
                if b.label == args.label and min(b.width, b.height) >= args.min_size:
                    candidates.append((seq_dirs[seq.name], fr.num, b))
                    per_sequence[seq.name] = per_sequence.get(seq.name, 0) + 1

    print(f"'{args.label}' boxes (side >= {args.min_size}px): {len(candidates)} "
          f"in {len(per_sequence)} sequences")
    if not candidates:
        return 1

    random.seed(args.seed)
    picks = random.sample(candidates, min(args.n, len(candidates)))

    tiles = []
    for seq_dir, num, b in picks:
        img = cv2.imread(str(frame_image_path(seq_dir, num)))
        if img is None:
            continue
        c = clip_box(b, img.shape[1], img.shape[0])
        crop = img[int(c.top):int(c.top + c.height), int(c.left):int(c.left + c.width)]
        tiles.append(cv2.resize(crop, (TILE, TILE)))

    rows = -(-len(tiles) // COLS)
    grid = np.full((rows * TILE, COLS * TILE, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, col = divmod(k, COLS)
        grid[r * TILE:(r + 1) * TILE, col * TILE:(col + 1) * TILE] = t

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Saved {len(tiles)} crops to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
