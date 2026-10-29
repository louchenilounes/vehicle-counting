"""Check the environment and run a quick YOLO test on a sample image.

Usage (from the repo root):
    python scripts/check_env.py
"""

import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
MODEL_NAME = "yolo26n.pt"  # downloaded automatically the first time
N_RUNS = 10                # number of timed runs


def print_versions() -> str:
    """Print library versions and return the device to use."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Environment ===")
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"NumPy       : {np.__version__}")
    if device == "cuda":
        print(f"Device      : GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Device      : CPU")
    return device


def main() -> int:
    device = print_versions()

    image_path = ASSETS / "bus.jpg"  # sample image shipped with Ultralytics
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: cannot read sample image {image_path}")
        return 1

    print(f"\n=== YOLO test ({MODEL_NAME}) ===")
    model = YOLO(MODEL_NAME)

    # Warm-up run (the first run is always slower)
    model.predict(image, device=device, verbose=False)

    times_ms = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        results = model.predict(image, device=device, verbose=False)
        times_ms.append((time.perf_counter() - start) * 1000)

    result = results[0]
    names = [result.names[int(c)] for c in result.boxes.cls]
    counts = Counter(names)

    print(f"Image size  : {image.shape[1]}x{image.shape[0]}")
    print(f"Detections  : {len(names)} -> {dict(counts)}")
    print(f"Time / image: median {statistics.median(times_ms):.1f} ms "
          f"(min {min(times_ms):.1f}, max {max(times_ms):.1f}, {N_RUNS} runs)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "check_env_bus.jpg"
    cv2.imwrite(str(out_path), result.plot())
    print(f"Saved       : {out_path}")

    if len(names) == 0:
        print("\nWARNING: no detection. Something is wrong.")
        return 1

    print("\nOK: environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
