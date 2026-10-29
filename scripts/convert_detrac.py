"""Convert UA-DETRAC (XML + images) to YOLO format with our 4 classes.

Usage:
    python scripts/convert_detrac.py --xml-dir RAW/annotations --images-dir RAW/images \
        --out-dir OUT/ua_detrac --step 10
"""

import argparse
import json
import sys
import time

from vehicle_counter.data.classes import class_ids, load_config, source_mapping
from vehicle_counter.data.detrac import convert_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-dir", required=True, help="folder with the sequence XML files")
    p.add_argument("--images-dir", required=True, help="folder that contains MVI_* folders")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--step", type=int, default=10, help="keep 1 frame out of N")
    p.add_argument("--config", default=None, help="classes.yaml (default: configs/classes.yaml)")
    args = p.parse_args()

    config = load_config(args.config) if args.config else load_config()
    mapping = source_mapping(config, "ua_detrac")
    names = {i: n for n, i in class_ids(config).items()}
    print("Mapping:", config["sources"]["ua_detrac"]["mapping"])

    start = time.time()
    summary = convert_dataset(args.xml_dir, args.images_dir, args.out_dir,
                              mapping, names, step=args.step)
    summary["seconds"] = round(time.time() - start, 1)
    print(json.dumps(summary, indent=2))

    if summary["images"] == 0:
        print("ERROR: no image converted. Check the paths.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
