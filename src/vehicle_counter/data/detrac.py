"""Read UA-DETRAC XML annotations and convert them to YOLO format.

UA-DETRAC XML structure (one file per sequence):

    <sequence name="MVI_20011">
      <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
      <ignored_region>
        <box left=".." top=".." width=".." height=".."/>
      </ignored_region>
      <frame density="7" num="1">
        <target_list>
          <target id="1">
            <box left=".." top=".." width=".." height=".."/>
            <attribute vehicle_type="car" .../>
          </target>
        </target_list>
      </frame>
    </sequence>
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MASK_COLOR = (114, 114, 114)   # gray, same value YOLO uses for padding
MAX_IGNORED_OVERLAP = 0.5      # drop a box if > 50% of it is in an ignored region
MIN_BOX_SIZE = 2.0             # pixels, after clipping


@dataclass
class Box:
    left: float
    top: float
    width: float
    height: float
    label: str = ""

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass
class Frame:
    num: int
    boxes: list[Box] = field(default_factory=list)


@dataclass
class Sequence:
    name: str
    weather: str
    camera_state: str
    ignored: list[Box]
    frames: list[Frame]


def _box(el: ET.Element, label: str = "") -> Box:
    return Box(
        float(el.get("left")), float(el.get("top")),
        float(el.get("width")), float(el.get("height")), label,
    )


def parse_sequence(xml_path: str | Path) -> Sequence:
    root = ET.parse(xml_path).getroot()
    attr = root.find("sequence_attribute")
    weather = attr.get("sence_weather", attr.get("scene_weather", "")) if attr is not None else ""
    camera = attr.get("camera_state", "") if attr is not None else ""

    ignored = [_box(b) for b in root.findall("./ignored_region/box")]

    frames = []
    for fr in root.findall("frame"):
        boxes = []
        for tg in fr.findall("./target_list/target"):
            b = tg.find("box")
            a = tg.find("attribute")
            label = a.get("vehicle_type", "") if a is not None else ""
            boxes.append(_box(b, label))
        frames.append(Frame(int(fr.get("num")), boxes))

    return Sequence(root.get("name"), weather, camera, ignored, frames)


def clip_box(b: Box, img_w: int, img_h: int) -> Box:
    x1, y1 = max(b.left, 0.0), max(b.top, 0.0)
    x2, y2 = min(b.left + b.width, img_w), min(b.top + b.height, img_h)
    return Box(x1, y1, x2 - x1, y2 - y1, b.label)


def overlap_ratio(b: Box, regions: list[Box]) -> float:
    """Fraction of box b covered by the regions (regions assumed not to overlap)."""
    if b.area <= 0:
        return 0.0
    inter = 0.0
    for r in regions:
        w = min(b.left + b.width, r.left + r.width) - max(b.left, r.left)
        h = min(b.top + b.height, r.top + r.height) - max(b.top, r.top)
        if w > 0 and h > 0:
            inter += w * h
    return min(inter / b.area, 1.0)


def mask_regions(image: np.ndarray, regions: list[Box]) -> None:
    """Fill regions with gray, in place."""
    h, w = image.shape[:2]
    for r in regions:
        c = clip_box(r, w, h)
        x1, y1 = int(c.left), int(c.top)
        x2, y2 = int(np.ceil(c.left + c.width)), int(np.ceil(c.top + c.height))
        image[y1:y2, x1:x2] = MASK_COLOR


def to_yolo_line(b: Box, class_id: int, img_w: int, img_h: int) -> str:
    xc = (b.left + b.width / 2) / img_w
    yc = (b.top + b.height / 2) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {b.width / img_w:.6f} {b.height / img_h:.6f}"


def convert_frame(
    frame: Frame,
    ignored: list[Box],
    mapping: dict[str, int | None],
    img_w: int,
    img_h: int,
) -> tuple[list[str], list[Box], dict[int, int]]:
    """Convert one frame.

    Returns (yolo_lines, regions_to_mask, counts_per_class_id).
    Boxes whose label maps to None, or that are not in the mapping,
    are masked in gray (so the model is not trained on unlabeled vehicles).
    """
    lines: list[str] = []
    to_mask: list[Box] = list(ignored)
    counts: dict[int, int] = {}

    for raw in frame.boxes:
        b = clip_box(raw, img_w, img_h)
        if b.width < MIN_BOX_SIZE or b.height < MIN_BOX_SIZE:
            continue
        if overlap_ratio(b, ignored) > MAX_IGNORED_OVERLAP:
            continue
        class_id = mapping.get(b.label)
        if class_id is None:
            to_mask.append(b)
            continue
        lines.append(to_yolo_line(b, class_id, img_w, img_h))
        counts[class_id] = counts.get(class_id, 0) + 1

    return lines, to_mask, counts


def find_sequence_dirs(images_root: str | Path) -> dict[str, Path]:
    """Map sequence name (e.g. MVI_20011) -> folder with its images."""
    return {p.name: p for p in Path(images_root).rglob("MVI_*") if p.is_dir()}


def frame_image_path(seq_dir: Path, num: int) -> Path:
    return seq_dir / f"img{num:05d}.jpg"


METADATA_FIELDS = ["image", "source", "sequence", "frame", "weather", "camera_state",
                   "n_boxes"]


def convert_dataset(
    xml_dir: str | Path,
    images_root: str | Path,
    out_dir: str | Path,
    mapping: dict[str, int | None],
    class_names: dict[int, str],
    step: int = 10,
) -> dict:
    """Convert all sequences. Keeps frames where (num - 1) % step == 0.

    Writes out_dir/images/*.jpg, out_dir/labels/*.txt and out_dir/metadata.csv.
    Returns a summary dict.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    seq_dirs = find_sequence_dirs(images_root)
    xml_files = sorted(Path(xml_dir).rglob("*.xml"))

    summary = {"sequences": 0, "images": 0, "missing_sequences": [], "missing_images": 0,
               "duplicate_xml": [], "unknown_labels": set(),
               "boxes": {name: 0 for name in class_names.values()}}
    seen = set()

    rows = []
    for xml_path in xml_files:
        seq = parse_sequence(xml_path)
        if seq.name in seen:  # same sequence annotated twice (e.g. 2 versions)
            summary["duplicate_xml"].append(str(xml_path))
            continue
        seen.add(seq.name)
        seq_dir = seq_dirs.get(seq.name)
        if seq_dir is None:
            summary["missing_sequences"].append(seq.name)
            continue
        summary["sequences"] += 1

        for frame in seq.frames:
            if (frame.num - 1) % step != 0:
                continue
            img_path = frame_image_path(seq_dir, frame.num)
            image = cv2.imread(str(img_path))
            if image is None:
                summary["missing_images"] += 1
                continue
            h, w = image.shape[:2]

            for b in frame.boxes:
                if b.label not in mapping:
                    summary["unknown_labels"].add(b.label)

            lines, to_mask, counts = convert_frame(frame, seq.ignored, mapping, w, h)
            mask_regions(image, to_mask)

            stem = f"detrac_{seq.name}_{frame.num:05d}"
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            row = {"image": f"{stem}.jpg", "source": "ua_detrac", "sequence": seq.name,
                   "frame": frame.num, "weather": seq.weather,
                   "camera_state": seq.camera_state, "n_boxes": len(lines)}
            for cid, name in class_names.items():
                row[f"n_{name}"] = counts.get(cid, 0)
                summary["boxes"][name] += counts.get(cid, 0)
            rows.append(row)
            summary["images"] += 1

    fields = METADATA_FIELDS + [f"n_{name}" for name in class_names.values()]
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary["unknown_labels"] = sorted(summary["unknown_labels"])
    return summary
