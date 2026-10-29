"""Tests for the UA-DETRAC -> YOLO conversion."""

import csv

import cv2
import numpy as np
import pytest

from vehicle_counter.data.classes import class_ids, load_config, source_mapping
from vehicle_counter.data.detrac import (MASK_COLOR, Box, Frame, convert_dataset,
                                         convert_frame, overlap_ratio, parse_sequence)

W, H = 960, 540

XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_00001">
  <sequence_attribute camera_state="stable" sence_weather="night"/>
  <ignored_region>
    <box left="0" top="0" width="100" height="100"/>
  </ignored_region>
  <frame density="3" num="1">
    <target_list>
      <target id="1">
        <box left="200" top="100" width="96" height="54"/>
        <attribute vehicle_type="car"/>
      </target>
      <target id="2">
        <box left="10" top="10" width="50" height="50"/>
        <attribute vehicle_type="bus"/>
      </target>
      <target id="3">
        <box left="900" top="500" width="100" height="80"/>
        <attribute vehicle_type="others"/>
      </target>
    </target_list>
  </frame>
  <frame density="1" num="2">
    <target_list>
      <target id="1">
        <box left="210" top="100" width="96" height="54"/>
        <attribute vehicle_type="car"/>
      </target>
    </target_list>
  </frame>
</sequence>
"""

MAPPING = {"car": 0, "others": 1, "bus": 2, "van": 3}


@pytest.fixture
def raw(tmp_path):
    xml_dir = tmp_path / "xml"
    img_dir = tmp_path / "images" / "MVI_00001"
    xml_dir.mkdir()
    img_dir.mkdir(parents=True)
    (xml_dir / "MVI_00001.xml").write_text(XML, encoding="utf-8")
    for num in (1, 2):
        cv2.imwrite(str(img_dir / f"img{num:05d}.jpg"), np.zeros((H, W, 3), np.uint8))
    return tmp_path


def test_parse_sequence(raw):
    seq = parse_sequence(raw / "xml" / "MVI_00001.xml")
    assert seq.name == "MVI_00001"
    assert seq.weather == "night"
    assert len(seq.ignored) == 1
    assert [f.num for f in seq.frames] == [1, 2]
    assert [b.label for b in seq.frames[0].boxes] == ["car", "bus", "others"]


def test_convert_frame_values_and_rules(raw):
    seq = parse_sequence(raw / "xml" / "MVI_00001.xml")
    lines, _, counts = convert_frame(seq.frames[0], seq.ignored, MAPPING, W, H)
    # car: center (248, 127) -> (0.258333, 0.235185), size (0.1, 0.1)
    assert lines[0] == "0 0.258333 0.235185 0.100000 0.100000"
    # bus is inside the ignored region -> dropped
    assert all(not l.startswith("2 ") for l in lines)
    # "others" is clipped to the image: x 900-960, y 500-540
    cid, xc, yc, bw, bh = lines[1].split()
    assert cid == "1"
    assert float(bw) == pytest.approx(60 / W, abs=1e-6)
    assert float(bh) == pytest.approx(40 / H, abs=1e-6)
    assert counts == {0: 1, 1: 1}


def test_unmapped_label_is_masked():
    frame = Frame(1, [Box(200, 100, 50, 50, "others")])
    lines, to_mask, counts = convert_frame(frame, [], {"car": 0, "others": None}, W, H)
    assert lines == [] and counts == {}
    assert len(to_mask) == 1


def test_overlap_ratio():
    assert overlap_ratio(Box(0, 0, 10, 10), [Box(5, 0, 10, 10)]) == pytest.approx(0.5)
    assert overlap_ratio(Box(0, 0, 10, 10), []) == 0.0


def test_convert_dataset_end_to_end(raw):
    out = raw / "out"
    names = {0: "car", 1: "truck", 2: "bus", 3: "van"}
    summary = convert_dataset(raw / "xml", raw / "images", out, MAPPING, names, step=10)

    # step=10 keeps frame 1 only
    assert summary["images"] == 1
    assert summary["boxes"] == {"car": 1, "truck": 1, "bus": 0, "van": 0}
    assert (out / "labels" / "detrac_MVI_00001_00001.txt").exists()
    assert not (out / "labels" / "detrac_MVI_00001_00002.txt").exists()

    # ignored region is gray in the saved image (JPEG: allow small error)
    img = cv2.imread(str(out / "images" / "detrac_MVI_00001_00001.jpg"))
    assert np.abs(img[50, 50].astype(int) - MASK_COLOR[0]).max() <= 3

    with open(out / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["weather"] == "night" and rows[0]["n_car"] == "1"


def test_project_config_mapping():
    config = load_config()
    assert class_ids(config) == {"car": 0, "truck": 1, "bus": 2, "van": 3}
    mapping = source_mapping(config, "ua_detrac")
    assert set(mapping) == {"car", "bus", "van", "others"}


def test_duplicate_sequence_is_skipped(raw):
    (raw / "xml" / "copy").mkdir()
    (raw / "xml" / "copy" / "MVI_00001.xml").write_text(XML, encoding="utf-8")
    names = {0: "car", 1: "truck", 2: "bus", 3: "van"}
    summary = convert_dataset(raw / "xml", raw / "images", raw / "out", MAPPING, names, step=10)
    assert summary["images"] == 1
    assert len(summary["duplicate_xml"]) == 1
