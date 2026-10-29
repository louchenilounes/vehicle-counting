# Step 0 — Project Scope

## 1. Goal

Build a real-time system that **detects, classifies, tracks and counts vehicles** in traffic videos and live streams.

- Detection: Ultralytics YOLO (nano), fine-tuned
- Tracking: ByteTrack (one unique ID per vehicle)
- Counting: virtual line, per class and per direction
- Inputs: video files and live streams (YouTube live via yt-dlp, public HLS/MJPEG traffic cameras)
- Outputs: FastAPI API + Docker, Streamlit demo, clean GitHub repo, demo GIF, results table

## 2. Scope

### Classes

| ID | Class |
|----|-------|
| 0 | car |
| 1 | truck |
| 2 | bus |
| 3 | van |

Motorcycles are **out of scope** (few examples in fixed traffic-camera views).

### Target scenario

- **In scope:** fixed, elevated traffic camera (road or highway), day and night, dry and rainy weather.
- **Out of scope:** dashcam views (BDD100K is used only as extra training data), drones, indoor parking.

### Hardware

- Training: Google Colab or Kaggle (free GPU).
- Inference: CPU (ONNX / OpenVINO export, process 1 frame out of 2 or 3).

## 3. Metrics

| Block | Metrics |
|-------|---------|
| Detection | mAP50, mAP50-95, precision and recall per class, confusion matrix |
| Tracking | number of ID switches on test clips |
| Counting | absolute error and % error per class and per direction, vs manual ground truth |
| Speed | FPS and latency per frame on CPU (PyTorch vs ONNX vs OpenVINO) |
| Robustness | same metrics split by condition: day / night / rain |

**Rule:** every number in this repo comes from a real test. Target thresholds will be set after the baseline (Step 5), based on measured results.

## 4. Data

| Source | Use | License note |
|--------|-----|--------------|
| UA-DETRAC | main training data (fixed traffic cameras) | research use, no redistribution |
| BDD100K | extra data (trucks, night, rain) | research use, no redistribution |
| Roboflow Universe | extra data (trucks, vans) | license checked per dataset |
| Own annotations (200–300 images) | target domain + hard cases | images from free-license sources only |

Datasets are **never** stored in this repo. The repo only contains download and conversion scripts.

## 5. Legal and ethics

- Code license: **AGPL-3.0** (required by Ultralytics).
- Live streams: processed in memory only. No raw video is saved or shared.
- Demo GIF: made from a free-license video (source and license cited in the README).
- No personal data is stored (no license plates, no faces).

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Few "truck" examples in UA-DETRAC (trucks are in "others") | add Roboflow + BDD100K trucks, check class balance |
| Domain gap: BDD100K is a car view, not a fixed camera | evaluate on fixed-camera test set only |
| Few night / rain examples | include them in own annotations |
| Limited GPU quotas | short trainings, save checkpoints to Drive |
| Unstable live streams (cuts, URL changes) | reconnection logic, timeouts |

## 7. Plan

| # | Step |
|---|------|
| 0 | Scope (this document) |
| 1 | Environment and repo |
| 2 | Public data: collection and class mapping |
| 3 | Own annotations |
| 4 | Dataset split and analysis |
| 5 | Baseline (pretrained COCO model) |
| 6 | Fine-tuning |
| 7 | Detection evaluation |
| 8 | Tracking (ByteTrack) |
| 9 | Line counting |
| 10 | Counting evaluation |
| 11 | Robustness test (day / night / rain) |
| 12 | CPU optimization (ONNX / OpenVINO) |
| 13 | Video sources (files + live streams) |
| 14 | FastAPI API |
| 15 | Docker + Streamlit demo |
| 16 | Final deliverables (README, GIF, results table) |
