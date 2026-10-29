# Vehicle Detection, Tracking and Counting

Real-time system that detects, classifies, tracks and counts vehicles (**car, truck, bus, van**) in traffic videos and live streams.

> 🚧 Work in progress. All numbers in this README come from real tests.

## Overview

- **Detection:** Ultralytics YOLO (nano), fine-tuned
- **Tracking:** ByteTrack (one unique ID per vehicle)
- **Counting:** virtual line, per class and per direction
- **Inputs:** video files and live streams (YouTube live, public HLS/MJPEG traffic cameras)
- **Deployment:** FastAPI + Docker, Streamlit demo, CPU inference (ONNX / OpenVINO)

## Demo

_Coming soon._

## Results

_Coming soon._

| Model | mAP50 | mAP50-95 | Counting error | FPS (CPU) |
|-------|-------|----------|----------------|-----------|
| Baseline (COCO) | – | – | – | – |
| Fine-tuned | – | – | – | – |

## Project structure

```
vehicle-counting/
├── configs/              # settings (classes, line, tracker)
├── data/                 # datasets (not tracked by git)
├── docs/                 # scope and step reports
├── notebooks/            # training on Colab / Kaggle
├── scripts/              # data download and conversion
├── src/vehicle_counter/  # main code
├── api/                  # FastAPI
├── app/                  # Streamlit demo
└── tests/
```

## Documentation

- [Step 0 — Project scope](docs/00_cadrage.md)

## Data and licenses

- Datasets (UA-DETRAC, BDD100K, Roboflow Universe) are **not** included in this repo. Only download and conversion scripts are provided.
- Live streams are processed in memory only. No raw video is stored or shared.

## License

This project is licensed under the **GNU AGPL-3.0** (see [LICENSE](LICENSE)), as required by Ultralytics YOLO.
