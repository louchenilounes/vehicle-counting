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

## Installation

Requirements: Python 3.10+ and Git.

```bash
git clone https://github.com/louchenilounes/vehicle-counting.git
cd vehicle-counting
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Check the environment (runs YOLO on a sample image):

```bash
python scripts/check_env.py
pytest
```

Training runs on Google Colab: see [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb).

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
- [Step 2 — Public data](docs/02_data.md)

## Data and licenses

- Datasets (UA-DETRAC, Roboflow Universe) are **not** included in this repo. Only download and conversion scripts are provided.
- UA-DETRAC: Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, CVIU 2020.
- Live streams are processed in memory only. No raw video is stored or shared.

## License

This project is licensed under the **GNU AGPL-3.0** (see [LICENSE](LICENSE)), as required by Ultralytics YOLO.
