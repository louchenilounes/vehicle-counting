# Step 2 — Public data

## Target classes

| ID | Class |
|----|-------|
| 0 | car |
| 1 | truck |
| 2 | bus |
| 3 | van |

The mapping from each source is defined in [`configs/classes.yaml`](../configs/classes.yaml).

## Sources

| Source | Status | Use | Terms |
|--------|--------|-----|-------|
| UA-DETRAC | used | main data: fixed traffic cameras | official download from the authors; cite the paper (below); not redistributed |
| Roboflow Universe | to choose | extra trucks and vans | license checked per dataset |
| BDD100K | not used for now | — | vans are labeled as "car" (conflicts with our `van` class) and it is a dashcam view (out of scope). May be reused only for trucks if truck results are weak (Step 7). |

## UA-DETRAC processing

1. **Class mapping:** `car → car`, `bus → bus`, `van → van`, `others → truck` (to confirm with a visual check of 100 random `others` crops).
2. **Frame sampling:** 1 frame out of 10. Consecutive video frames are almost identical; keeping all of them adds duplicates and inflates the metrics.
3. **Ignored regions:** UA-DETRAC marks regions that are not annotated. They are filled with gray (114, 114, 114), and boxes more than 50% inside them are removed, so the model is not penalized for detecting real but unlabeled vehicles.
4. **Boxes:** clipped to the image; boxes smaller than 2 px after clipping are removed.
5. **Metadata:** `metadata.csv` stores the sequence, frame, weather and box counts of each image. It is used for the split by sequence (Step 4) and the day/night/rain test (Step 11).

Code: [`src/vehicle_counter/data/detrac.py`](../src/vehicle_counter/data/detrac.py), tested in [`tests/test_convert.py`](../tests/test_convert.py).
Notebook: [`notebooks/01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb).

## Results

_To be filled with the real output of `scripts/dataset_stats.py`._

## Citation

Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, Computer Vision and Image Understanding (CVIU), 2020.
