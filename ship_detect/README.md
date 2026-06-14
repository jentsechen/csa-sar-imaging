# ship_detect — standalone SAR ship detection

Self-contained copy of the trained YOLOv8m ship detector (from `UC_yolov8m_train`,
Umbra/Capella domain, single class `ship`). Runs on **CPU or GPU**, no dependency
on the original `yolo-llm` repo or the LLM half.

## Contents
```
weights/best.pt              trained YOLOv8m detector (single class: ship)
detect.py                    standalone inference (patches folder OR full GeoTIFF)
requirements-cpu.txt         minimal CPU deps
test_data/patches/           6 sample 640px patches (+ .txt ground-truth labels)
test_data/scene/             ChangXingShipyard_wgs84.tif (full SAR scene)
results/                     output CSVs + annotated/ images
```

## Setup (fresh machine, CPU)
```bash
pip install -r requirements-cpu.txt
# or, for a guaranteed CPU torch build:
# pip install ultralytics opencv-python numpy pandas
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run
```bash
# A) folder of patches (fast)
python3 detect.py --source test_data/patches --device cpu

# B) folder of patches + evaluate against ground-truth labels
python3 detect.py --source test_data/patches --device cpu --eval

# C) full scene, multi-scale sliding window
python3 detect.py --source test_data/scene/ChangXingShipyard_wgs84.tif \
    --device cpu --stride 512 --scales 1.0

# on a GPU box, swap:  --device 0
```
Outputs a CSV of detections (`image, cls, x_center, y_center, width, height, conf`)
and annotated images under `results/annotated/`.

## Evaluation (`--eval`)
Compares model predictions against the YOLO-format `.txt` ground-truth labels that
sit alongside each patch image. Metrics are computed at IoU ≥ 0.5.

| Metric | Description |
|---|---|
| Precision | Fraction of predicted boxes that match a GT box |
| Recall | Fraction of GT boxes that were detected |
| F1 | Harmonic mean of precision and recall |
| AP@50 | Area under the precision-recall curve at IoU ≥ 0.5 |

Per-image and overall results are printed to stdout and saved to `results/metrics.csv`.
`--eval` is only meaningful in patch mode (scene GeoTIFFs have no GT labels).

## Notes
- `--conf` controls the confidence threshold (default 0.25). Lower it (e.g. `--conf 0.1`) to improve recall at the cost of more false positives.
- GeoTIFFs are read with OpenCV and normalized to 8-bit 3-channel automatically.
  For geo-referenced (lat/lon) output, add `rasterio` and map pixel→CRS coords.
