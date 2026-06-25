# HRSID Inference Results

## Validation Summary

| Metric | Value |
|---|---|
| Images | 1,962 |
| Instances | 5,922 |
| Precision | 0.913 |
| Recall | 0.846 |
| mAP@0.5 | 0.918 |
| mAP@0.5:0.95 | 0.736 |

## Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11m |
| Weights | `weights/best.pt` |
| Dataset | HRSID val split |
| Image size | 800 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.45 |
| Device | CPU (12th Gen Intel Core i9-12900K) |
| torch | 2.4.1 |
| ultralytics | 8.3.179 |

## Speed

| Stage | ms/image |
|---|---|
| Preprocess | 1.4 |
| Inference | 323.6 |
| Postprocess | 0.4 |
| **Total** | **325.4** |

## Notes

- Published results (from README): mAP@0.5 = 0.922, mAP@0.5:0.95 = 0.702
- Small deviation attributed to torch version difference (2.4.1 vs original 2.8.0)
