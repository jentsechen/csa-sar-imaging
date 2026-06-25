# HRSID Val — Inshore

**Date:** 2026-06-25 11:10
**Weights:** `weights/best.pt`
**Dataset yaml:** `HRSID_YOLO/data_val_inshore.yaml`
**Images:** 369 · **imgsz:** 800 · **conf:** 0.25 · **IoU:** 0.45
**Wall-clock time:** 123.5 s

## Overall Metrics

| Metric           | Value  |
|------------------|--------|
| Precision        | 0.8034 |
| Recall           | 0.7406 |
| mAP@0.5          | 0.8170 |
| mAP@0.5:0.95     | 0.5937 |
| Inference speed  | 324.6 ms/img |

## Per-Class Metrics

| Class        | Precision | Recall | F1   | AP@0.5   | AP@0.5:0.95 |
|--------------|-----------|--------|------|----------|-------------|
| ship         |     0.803 |  0.741 |  0.771 |    0.817 |        0.594 |

## Plots

| Plot | File |
|------|------|
| PR curve        | [BoxPR_curve.png](BoxPR_curve.png) |
| Precision curve | [BoxP_curve.png](BoxP_curve.png) |
| Recall curve    | [BoxR_curve.png](BoxR_curve.png) |
| F1 curve        | [BoxF1_curve.png](BoxF1_curve.png) |
| Confusion matrix | [confusion_matrix_normalized.png](confusion_matrix_normalized.png) |
