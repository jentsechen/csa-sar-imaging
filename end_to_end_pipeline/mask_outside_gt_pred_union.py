#!/usr/bin/env python3
"""Build a new eval set from original_eval/images: for each image, zero out
every pixel that falls outside the union of (ground-truth boxes) and
(model-predicted boxes). Reports the percentage of nonzero pixels remaining
per image.

This does NOT re-run inference on the masked images -- that's a separate,
later step. This script only masks + measures nonzero pixel percentage.

Usage:
    python mask_outside_gt_pred_union.py --device cpu
"""
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_IMAGES_DIR = os.path.join(BASE, "original_eval", "images")
SRC_LABELS_DIR = os.path.join(BASE, "original_eval", "labels")
OUT_DIR = os.path.join(BASE, "union_masked")
OUT_IMAGES_DIR = os.path.join(OUT_DIR, "images")
RESULTS_CSV = os.path.join(OUT_DIR, "nonzero_pct.csv")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                ])
    return boxes


def mask_outside_union(img, gt_boxes, pred_boxes):
    h, w = img.shape
    keep = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in gt_boxes + pred_boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        keep[y1:y2, x1:x2] = True
    return np.where(keep, img, 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(SRC_IMAGES_DIR) if f.endswith(".jpg"))
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)

    model = YOLO(WEIGHTS)

    rows = []
    for stem in stems:
        image_path = os.path.join(SRC_IMAGES_DIR, stem + ".jpg")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        gt_boxes = load_yolo_labels(os.path.join(SRC_LABELS_DIR, stem + ".txt"), w, h)

        r = model.predict(source=image_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                           device=args.device, verbose=False)[0]
        pred_boxes = r.boxes.xyxy.tolist()

        masked = mask_outside_union(img, gt_boxes, pred_boxes)
        cv2.imwrite(os.path.join(OUT_IMAGES_DIR, stem + ".jpg"), masked)

        pct = (masked != 0).sum() / masked.size * 100
        rows.append((stem, len(gt_boxes), len(pred_boxes), pct))
        print(f"  {stem}: GT={len(gt_boxes)} Pred={len(pred_boxes)} nonzero={pct:.4f}%")

    with open(RESULTS_CSV, "w") as f:
        f.write("stem,n_gt,n_pred,nonzero_pct\n")
        for stem, n_gt, n_pred, pct in rows:
            f.write(f"{stem},{n_gt},{n_pred},{pct:.4f}\n")

    avg_pct = sum(r[3] for r in rows) / len(rows) if rows else 0.0
    print(f"\n{len(rows)} images -> {OUT_IMAGES_DIR}")
    print(f"Average nonzero pixel percentage: {avg_pct:.4f}%")
    print(f"Per-image results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
