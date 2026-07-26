#!/usr/bin/env python3
"""Sweep the threshold value used in jpg_to_point_target.py and measure how
Precision/Recall/mAP@0.5 degrade as a function of threshold, on the same 100
(offshore) scenes. No echo/CSA involved -- pure thresholding effect.

Usage:
    python threshold_sweep_eval.py --device cpu
    python threshold_sweep_eval.py --thresholds 0 50 100 150 200
"""
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE, "images")
LABELS_DIR = os.path.join(BASE, "labels")
SWEEP_DIR = os.path.join(BASE, "threshold_sweep")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")
RESULTS_CSV = os.path.join(BASE, "threshold_sweep_results.csv")

DEFAULT_THRESHOLDS = [0, 20, 40, 60, 80, 100, 120, 150, 180, 200, 220]


def all_stems():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg"))


def build_thresholded_set(threshold, stems):
    eval_dir = os.path.join(SWEEP_DIR, f"t_{threshold}")
    images_out = os.path.join(eval_dir, "images")
    labels_out = os.path.join(eval_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    total_nonzero_pct = 0.0
    for stem in stems:
        img = cv2.imread(os.path.join(IMAGES_DIR, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        thresholded = np.where(img < threshold, 0, img).astype(np.uint8)
        total_nonzero_pct += (thresholded != 0).sum() / img.size * 100

        img_out = os.path.join(images_out, stem + ".jpg")
        if not os.path.exists(img_out):
            cv2.imwrite(img_out, thresholded)

        lbl_src = os.path.join(LABELS_DIR, stem + ".txt")
        lbl_link = os.path.join(labels_out, stem + ".txt")
        if os.path.exists(lbl_src) and not os.path.exists(lbl_link):
            os.symlink(lbl_src, lbl_link)

    yaml_path = os.path.join(eval_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {eval_dir}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: ship\n"
        )
    avg_nonzero_pct = total_nonzero_pct / len(stems) if stems else 0.0
    return yaml_path, eval_dir, avg_nonzero_pct


def run_val(model, yaml_path, eval_dir, args):
    metrics = model.val(
        data=yaml_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=eval_dir,
        name="run",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    return box.mp, box.mr, box.map50, box.map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    stems = all_stems()
    print(f"Sweeping {len(args.thresholds)} threshold(s) over {len(stems)} scene(s)\n")

    model = YOLO(WEIGHTS)

    rows = []
    for t in args.thresholds:
        yaml_path, eval_dir, avg_nonzero_pct = build_thresholded_set(t, stems)
        p, r, map50, map5095 = run_val(model, yaml_path, eval_dir, args)
        rows.append((t, p, r, map50, map5095, avg_nonzero_pct))
        print(f"  threshold={t:>4}: Precision={p:.4f} Recall={r:.4f} "
              f"mAP@0.5={map50:.4f} mAP@0.5:0.95={map5095:.4f} "
              f"avg_nonzero_pct={avg_nonzero_pct:.4f}%")

    with open(RESULTS_CSV, "w") as f:
        f.write("threshold,precision,recall,map50,map5095,avg_nonzero_pct\n")
        for t, p, r, map50, map5095, avg_nonzero_pct in rows:
            f.write(f"{t},{p:.4f},{r:.4f},{map50:.4f},{map5095:.4f},{avg_nonzero_pct:.4f}\n")

    print(f"\nResults -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
