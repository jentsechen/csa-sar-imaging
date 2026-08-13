#!/usr/bin/env python3
"""Overlay horizontal/vertical intensity cross-sections through the ship in
P0002_3600_4400_1200_2000: union_masked input, union-mask CSA output (no
threshold), and CSA output with a fixed intensity threshold applied (see
threshold_union_csa.py) -- to visualize how thresholding reshapes the
profile relative to both the raw CSA reconstruction and the masked input.

The "input" line is read directly from union_masked/images/<stem>.jpg (the
GT-union-predicted-box-masked original, raw pixel values kept inside the
mask) -- NOT from union_pipeline/point_target_location/<stem>.json, even
though the two are byte-identical (gen_echo_signal_union_batch.py builds the
point_target_location JSON by dumping this exact image with img.tolist()).
Reading union_masked directly makes clear this is the masked *input image*,
not a sparse point-scatterer representation.

Does not modify any existing pipeline code -- standalone experiment script.

Usage:
    python plot_point_target_vs_csa_crosssection.py --threshold 120
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")
STEM = "P0002_3600_4400_1200_2000"
UNION_MASKED_PATH = os.path.join(BASE, "union_masked", "images", STEM + ".jpg")
CSA_PATH = os.path.join(BASE, "union_pipeline", "csa_jpg", STEM + ".jpg")
OUT_PATH = os.path.join(DIAGRAM_DIR, "point_target_vs_csa_crosssection.png")

ROW = 75
COL = 699
ROW_WINDOW = (30, 130)
COL_WINDOW = (650, 750)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(DIAGRAM_DIR, exist_ok=True)

    thresholded_path = os.path.join(BASE, "union_pipeline", f"csa_jpg_t{args.threshold}", STEM + ".jpg")

    point_target = cv2.imread(UNION_MASKED_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    csa = cv2.imread(CSA_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    csa_t = cv2.imread(thresholded_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    r0, r1 = COL_WINDOW
    x = np.arange(r0, r1)
    axes[0].plot(x, point_target[ROW, r0:r1], color="tab:blue", linestyle="-", label="union_masked (input)")
    axes[0].plot(x, csa[ROW, r0:r1], color="tab:orange", linestyle="-", label="CSA output")
    axes[0].plot(x, csa_t[ROW, r0:r1], color="tab:green", linestyle="--", label=f"CSA output + threshold={args.threshold}")
    axes[0].set_title(f"Horizontal cross-section (row={ROW})")
    axes[0].set_xlabel("column (range direction)")
    axes[0].set_ylabel("intensity (0-255)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    c0, c1 = ROW_WINDOW
    y = np.arange(c0, c1)
    axes[1].plot(y, point_target[c0:c1, COL], color="tab:blue", linestyle="-", label="union_masked (input)")
    axes[1].plot(y, csa[c0:c1, COL], color="tab:orange", linestyle="-", label="CSA output")
    axes[1].plot(y, csa_t[c0:c1, COL], color="tab:green", linestyle="--", label=f"CSA output + threshold={args.threshold}")
    axes[1].set_title(f"Vertical cross-section (col={COL})")
    axes[1].set_xlabel("row (azimuth direction)")
    axes[1].set_ylabel("intensity (0-255)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
