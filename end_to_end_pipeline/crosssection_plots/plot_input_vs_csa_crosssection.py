#!/usr/bin/env python3
"""Same as plot_crosssection_scene.py, but with only Input SAR Image and CSA
(no CSA+Refinement/threshold curve) -- for looking at the raw CSA
reconstruction against the input on its own, without a refinement step
confounding the comparison. Combines the horizontal and vertical cuts into
ONE figure (two subplots) rather than two separate files.

Does not modify any existing pipeline code -- standalone experiment script.

Usage:
    python plot_input_vs_csa_crosssection.py P0033_1800_2600_4200_5000 \
        --row 147 --col 714 --row-window 80 215 --col-window 650 780
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from plot_crosssection_scene import get_gt_and_pred_boxes, edges_crossing, draw_box_edges

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--row-window", type=int, nargs=2, required=True, metavar=("R0", "R1"))
    ap.add_argument("--col-window", type=int, nargs=2, required=True, metavar=("C0", "C1"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-boxes", action="store_true")
    args = ap.parse_args()

    gt_edges_h, pred_edges_h = [], []
    gt_edges_v, pred_edges_v = [], []
    if not args.no_boxes:
        gt_boxes, pred_boxes = get_gt_and_pred_boxes(args.stem)
        gt_edges_h = edges_crossing(gt_boxes, fixed_row=args.row)
        pred_edges_h = edges_crossing(pred_boxes, fixed_row=args.row)
        gt_edges_v = edges_crossing(gt_boxes, fixed_col=args.col)
        pred_edges_v = edges_crossing(pred_boxes, fixed_col=args.col)

    union_masked_path = os.path.join(BASE, "union_masked", "images", args.stem + ".jpg")
    csa_path = os.path.join(BASE, "union_pipeline", "csa_jpg", args.stem + ".jpg")
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    out_path = args.out or os.path.join(DIAGRAM_DIR, f"crosssection_input_vs_csa_{args.stem}.png")

    point_target = cv2.imread(union_masked_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    csa = cv2.imread(csa_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c0, c1 = args.col_window
    x = np.arange(c0, c1)
    axes[0].plot(x, point_target[args.row, c0:c1], color="tab:blue", label="Input SAR Image")
    axes[0].plot(x, csa[args.row, c0:c1], color="tab:orange", label="CSA")
    draw_box_edges(axes[0], gt_edges_h, pred_edges_h, c0, c1)
    axes[0].set_title(f"range cross-section (row={args.row})")
    axes[0].set_xlabel("range direction")
    axes[0].set_ylabel("intensity (0-255)")
    axes[0].set_ylim(0, 255)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    r0, r1 = args.row_window
    y = np.arange(r0, r1)
    axes[1].plot(y, point_target[r0:r1, args.col], color="tab:blue", label="Input SAR Image")
    axes[1].plot(y, csa[r0:r1, args.col], color="tab:orange", label="CSA")
    draw_box_edges(axes[1], gt_edges_v, pred_edges_v, r0, r1)
    axes[1].set_title(f"azimuth cross-section (col={args.col})")
    axes[1].set_xlabel("azimuth direction")
    axes[1].set_ylabel("intensity (0-255)")
    axes[1].set_ylim(0, 255)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
