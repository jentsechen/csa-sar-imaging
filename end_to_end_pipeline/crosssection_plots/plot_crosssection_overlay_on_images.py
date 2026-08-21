#!/usr/bin/env python3
"""Show the union_masked / CSA / CSA+threshold images (cropped around the
region of interest) with:
  - red solid lines marking the row/col used by plot_crosssection_scene.py's
    cross-sections, so the cut lines can be seen directly on the image
    content
  - GT box(es) (tab:purple rectangle) and the model's predicted box(es) on
    the ORIGINAL image (tab:cyan rectangle) -- the same two box sets
    mask_outside_gt_pred_union.py unions to build union_masked/ -- kept
    visually distinct so GT vs. detection can be compared directly on top
    of each processing stage

Each image is saved as its own SEPARATE .png file (suffixed _union_masked /
_csa / _csa_t<threshold>), not one combined figure. Output defaults to
diagram/thresholding/union_csa/ (repo-level diagram folder, not under
end_to_end_pipeline/) -- pass --out to override.

Usage:
    python plot_crosssection_overlay_on_images.py P0033_1800_2600_4200_5000 \
        --row 147 --col 714 --row-window 80 215 --col-window 650 780 \
        --threshold 120
"""
import argparse
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from plot_crosssection_scene import get_gt_and_pred_boxes

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")


def draw_boxes(ax, boxes, color, label):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.4, edgecolor=color, facecolor="none",
            label=label if i == 0 else None,
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--row", type=int, required=True, help="row marked with a horizontal red line")
    ap.add_argument("--col", type=int, required=True, help="col marked with a vertical red line")
    ap.add_argument("--row-window", type=int, nargs=2, required=True, metavar=("R0", "R1"),
                     help="crop window (rows) shown in each panel")
    ap.add_argument("--col-window", type=int, nargs=2, required=True, metavar=("C0", "C1"),
                     help="crop window (cols) shown in each panel")
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-boxes", action="store_true",
                     help="don't overlay GT / predicted box rectangles")
    args = ap.parse_args()

    gt_boxes, pred_boxes = ([], []) if args.no_boxes else get_gt_and_pred_boxes(args.stem)

    union_masked_path = os.path.join(BASE, "union_masked", "images", args.stem + ".jpg")
    csa_path = os.path.join(BASE, "union_pipeline", "csa_jpg", args.stem + ".jpg")
    thresholded_path = os.path.join(BASE, "union_pipeline", f"csa_jpg_t{args.threshold}", args.stem + ".jpg")
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    out_base = args.out or os.path.join(
        DIAGRAM_DIR, f"image_overlay_{args.stem}_t{args.threshold}.png"
    )
    out_root, out_ext = os.path.splitext(out_base)

    r0, r1 = args.row_window
    c0, c1 = args.col_window

    panels = [
        ("union_masked", "Input SAR Image", cv2.imread(union_masked_path, cv2.IMREAD_GRAYSCALE)),
        ("csa", "CSA", cv2.imread(csa_path, cv2.IMREAD_GRAYSCALE)),
        (f"csa_t{args.threshold}", "CSA + Refinement",
         cv2.imread(thresholded_path, cv2.IMREAD_GRAYSCALE)),
    ]

    for suffix, title, img in panels:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        crop = img[r0:r1, c0:c1]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=[c0, c1, r0, r1], origin="lower")
        ax.axhline(args.row, color="red", linestyle="-", linewidth=1.2, label=f"row={args.row}")
        ax.axvline(args.col, color="red", linestyle="-", linewidth=1.2, label=f"col={args.col}")
        draw_boxes(ax, gt_boxes, "tab:purple", "GT Box")
        draw_boxes(ax, pred_boxes, "tab:cyan", "Pred Box")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("range direction")
        ax.set_ylabel("azimuth direction")
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        panel_path = f"{out_root}_{suffix}{out_ext}"
        plt.savefig(panel_path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {panel_path}")


if __name__ == "__main__":
    main()
