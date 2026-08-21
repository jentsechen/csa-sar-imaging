#!/usr/bin/env python3
"""Generalized version of plot_point_target_vs_csa_crosssection.py: overlay
horizontal/vertical intensity cross-sections (union_masked input, CSA output,
CSA output + threshold) for an arbitrary scene/row/col/window -- used to
visualize per-image cases found by find_threshold_diff_images.py where
thresholding measurably changes the detection outcome.

The "input" line is read directly from union_masked/images/<stem>.jpg (the
GT-union-predicted-box-masked original, raw pixel values kept inside the
mask) -- NOT from union_pipeline/point_target_location/<stem>.json, even
though the two are byte-identical (gen_echo_signal_union_batch.py builds the
point_target_location JSON by dumping this exact image with img.tolist()).
Reading union_masked directly makes clear this is the masked *input image*,
not a sparse point-scatterer representation.

Saves the horizontal and vertical cross-sections as two SEPARATE .png files
(suffixed _horizontal / _vertical) rather than one combined figure.

By default also overlays the extent of the GT box(es) and the model's
predicted box(es) on the ORIGINAL image (the same two box sets
mask_outside_gt_pred_union.py unions to build union_masked/) as separate
dotted vertical/horizontal lines wherever a box crosses the cut row/col --
so the reader can see how far each box extends relative to the profile, and
whether GT and detection agree. Pass --no-boxes to disable (re-runs YOLO on
the original image each time, same as mask_outside_gt_pred_union.py's own
box source).

Output defaults to diagram/thresholding/union_csa/ (repo-level diagram
folder, not under end_to_end_pipeline/) -- pass --out to override.

Usage:
    python plot_crosssection_scene.py P0029_3000_3800_4800_5600 \
        --row 0 --col 365 --row-window 0 20 --col-window 335 400 \
        --threshold 120 --out /path/to/crosssection_P0029.png
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_IMAGES_DIR = os.path.join(BASE, "images")
LABELS_DIR = os.path.join(BASE, "labels")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")


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


def get_gt_and_pred_boxes(stem, imgsz=800, conf=0.25, iou=0.45, device="cpu"):
    """GT boxes and predicted boxes on the ORIGINAL image, kept separate --
    mask_outside_gt_pred_union.py unions these two sets to build
    union_masked/images/, but here we want to tell them apart."""
    from ultralytics import YOLO

    image_path = os.path.join(ORIGINAL_IMAGES_DIR, stem + ".jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    gt_boxes = load_yolo_labels(os.path.join(LABELS_DIR, stem + ".txt"), w, h)

    model = YOLO(WEIGHTS)
    r = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    pred_boxes = r.boxes.xyxy.tolist()

    return gt_boxes, pred_boxes


def edges_crossing(boxes, fixed_row=None, fixed_col=None):
    """Edge positions of boxes[i] along the axis perpendicular to the cut,
    for boxes whose extent actually contains the fixed row/col."""
    edges = []
    for x1, y1, x2, y2 in boxes:
        if fixed_row is not None and y1 <= fixed_row <= y2:
            edges.extend([x1, x2])
        if fixed_col is not None and x1 <= fixed_col <= x2:
            edges.extend([y1, y2])
    return edges


def normalize_to_peak(arr, target):
    """Scale a 1D profile so its own max within the plotted window equals
    `target` -- ONLY for visually comparing relative shape across curves
    whose absolute 0-255 scales are not calibrated to each other
    (union_masked is a raw pixel value, CSA output is independently
    re-normalized per-image to its own dB peak -- see conversation notes).
    Not meaningful as an absolute physical quantity, observation purposes
    only. union_masked itself is intentionally left un-normalized (its raw
    pixel value already IS a meaningful physical quantity) -- CSA output
    and CSA+threshold get peak-scaled to `target`, which the caller sets to
    union_masked's own peak within the same window, so all three curves
    reach the same visual peak height for shape comparison."""
    peak = np.max(arr)
    return arr * (target / peak) if peak > 0 else arr


def draw_box_edges(ax, gt_edges, pred_edges, lo, hi):
    for i, edge in enumerate(gt_edges):
        if lo <= edge <= hi:
            ax.axvline(edge, color="tab:purple", linestyle="-", linewidth=1.3,
                       label="GT Box" if i == 0 else None)
    for i, edge in enumerate(pred_edges):
        if lo <= edge <= hi:
            ax.axvline(edge, color="tab:cyan", linestyle="-", linewidth=1.3,
                       label="Pred Box" if i == 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--row", type=int, required=True, help="fixed row for the horizontal cross-section")
    ap.add_argument("--col", type=int, required=True, help="fixed col for the vertical cross-section")
    ap.add_argument("--row-window", type=int, nargs=2, required=True, metavar=("R0", "R1"))
    ap.add_argument("--col-window", type=int, nargs=2, required=True, metavar=("C0", "C1"))
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csa-path", default=None,
                     help="override path to the CSA output image (e.g. from a standalone "
                          "experiment directory with different radar parameters, such as "
                          "half_bandwidth_experiment/csa_jpg/<stem>.jpg)")
    ap.add_argument("--union-masked-path", default=None,
                     help="override path to the union_masked (input) image")
    ap.add_argument("--thresholded-path", default=None,
                     help="override path to the CSA+threshold image (e.g. a lossless .png saved "
                          "alongside the pipeline's .jpg, to avoid JPEG re-compression artifacts "
                          "in the cross-section)")
    ap.add_argument("--no-boxes", action="store_true",
                     help="don't overlay GT / predicted box extents")
    ap.add_argument("--normalize", action="store_true",
                     help="scale CSA and CSA+threshold each to union_masked's own peak within the "
                          "plotted window (union_masked is left as raw pixel values, unchanged), "
                          "for OBSERVATION only -- union_masked/CSA/CSA+threshold are not on a "
                          "shared absolute intensity scale (see conversation notes), so this is "
                          "for comparing relative shape, not real physical values")
    args = ap.parse_args()

    gt_edges_h, pred_edges_h = [], []  # column positions (for the horizontal cut)
    gt_edges_v, pred_edges_v = [], []  # row positions (for the vertical cut)
    if not args.no_boxes:
        gt_boxes, pred_boxes = get_gt_and_pred_boxes(args.stem)
        gt_edges_h = edges_crossing(gt_boxes, fixed_row=args.row)
        pred_edges_h = edges_crossing(pred_boxes, fixed_row=args.row)
        gt_edges_v = edges_crossing(gt_boxes, fixed_col=args.col)
        pred_edges_v = edges_crossing(pred_boxes, fixed_col=args.col)

    union_masked_path = args.union_masked_path or os.path.join(BASE, "union_masked", "images", args.stem + ".jpg")
    csa_path = args.csa_path or os.path.join(BASE, "union_pipeline", "csa_jpg", args.stem + ".jpg")
    thresholded_path = args.thresholded_path or os.path.join(
        BASE, "union_pipeline", f"csa_jpg_t{args.threshold}", args.stem + ".jpg"
    )
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    out_base = args.out or os.path.join(DIAGRAM_DIR, f"crosssection_{args.stem}_t{args.threshold}.png")
    out_root, out_ext = os.path.splitext(out_base)

    point_target = cv2.imread(union_masked_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    csa = cv2.imread(csa_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    csa_t = cv2.imread(thresholded_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    fig, ax = plt.subplots(figsize=(7, 5))
    c0, c1 = args.col_window
    x = np.arange(c0, c1)
    pt_h, csa_h, csa_t_h = point_target[args.row, c0:c1], csa[args.row, c0:c1], csa_t[args.row, c0:c1]
    if args.normalize:
        target_h = np.max(pt_h)
        csa_h, csa_t_h = normalize_to_peak(csa_h, target_h), normalize_to_peak(csa_t_h, target_h)
    ax.plot(x, pt_h, color="tab:blue", linestyle="-", label="Input SAR Image")
    ax.plot(x, csa_h, color="tab:orange", linestyle="-", label="CSA")
    ax.plot(x, csa_t_h, color="tab:green", linestyle="--", label="CSA + Refinement")
    draw_box_edges(ax, gt_edges_h, pred_edges_h, c0, c1)
    ax.set_title("range cross-section")
    ax.set_xlabel("range direction")
    ax.set_ylabel("intensity (0-255)")
    ax.set_ylim(0, 255)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    horizontal_path = f"{out_root}_horizontal{out_ext}"
    plt.savefig(horizontal_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {horizontal_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    r0, r1 = args.row_window
    y = np.arange(r0, r1)
    pt_v, csa_v, csa_t_v = point_target[r0:r1, args.col], csa[r0:r1, args.col], csa_t[r0:r1, args.col]
    if args.normalize:
        target_v = np.max(pt_v)
        csa_v, csa_t_v = normalize_to_peak(csa_v, target_v), normalize_to_peak(csa_t_v, target_v)
    ax.plot(y, pt_v, color="tab:blue", linestyle="-", label="Input SAR Image")
    ax.plot(y, csa_v, color="tab:orange", linestyle="-", label="CSA")
    ax.plot(y, csa_t_v, color="tab:green", linestyle="--", label="CSA + Refinement")
    draw_box_edges(ax, gt_edges_v, pred_edges_v, r0, r1)
    ax.set_title("azimuth cross-section")
    ax.set_xlabel("azimuth direction")
    ax.set_ylabel("intensity (0-255)")
    ax.set_ylim(0, 255)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    vertical_path = f"{out_root}_vertical{out_ext}"
    plt.savefig(vertical_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {vertical_path}")


if __name__ == "__main__":
    main()
