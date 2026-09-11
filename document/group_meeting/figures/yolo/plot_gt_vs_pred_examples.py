#!/usr/bin/env python3
"""Pick one inshore and one offshore HRSID scene (higher ship count than the
first pass), run YOLO inference, and save each with GT boxes (green) and
predicted boxes (red) overlaid -- for the group-meeting report's
Introduction/System Definition figures.

Both scenes come directly from the full HRSID_YOLO val set (the
end_to_end_pipeline/images/ selection is offshore-only and its offshore
scenes only have up to ~5 ships, too few for this pass), restricted to the
official inshore/offshore split in
sar_ship_detect/HRSID_JPG/inshore_offshore/{inshore,offshore}.json.

Usage:
    python plot_gt_vs_pred_examples.py
"""
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
HRSID_YOLO = os.path.join(REPO, "sar_ship_detect", "HRSID_YOLO")
WEIGHTS = os.path.join(REPO, "sar_ship_detect", "weights", "best.pt")

EXAMPLES = [
    {
        "region": "offshore",
        "stem": "P0105_3000_3800_15600_16400",
    },
    {
        "region": "inshore",
        "stem": "P0019_600_1400_7200_8000",
    },
]
for ex in EXAMPLES:
    ex["image"] = os.path.join(HRSID_YOLO, "images", "val", ex["stem"] + ".jpg")
    ex["label"] = os.path.join(HRSID_YOLO, "labels", "val", ex["stem"] + ".txt")

IMGSZ, CONF, IOU = 800, 0.25, 0.45


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
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


def draw_boxes(ax, boxes, color, label, linestyle="-"):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.6, edgecolor=color, facecolor="none", linestyle=linestyle,
            label=label if i == 0 else None,
        ))


def main():
    model = YOLO(WEIGHTS)

    for ex in EXAMPLES:
        img = cv2.imread(ex["image"], cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        gt_boxes = load_yolo_labels(ex["label"], w, h)

        r = model.predict(source=ex["image"], imgsz=IMGSZ, conf=CONF, iou=IOU,
                           device="cpu", verbose=False)[0]
        pred_boxes = r.boxes.xyxy.tolist()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower")
        draw_boxes(ax, gt_boxes, "tab:green", "GT box")
        draw_boxes(ax, pred_boxes, "tab:red", "Pred box", linestyle="--")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{ex['region']}: {ex['stem']}")
        ax.set_xlabel("range direction")
        ax.set_ylabel("azimuth direction")
        plt.tight_layout()

        out_path = os.path.join(HERE, f"{ex['region']}_example_gt_vs_pred.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out_path}  (GT={len(gt_boxes)}, pred={len(pred_boxes)})")


if __name__ == "__main__":
    main()
