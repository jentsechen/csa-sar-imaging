#!/usr/bin/env python3
"""Run YOLO validation on union_masked/images (pixels outside the GT+pred box
union zeroed out) to confirm detection performance matches the original.

Usage:
    python eval_union_masked.py --device cpu
"""
import argparse
import os

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
MASKED_IMAGES_DIR = os.path.join(BASE, "union_masked", "images")
LABELS_DIR = os.path.join(BASE, "labels")
EVAL_DIR = os.path.join(BASE, "union_masked_eval")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")


def build_eval_set(stems):
    images_out = os.path.join(EVAL_DIR, "images")
    labels_out = os.path.join(EVAL_DIR, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for stem in stems:
        img_link = os.path.join(images_out, stem + ".jpg")
        if not os.path.exists(img_link):
            os.symlink(os.path.join(MASKED_IMAGES_DIR, stem + ".jpg"), img_link)

        lbl_src = os.path.join(LABELS_DIR, stem + ".txt")
        lbl_link = os.path.join(labels_out, stem + ".txt")
        if os.path.exists(lbl_src) and not os.path.exists(lbl_link):
            os.symlink(lbl_src, lbl_link)

    yaml_path = os.path.join(EVAL_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {EVAL_DIR}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: ship\n"
        )
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(MASKED_IMAGES_DIR) if f.endswith(".jpg"))
    print(f"Evaluating {len(stems)} masked image(s)...")

    yaml_path = build_eval_set(stems)
    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=yaml_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=EVAL_DIR,
        name="run",
        exist_ok=True,
        plots=False,
    )

    box = metrics.box
    print(f"\nPrecision: {box.mp:.4f}")
    print(f"Recall:    {box.mr:.4f}")
    print(f"mAP@0.5:   {box.map50:.4f}")


if __name__ == "__main__":
    main()
