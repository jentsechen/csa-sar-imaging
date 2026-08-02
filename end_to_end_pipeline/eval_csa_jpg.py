#!/usr/bin/env python3
"""Run YOLO validation (Precision/Recall/mAP@0.5) on csa_jpg/*.jpg against the
matching ground-truth labels in labels/, for whichever scenes have made it
through the full pipeline (jpg -> point_target -> echo_signal -> CSA -> jpg).

Usage:
    python eval_csa_jpg.py --device cpu
"""
import argparse
import os

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
CSA_JPG_DIR = os.path.join(BASE, "csa_jpg")
LABELS_DIR = os.path.join(BASE, "labels")
EVAL_DIR = os.path.join(BASE, "csa_eval")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")


def build_eval_set():
    images_out = os.path.join(EVAL_DIR, "images")
    labels_out = os.path.join(EVAL_DIR, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(CSA_JPG_DIR) if f.endswith(".jpg"))
    for stem in stems:
        img_link = os.path.join(images_out, stem + ".jpg")
        if not os.path.exists(img_link):
            os.symlink(os.path.join(CSA_JPG_DIR, stem + ".jpg"), img_link)

        lbl_src = os.path.join(LABELS_DIR, stem + ".txt")
        lbl_link = os.path.join(labels_out, stem + ".txt")
        if os.path.exists(lbl_src) and not os.path.exists(lbl_link):
            os.symlink(lbl_src, lbl_link)

    return stems


def write_data_yaml():
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

    stems = build_eval_set()
    yaml_path = write_data_yaml()
    print(f"Evaluating {len(stems)} CSA-reconstructed image(s)...")

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
    print(f"\nImages: {len(stems)}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall:    {box.mr:.4f}")
    print(f"mAP@0.5:   {box.map50:.4f}")


if __name__ == "__main__":
    main()
