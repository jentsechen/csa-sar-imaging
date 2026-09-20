#!/usr/bin/env python3
"""Compare YOLO Precision/Recall/mAP@0.5 between the original images, the
union-mask CSA output (no threshold), and the union-mask CSA output with a
fixed intensity threshold applied (see threshold_union_csa.py), over the
same scene set.

Usage:
    python eval_union_csa_threshold.py --threshold 120 --device cpu
"""
import argparse
import os

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE, "images")
UNION_CSA_DIR = os.path.join(BASE, "union_pipeline", "csa_jpg")
LABELS_DIR = os.path.join(BASE, "labels")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")


def processed_stems(image_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg"))


def build_eval_set(image_dir, eval_dir, stems):
    images_out = os.path.join(eval_dir, "images")
    labels_out = os.path.join(eval_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for stem in stems:
        img_link = os.path.join(images_out, stem + ".jpg")
        if not os.path.exists(img_link):
            os.symlink(os.path.join(image_dir, stem + ".jpg"), img_link)

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
    return yaml_path


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
    return box.mp, box.mr, box.map50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    thresholded_dir = os.path.join(BASE, "union_pipeline", f"csa_jpg_t{args.threshold}")
    sets = {
        "original": ORIGINAL_DIR,
        "union_csa": UNION_CSA_DIR,
        f"union_csa_t{args.threshold}": thresholded_dir,
    }

    stems = processed_stems(UNION_CSA_DIR)
    print(f"Comparing on {len(stems)} scene(s)\n")

    model = YOLO(WEIGHTS)

    results = {}
    for name, image_dir in sets.items():
        eval_dir = os.path.join(BASE, f"{name}_union_eval")
        yaml_path = build_eval_set(image_dir, eval_dir, stems)
        results[name] = run_val(model, yaml_path, eval_dir, args)

    print(f"{'Set':<20} {'Precision':>10} {'Recall':>8} {'mAP@0.5':>9}")
    for name, (p, r, map50) in results.items():
        print(f"{name:<20} {p:>10.4f} {r:>8.4f} {map50:>9.4f}")


if __name__ == "__main__":
    main()
