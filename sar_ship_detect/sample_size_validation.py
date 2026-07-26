#!/usr/bin/env python3
"""Empirically find how few val images are needed to reproduce the full-set
Precision / Recall / mAP@0.5.

For each candidate subset size, draws multiple random samples (no replacement)
from HRSID_YOLO/images/val, runs model.val() on each, and reports the mean +
spread of the metrics so you can see at what size they stabilize near the
full 1962-image result.

Usage:
    python sample_size_validation.py
    python sample_size_validation.py --sizes 100 300 500 1000 --trials 3 --device cpu
"""
import argparse
import os
import random
import statistics

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.join(BASE, "HRSID_YOLO")
VAL_DIR = os.path.join(YOLO_ROOT, "images", "val")
TMP_DIR = os.path.join(BASE, "runs", "sample_size_check")

DEFAULT_SIZES = [100, 200, 300, 500, 800, 1200]


def list_val_images():
    return sorted(f for f in os.listdir(VAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def write_subset_yaml(image_names, tag):
    os.makedirs(TMP_DIR, exist_ok=True)
    list_path = os.path.join(TMP_DIR, f"{tag}.txt")
    with open(list_path, "w") as f:
        for name in image_names:
            f.write(os.path.join(VAL_DIR, name) + "\n")

    yaml_path = os.path.join(TMP_DIR, f"{tag}.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {YOLO_ROOT}\n"
            f"train: images/train\n"
            f"val: {os.path.relpath(list_path, YOLO_ROOT)}\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: ship\n"
        )
    return yaml_path


def run_val(model, yaml_path, args):
    metrics = model.val(
        data=yaml_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=TMP_DIR,
        name="run",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    return box.mp, box.mr, box.map50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--trials", type=int, default=3, help="random samples per size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the full-set reference run (for quick smoke tests)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    all_images = list_val_images()
    n_full = len(all_images)
    print(f"Full val set: {n_full} images\n")

    rng = random.Random(args.seed)

    print(f"{'Size':>6} {'Trial':>5} {'Precision':>10} {'Recall':>8} {'mAP@0.5':>8}")
    results = {}
    for size in args.sizes:
        if size >= n_full:
            continue
        trial_metrics = []
        for trial in range(args.trials):
            sample = rng.sample(all_images, size)
            yaml_path = write_subset_yaml(sample, f"n{size}_t{trial}")
            p, r, map50 = run_val(model, yaml_path, args)
            trial_metrics.append((p, r, map50))
            print(f"{size:>6} {trial:>5} {p:>10.4f} {r:>8.4f} {map50:>8.4f}")
        results[size] = trial_metrics

    # full set as the reference
    if args.skip_full:
        p_full = r_full = map50_full = float("nan")
    else:
        full_yaml = write_subset_yaml(all_images, "full")
        p_full, r_full, map50_full = run_val(model, full_yaml, args)
        print(f"{n_full:>6} {'ref':>5} {p_full:>10.4f} {r_full:>8.4f} {map50_full:>8.4f}")

    print("\n=== Summary (mean +/- std across trials, vs full-set reference) ===")
    print(f"{'Size':>6} {'Precision':>16} {'Recall':>16} {'mAP@0.5':>16}")
    for size in args.sizes:
        if size not in results:
            continue
        ps = [m[0] for m in results[size]]
        rs = [m[1] for m in results[size]]
        maps = [m[2] for m in results[size]]

        def fmt(vals, ref):
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mean:.3f}+/-{std:.3f} (ref {ref:.3f})"

        print(f"{size:>6} {fmt(ps, p_full):>16} {fmt(rs, r_full):>16} {fmt(maps, map50_full):>16}")
    print(f"{n_full:>6} {'(reference, full set)':>50}")


if __name__ == "__main__":
    main()
