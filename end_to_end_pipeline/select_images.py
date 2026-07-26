#!/usr/bin/env python3
"""Randomly select N images (+ matching GT labels) from HRSID val set into
this directory, as the input set for the JPG -> point-target -> echo ->
CSA -> focused-image -> JPG -> infer.py end-to-end pipeline.

Usage:
    python select_images.py
    python select_images.py --n 100 --seed 0
    python select_images.py --region offshore
"""
import argparse
import json
import os
import random
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_IMAGES = os.path.join(BASE, "..", "sar_ship_detect", "HRSID_YOLO", "images", "val")
SRC_LABELS = os.path.join(BASE, "..", "sar_ship_detect", "HRSID_YOLO", "labels", "val")
INSHORE_OFFSHORE_DIR = os.path.join(BASE, "..", "sar_ship_detect", "HRSID_JPG", "inshore_offshore")
OUT_IMAGES = os.path.join(BASE, "images")
OUT_LABELS = os.path.join(BASE, "labels")


def region_filenames(region):
    with open(os.path.join(INSHORE_OFFSHORE_DIR, region + ".json")) as f:
        return {os.path.basename(im["file_name"]) for im in json.load(f)["images"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", choices=["all", "inshore", "offshore"], default="all",
                     help="restrict selection to the official HRSID inshore/offshore split")
    args = ap.parse_args()

    all_images = sorted(f for f in os.listdir(SRC_IMAGES) if f.lower().endswith(".jpg"))
    if args.region != "all":
        allowed = region_filenames(args.region)
        all_images = [f for f in all_images if f in allowed]

    rng = random.Random(args.seed)
    selected = rng.sample(all_images, args.n)

    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)

    for name in selected:
        stem = os.path.splitext(name)[0]
        shutil.copyfile(os.path.join(SRC_IMAGES, name), os.path.join(OUT_IMAGES, name))
        label_src = os.path.join(SRC_LABELS, stem + ".txt")
        if os.path.exists(label_src):
            shutil.copyfile(label_src, os.path.join(OUT_LABELS, stem + ".txt"))

    manifest_path = os.path.join(BASE, "selected_images.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(selected) + "\n")

    print(f"Copied {len(selected)} images -> {OUT_IMAGES}")
    print(f"Copied matching labels -> {OUT_LABELS}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
