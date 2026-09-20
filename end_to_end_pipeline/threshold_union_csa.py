#!/usr/bin/env python3
"""Apply a fixed intensity threshold to every union-mask CSA output JPG
(pixels below THRESHOLD -> 0), producing a second image set for a YOLO
accuracy comparison against the un-thresholded union_csa set.

Follow-up to the single-crop threshold sweep in diagram/thresholding/union_csa/
(which found higher thresholds progressively erode real ship pixels without
fully removing background speckle) -- this applies the same idea across the
full 92-scene union-mask pipeline output to see the effect on detection metrics.

Does not modify any existing pipeline code -- standalone experiment script.

Usage:
    python threshold_union_csa.py --threshold 120
"""
import argparse
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "union_pipeline", "csa_jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=120)
    args = ap.parse_args()

    dst_dir = os.path.join(BASE, "union_pipeline", f"csa_jpg_t{args.threshold}")
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".jpg"))
    for f in files:
        img = cv2.imread(os.path.join(SRC_DIR, f), cv2.IMREAD_GRAYSCALE)
        thresholded = np.where(img < args.threshold, 0, img).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_dir, f), thresholded)

    print(f"Thresholded {len(files)} image(s) at T={args.threshold} -> {dst_dir}")


if __name__ == "__main__":
    main()
