#!/usr/bin/env python3
"""Convert point_target_location/*.json (already thresholded, no CSA) directly
to grayscale JPGs, so thresholding-only accuracy loss can be isolated from
CSA-reconstruction accuracy loss.

Usage:
    python threshold_json_to_jpg_batch.py
"""
import json
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
POINT_TARGET_DIR = os.path.join(BASE, "point_target_location")
OUT_DIR = os.path.join(BASE, "threshold_jpg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(POINT_TARGET_DIR) if f.endswith(".json"))

    for stem in stems:
        with open(os.path.join(POINT_TARGET_DIR, stem + ".json")) as f:
            data = json.load(f)
        img = np.array(data, dtype=np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, stem + ".jpg"), img)

    print(f"Converted {len(stems)} scene(s) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
