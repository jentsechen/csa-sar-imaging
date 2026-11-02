#!/usr/bin/env python3
"""Linear vs log check on the HRSID pixel value (personal meeting 2026-11-02, Test 1).

Test data: homogeneous 32x32 sea patches (fully farther than GUARD px from every
HRSID ship polygon, no 255 pixels, at most 1/4 zeros), all offshore scenes in images/.

Slope k and intercept b of log std(x) = k log mean(x) + b over the patches
(k = 1 for linear amplitude or intensity, k = 0 after a log mapping).

Outputs end_to_end_pipeline/scene_stats/sea_speckle.json.
Usage: python3 scene_sea_speckle.py
"""
import json
import os

import cv2
import numpy as np

from scene_statistics import IMAGES, OUT, load_polygons, masks

GUARD = 10
P = 32


def sea_patches():
    """Mean and std of every homogeneous sea patch."""
    polys = load_polygons()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    stats = []
    for stem in stems:
        img = cv2.imread(os.path.join(IMAGES, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        ship, dist = masks(img.shape, polys.get(stem, []))
        sea = (~ship) & (dist > GUARD)
        for i in range(0, img.shape[0] - P + 1, P):
            for j in range(0, img.shape[1] - P + 1, P):
                if not sea[i:i + P, j:j + P].all():
                    continue
                x = img[i:i + P, j:j + P].astype(np.float64)
                # skip saturated patches and no-data areas (image borders filled with 0)
                if x.max() == 255 or np.count_nonzero(x == 0) > x.size // 4:
                    continue
                stats.append((x.mean(), x.std()))
    return np.array(stats)


def main():
    stats = sea_patches()
    k, b = np.polyfit(np.log(stats[:, 0]), np.log(stats[:, 1]), 1)
    out = {"patches": len(stats), "slope_k": float(k), "intercept_b": float(b)}
    with open(os.path.join(OUT, "sea_speckle.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
