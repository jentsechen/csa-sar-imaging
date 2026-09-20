#!/usr/bin/env python3
"""Apply an adaptive Lee speckle filter to a union-mask CSA output JPG, as a
different-in-kind "refinement" algorithm to compare against the fixed
global intensity threshold used elsewhere (threshold_union_csa.py).

Where thresholding makes a hard, non-adaptive binary decision per pixel
(zero out anything below T, keep everything above T unchanged), the Lee
filter is a LOCAL-STATISTICS adaptive smoother: at every pixel it blends the
pixel's own value with its local neighborhood mean, weighted by how
"noisy" (high local coefficient of variation) vs. "textured/edge-like" (low
local coefficient of variation relative to the assumed noise level) that
neighborhood looks. This is the classic despeckle-filter family (Phase 5 /
issue #21 of this project's milestones -- Lee is a close cousin of the
Frost/Kuan/Gamma-MAP filters listed there), so unlike a threshold it:
  - never hard-zeros anything (no clean binary cutoff)
  - smooths near-uniform/background regions more aggressively
  - preserves strong edges/high-contrast scatterers (the ship's bright
    returns) with comparatively little smoothing, since their local
    coefficient of variation is high (looks "textured", not "noisy")

Lee filter formula (Lee, 1980), applied directly to the 0-255 dB-scaled CSA
JPEG (same representation threshold_union_csa.py operates on, not the raw
complex/intensity SAR data the filter was originally derived for -- treated
here purely as a different, plausible post-processing algorithm for
comparison, not a rigorous multi-look despeckle implementation):

    local_mean, local_var  = mean/variance in a WINDOW x WINDOW neighborhood
    Cu^2 = 1 / NUM_LOOKS                      (assumed noise coefficient of variation)
    Ci^2 = local_var / local_mean^2           (local coefficient of variation, pixel-wise)
    W    = clip(1 - Cu^2 / Ci^2, 0, 1)        (weight: 1 = keep detail, 0 = full smoothing)
    output = local_mean + W * (pixel - local_mean)

Does not modify any existing pipeline code -- standalone experiment script.

Usage:
    python lee_filter_experiment.py P0033_1800_2600_4200_5000 --window 7 --num-looks 4
    python lee_filter_experiment.py --all --window 7 --num-looks 4   # every completed scene
"""
import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "union_pipeline", "csa_jpg")
OUT_DIR = os.path.join(BASE, "union_pipeline", "csa_jpg_lee")


def lee_filter(img, window, num_looks):
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size=window)
    local_sqr_mean = uniform_filter(img * img, size=window)
    local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0.0)

    cu2 = 1.0 / num_looks
    with np.errstate(divide="ignore", invalid="ignore"):
        ci2 = local_var / np.maximum(local_mean ** 2, 1e-6)
        weight = np.where(ci2 > 0, 1.0 - cu2 / ci2, 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    out = local_mean + weight * (img - local_mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default=None)
    ap.add_argument("--all", action="store_true", help="process every completed scene in union_pipeline/csa_jpg/")
    ap.add_argument("--window", type=int, default=7, help="local neighborhood size (odd)")
    ap.add_argument("--num-looks", type=float, default=4.0,
                     help="assumed equivalent number of looks (controls how aggressively "
                          "low-contrast/background regions get smoothed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.all:
        os.makedirs(OUT_DIR, exist_ok=True)
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(SRC_DIR) if f.endswith(".jpg"))
        for stem in stems:
            img = cv2.imread(os.path.join(SRC_DIR, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
            filtered = lee_filter(img, args.window, args.num_looks)
            cv2.imwrite(os.path.join(OUT_DIR, stem + ".jpg"), filtered)
        print(f"Lee-filtered {len(stems)} image(s) (window={args.window}, num_looks={args.num_looks}) -> {OUT_DIR}")
        return

    if not args.stem:
        ap.error("either a stem or --all is required")

    src_path = os.path.join(SRC_DIR, args.stem + ".jpg")
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    filtered = lee_filter(img, args.window, args.num_looks)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, args.stem + ".png")
    cv2.imwrite(out_path, filtered)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
