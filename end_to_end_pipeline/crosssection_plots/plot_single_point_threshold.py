#!/usr/bin/env python3
"""Overlay the single-point-target range/azimuth profiles of CSA and
CSA + thresholding at THRESHOLD, to show the narrower mainlobe extent after
refinement.

Each 1D complex profile through the peak is sinc-upsampled (same as
gen_single_point_target.py), converted to the pipeline's 0-255 intensity
(30 dB clip below the peak + min-max normalize, as in
csa_to_jpg_union_batch.py), then thresholded as in threshold_union_csa.py:
    refined = 0 where intensity < THRESHOLD, else intensity
so values above the threshold are kept unchanged.

Does not modify any existing pipeline code -- standalone plotting script.

Usage:
    python plot_single_point_threshold.py --threshold 120 --out fig.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from gen_single_point_target import (  # noqa: E402
    FOCUSED_IMAGE_DIR, INPUT_PAR, SENSOR_SPEED_M_S, SPEED_OF_LIGHT_M_S, TARGET,
    UPSAMPLE_FACTOR, to_mag_db, upsample_via_zero_padding)

DYNAMIC_RANGE_DB = 30
HALF_WINDOW_PX = 8


def to_intensity(profile_db):
    peak = profile_db.max()
    clipped = np.clip(profile_db, peak - DYNAMIC_RANGE_DB, peak)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min()) * 255.0


def width_above(x, y, level):
    """Width of the contiguous region around the peak where y >= level."""
    p = int(np.argmax(y))
    left = p
    while left > 0 and y[left - 1] >= level:
        left -= 1
    right = p
    while right < len(y) - 1 and y[right + 1] >= level:
        right += 1
    return x[right] - x[left]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=120)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "single_point_threshold.png"))
    args = ap.parse_args()

    focused = np.load(os.path.join(FOCUSED_IMAGE_DIR, TARGET + ".npy"), mmap_mode="r")
    mag = np.abs(focused)
    peak_row, peak_col = np.unravel_index(np.argmax(mag), mag.shape)

    profiles = {
        "range": (np.asarray(focused[peak_row, :]),
                  SPEED_OF_LIGHT_M_S / (2 * INPUT_PAR["sampling_freq_hz"])),
        "azimuth": (np.asarray(focused[:, peak_col]),
                    SENSOR_SPEED_M_S / INPUT_PAR["pulse_rep_freq_hz"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, (profile, spacing_m)) in zip(axes, profiles.items()):
        up = upsample_via_zero_padding(profile, UPSAMPLE_FACTOR)
        csa = to_intensity(to_mag_db(up))
        refined = np.where(csa < args.threshold, 0.0, csa)

        peak = int(np.argmax(csa))
        x = (np.arange(len(csa)) - peak) / UPSAMPLE_FACTOR
        sel = np.abs(x) <= HALF_WINDOW_PX
        x, csa, refined = x[sel], csa[sel], refined[sel]

        # mainlobe extent = contiguous nonzero region around the peak
        w_csa = width_above(x, csa, 1e-9)
        w_ref = width_above(x, refined, 1e-9)
        print(f"{name}: mainlobe extent CSA={w_csa:.2f}px ({w_csa*spacing_m:.2f}m), "
              f"refined={w_ref:.2f}px ({w_ref*spacing_m:.2f}m)")

        ax.plot(x, csa, color="tab:blue", label="CSA")
        ax.plot(x, refined, color="tab:orange", linestyle="--",
                label=f"CSA + Refinement ($\\lambda$={args.threshold:g})")
        ax.set_title(f"{name} profile")
        ax.set_xlabel(f"{name} pixel (relative to peak)")
        ax.set_ylabel("intensity (0-255)")
        ax.set_ylim(0, 265)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
