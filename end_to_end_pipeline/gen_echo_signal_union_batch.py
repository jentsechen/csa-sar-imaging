#!/usr/bin/env python3
"""Batch-generate echo signals from the union-masked images (union_masked/images/*.jpg
-- pixels outside GT-union-predicted-box zeroed, raw pixel values kept inside),
using a separate self-contained working directory so it doesn't collide with
the threshold=60 pipeline already in point_target_location/ and echo_signal/.

Unlike jpg_to_point_target.py, no additional intensity threshold is applied --
the union mask itself is the "thresholding method" here, so every nonzero
pixel inside the box is fed to gen_echo_signal as a scatterer.

azi_win_en is set to False for this run (per instruction).

Runtime is predicted the same way as gen_echo_signal_batch.py (~0.0462 s per
nonzero pixel, r=0.9999 fit); scenes whose predicted runtime exceeds
--max-seconds are skipped up front, with a subprocess timeout as a safety net.

Usage:
    python gen_echo_signal_union_batch.py --max-seconds 300
"""
import argparse
import json
import os
import subprocess
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNION_MASKED_IMAGES_DIR = os.path.join(SCRIPT_DIR, "union_masked", "images")

BASE = os.path.join(SCRIPT_DIR, "union_pipeline")
POINT_TARGET_DIR = os.path.join(BASE, "point_target_location")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
GEN_ECHO_SIGNAL_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "build", "gen_echo_signal"))
SKIPPED_LOG = os.path.join(BASE, "skipped_scenes.txt")
TIMING_LOG = os.path.join(BASE, "echo_signal_timing.csv")

SECONDS_PER_NONZERO_PIXEL = 0.0462  # measured across 99 threshold=60 scenes, r=0.9999

INPUT_PAR = {
    "wavelength_m": 0.1152,
    "pulse_width_sec": 1.25e-5,
    "pulse_rep_freq_hz": 1e3,
    "bandwidth_hz": 50e6,
    "sampling_freq_hz": 64e6,
    "closest_slant_range_m": 4e3,
    "height_m": 0.0,
    "azi_win_en": False,   # per instruction
    "rng_pad_time": 4,
    "noise_en": False,
    "snr_db": 25.0,
    "coherent_scatter_en": False,
}


def write_input_par():
    with open(os.path.join(BASE, "input_par.json"), "w") as f:
        json.dump(INPUT_PAR, f)


def build_point_target_jsons():
    os.makedirs(POINT_TARGET_DIR, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(UNION_MASKED_IMAGES_DIR) if f.endswith(".jpg"))
    for stem in stems:
        out_path = os.path.join(POINT_TARGET_DIR, stem + ".json")
        if os.path.exists(out_path):
            continue
        img = cv2.imread(os.path.join(UNION_MASKED_IMAGES_DIR, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        with open(out_path, "w") as f:
            json.dump(img.tolist(), f)
    return stems


def count_nonzero(target):
    with open(os.path.join(POINT_TARGET_DIR, target + ".json")) as f:
        data = json.load(f)
    return sum(1 for row in data for v in row if v != 0)


def gen_echo_signal(target, timeout):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [GEN_ECHO_SIGNAL_BIN, target],
        cwd=BASE,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"gen_echo_signal failed for {target}:\n{proc.stderr}")
    return elapsed


def log_timing(target, nz, elapsed):
    write_header = not os.path.exists(TIMING_LOG)
    with open(TIMING_LOG, "a") as f:
        if write_header:
            f.write("target,nonzero,elapsed_s\n")
        f.write(f"{target},{nz},{elapsed:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of echo signals to generate")
    ap.add_argument("--max-seconds", type=float, default=300,
                     help="skip scenes whose predicted runtime (from nonzero pixel count) exceeds this")
    args = ap.parse_args()

    os.makedirs(BASE, exist_ok=True)
    write_input_par()
    os.makedirs(ECHO_SIGNAL_DIR, exist_ok=True)

    all_stems = build_point_target_jsons()
    targets = all_stems[: args.n]
    todo = [t for t in targets if not os.path.exists(os.path.join(ECHO_SIGNAL_DIR, t + ".npy"))]
    already_done = len(targets) - len(todo)

    times = []
    skipped = []
    for target in todo:
        nz = count_nonzero(target)
        predicted = nz * SECONDS_PER_NONZERO_PIXEL
        if predicted > args.max_seconds:
            print(f"  {target}: SKIPPED (nonzero={nz}, predicted={predicted:.1f}s > "
                  f"--max-seconds={args.max_seconds})")
            skipped.append((target, nz, predicted))
            continue

        try:
            elapsed = gen_echo_signal(target, timeout=args.max_seconds * 2)
            times.append(elapsed)
            log_timing(target, nz, elapsed)
            print(f"  {target}: {elapsed:.3f} s (nonzero={nz}, predicted={predicted:.1f}s)")
        except subprocess.TimeoutExpired:
            print(f"  {target}: TIMED OUT after {args.max_seconds * 2:.0f}s "
                  f"(nonzero={nz}, predicted={predicted:.1f}s)")
            skipped.append((target, nz, predicted))

    if skipped:
        with open(SKIPPED_LOG, "a") as f:
            for target, nz, predicted in skipped:
                f.write(f"{target},{nz},{predicted:.1f}\n")

    total = sum(times)
    avg = total / len(times) if times else 0.0
    print(f"\n{already_done} already done, {len(times)} generated, {len(skipped)} skipped this run")
    print(f"Total (this run): {total:.3f} s over {len(times)} scene(s)  (avg {avg:.3f} s/scene)")
    if times:
        print(f"Timing logged -> {TIMING_LOG}")
    if skipped:
        print(f"Skipped scenes logged -> {SKIPPED_LOG}")


if __name__ == "__main__":
    main()
