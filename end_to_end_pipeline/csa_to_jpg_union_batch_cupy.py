#!/usr/bin/env python3
"""GPU counterpart of csa_to_jpg_union_batch.py: CSA focusing + magnitude in dB
+ crop / 30 dB clip / normalize -> grayscale JPG, all on the GPU via
gpu_prototype/csa_cupy.py (cuFFT) instead of shelling out to TestMultiPointTarget.

Two input modes:
  --source point_target (default, fused): point_target_location/<stem>.json
      -> CUDA echo kernel (gpu_prototype/gen_echo_signal_cuda.py) -> CSA -> JPG.
      The 164 MB echo signal never leaves the GPU, so nothing large touches
      the disk (writing/reading echo_signal/*.npy is disk-bound on /home).
  --source echo_npy: reads union_pipeline/echo_signal/<stem>.npy instead.

--save-focused additionally writes focused_image/<stem>.npy and
<stem>_mag_db.npy like the C++ path (large, slow on this disk).

Usage:
    python csa_to_jpg_union_batch_cupy.py                 # -> union_pipeline/csa_jpg_gpu/
    python csa_to_jpg_union_batch_cupy.py --out-dir csa_jpg
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "gpu_prototype"))
from gen_echo_signal_cupy import build_imaging_axes, build_point_target_list  # noqa: E402
from gen_echo_signal_cuda import gen_echo_signal_cuda  # noqa: E402
from csa_cupy import ChirpScalingAlgo, calc_mag_db  # noqa: E402

BASE = os.path.join(SCRIPT_DIR, "union_pipeline")
POINT_TARGET_DIR = os.path.join(BASE, "point_target_location")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
FOCUSED_IMAGE_DIR = os.path.join(BASE, "focused_image")

# Same crop / dynamic range as csa_to_jpg_union_batch.py.
N_ROW, N_COL = 800, 800
DYNAMIC_RANGE_DB = 30


def mag_db_crop_to_gray(data):
    """Identical to csa_to_jpg_union_batch.mag_db_to_gray_jpg after its crop."""
    max_val = np.max(data)
    clipped = np.clip(data, max_val - DYNAMIC_RANGE_DB, max_val)
    gray = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.flipud(gray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="number of scenes to process")
    ap.add_argument("--source", choices=["point_target", "echo_npy"], default="point_target")
    ap.add_argument("--out-dir", default="csa_jpg_gpu", help="JPG output dir under union_pipeline/")
    ap.add_argument("--save-focused", action="store_true",
                    help="also write focused_image/<stem>.npy and <stem>_mag_db.npy")
    args = ap.parse_args()

    import cupy as cp

    out_dir = os.path.join(BASE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(BASE, "input_par.json")) as f:
        par = json.load(f)
    ax = build_imaging_axes(par)
    csa = ChirpScalingAlgo(cp, ax, par)

    src_dir, ext = ((POINT_TARGET_DIR, ".json") if args.source == "point_target"
                    else (ECHO_SIGNAL_DIR, ".npy"))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(src_dir) if f.endswith(ext))[: args.n]
    todo = [s for s in stems if not os.path.exists(os.path.join(out_dir, s + ".jpg"))]
    print(f"{len(stems) - len(todo)} already done, {len(todo)} to process -> {out_dir}", flush=True)

    r0, r1 = int(N_ROW * 3 / 2), int(N_ROW * 5 / 2)
    c0, c1 = int(N_COL * 3 / 2), int(N_COL * 5 / 2)
    gpu_times = []
    run_t0 = time.perf_counter()
    for idx, stem in enumerate(todo):
        if args.source == "point_target":
            with open(os.path.join(POINT_TARGET_DIR, stem + ".json")) as f:
                mask = json.load(f)
            az_off, rg_off, coef = build_point_target_list(
                mask, ax["n_row"], ax["n_col"], ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])
            t0 = time.perf_counter()
            echo, _ = gen_echo_signal_cuda(ax, az_off, rg_off, coef)
        else:
            echo_host = np.load(os.path.join(ECHO_SIGNAL_DIR, stem + ".npy"))
            t0 = time.perf_counter()
            echo = cp.asarray(echo_host)

        focused = csa.apply_csa(echo)
        del echo
        if args.save_focused:
            mag_db = calc_mag_db(cp, focused)
            crop = cp.asnumpy(mag_db[r0:r1, c0:c1])
        else:
            crop = cp.asnumpy(calc_mag_db(cp, focused[r0:r1, c0:c1]))
        gpu_times.append(time.perf_counter() - t0)  # asnumpy synchronizes

        cv2.imwrite(os.path.join(out_dir, stem + ".jpg"), mag_db_crop_to_gray(crop))
        if args.save_focused:
            os.makedirs(FOCUSED_IMAGE_DIR, exist_ok=True)
            np.save(os.path.join(FOCUSED_IMAGE_DIR, stem + ".npy"), cp.asnumpy(focused))
            np.save(os.path.join(FOCUSED_IMAGE_DIR, stem + "_mag_db.npy"), cp.asnumpy(mag_db))
        del focused

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(todo)}] {stem}: {gpu_times[-1]:.2f}s "
                  f"| cumulative {(time.perf_counter() - run_t0) / 60:.1f} min", flush=True)

    wall = time.perf_counter() - run_t0
    if gpu_times:
        print(f"\n{len(gpu_times)} scene(s): GPU {sum(gpu_times):.1f} s "
              f"(avg {np.mean(gpu_times):.3f} s/scene), wall {wall:.1f} s ({wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
