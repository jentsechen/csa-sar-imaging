#!/usr/bin/env python3
"""CuPy (GPU) counterpart of gen_echo_signal_union_batch.py: same INPUT_PAR,
same union_masked/images/*.jpg -> point_target_location/*.json construction,
same union_pipeline/ working directory and echo_signal/*.npy output format --
only the echo-signal computation itself runs on the GPU via
gpu_prototype/gen_echo_signal_cuda.py's hand-written CUDA kernel (default,
--backend kernel) or gpu_prototype/gen_echo_signal_cupy.py's vectorized CuPy
version (--backend cupy) instead of shelling out to the C++ gen_echo_signal
binary.

Per-image wall-clock time cap: if a scene doesn't finish within --max-seconds,
it is aborted (partial/incomplete result discarded) and logged to
skipped_scenes.txt instead of being written to echo_signal/.

Usage:
    python gen_echo_signal_union_batch_cupy.py --max-seconds 120
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
from gen_echo_signal_cupy import build_imaging_axes, build_point_target_list, gen_echo_signal  # noqa: E402
from gen_echo_signal_cuda import gen_echo_signal_cuda  # noqa: E402

UNION_MASKED_IMAGES_DIR = os.path.join(SCRIPT_DIR, "union_masked", "images")

BASE = os.path.join(SCRIPT_DIR, "union_pipeline")
POINT_TARGET_DIR = os.path.join(BASE, "point_target_location")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
SKIPPED_LOG = os.path.join(BASE, "skipped_scenes.txt")
TIMING_LOGS = {
    "kernel": os.path.join(BASE, "echo_signal_timing_kernel.csv"),
    "cupy": os.path.join(BASE, "echo_signal_timing_cupy.csv"),
}

# Same parameter set as gen_echo_signal_union_batch.py -- see that file's
# docstring for the azi_pad_time / real-PRF rationale.
INPUT_PAR = {
    "wavelength_m": 0.0555042,
    "pulse_width_sec": 11.99e-6,
    "pulse_rep_freq_hz": 1662.0375,
    "bandwidth_hz": 50e6,
    "sampling_freq_hz": 66.728e6,
    "closest_slant_range_m": 800e3,
    "height_m": 0.0,
    "azi_win_en": False,
    "rng_pad_time": 4,
    "noise_en": False,
    "snr_db": 25.0,
    "coherent_scatter_en": False,
    "sensor_speed_m_s": 7500,
    "azimuth_aperture_len_m": 12.3,
    "azi_pad_time": 4,
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


def log_timing(timing_log, target, n_targets, elapsed, timed_out):
    write_header = not os.path.exists(timing_log)
    with open(timing_log, "a") as f:
        if write_header:
            f.write("target,n_targets,elapsed_s,timed_out\n")
        f.write(f"{target},{n_targets},{elapsed:.3f},{timed_out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="number of echo signals to generate")
    ap.add_argument("--max-seconds", type=float, default=120,
                     help="per-scene wall-clock cap; scenes exceeding this are aborted and logged as skipped")
    ap.add_argument("--backend", choices=["kernel", "cupy"], default="kernel",
                     help="kernel: hand-written CUDA kernel (fast); cupy: vectorized CuPy reference")
    ap.add_argument("--overwrite", action="store_true",
                     help="regenerate scenes even if echo_signal/<target>.npy already exists")
    ap.add_argument("--no-save", action="store_true",
                     help="benchmark only: compute every scene but write no .npy / timing log "
                          "(implies --overwrite)")
    args = ap.parse_args()

    os.makedirs(BASE, exist_ok=True)
    write_input_par()
    os.makedirs(ECHO_SIGNAL_DIR, exist_ok=True)

    import cupy as cp
    _ = cp.asarray(np.zeros(1)) + 1  # warm up CUDA context
    cp.cuda.Stream.null.synchronize()

    ax = build_imaging_axes(INPUT_PAR)
    print(f"n_row={ax['n_row']} n_col={ax['n_col']}", flush=True)

    all_stems = build_point_target_jsons()
    targets = all_stems[: args.n]
    todo = [t for t in targets
            if args.overwrite or args.no_save or not os.path.exists(os.path.join(ECHO_SIGNAL_DIR, t + ".npy"))]
    timing_log = TIMING_LOGS[args.backend]
    already_done = len(targets) - len(todo)

    print(f"{already_done} already done, {len(todo)} to process", flush=True)

    times = []
    skipped = []
    run_t0 = time.perf_counter()
    for idx, target in enumerate(todo):
        with open(os.path.join(POINT_TARGET_DIR, target + ".json")) as f:
            mask = json.load(f)
        az_off, rg_off, coef = build_point_target_list(
            mask, ax["n_row"], ax["n_col"], ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])
        n_targets = len(az_off)

        t0 = time.perf_counter()
        deadline = t0 + args.max_seconds
        if args.backend == "kernel":
            out, timed_out = gen_echo_signal_cuda(ax, az_off, rg_off, coef, deadline=deadline)
        else:
            out, timed_out = gen_echo_signal(cp, ax, az_off, rg_off, coef, deadline=deadline)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        if timed_out:
            print(f"  [{idx+1}/{len(todo)}] {target}: TIMED OUT after {elapsed:.1f}s "
                  f"(n_targets={n_targets}, cap={args.max_seconds}s)", flush=True)
            skipped.append((target, n_targets, elapsed))
            if not args.no_save:
                log_timing(timing_log, target, n_targets, elapsed, True)
        else:
            if not args.no_save:
                out_host = cp.asnumpy(out)
                np.save(os.path.join(ECHO_SIGNAL_DIR, target + ".npy"), out_host)
                log_timing(timing_log, target, n_targets, elapsed, False)
            times.append(elapsed)
            if (idx + 1) % 20 == 0 or idx == 0:
                cum = time.perf_counter() - run_t0
                print(f"  [{idx+1}/{len(todo)}] {target}: {elapsed:.2f}s (n_targets={n_targets}) "
                      f"| cumulative {cum/60:.1f} min", flush=True)

        # Defensive periodic cleanup across a very long run.
        if (idx + 1) % 50 == 0:
            cp.get_default_memory_pool().free_all_blocks()

    if skipped and not args.no_save:
        write_header = not os.path.exists(SKIPPED_LOG)
        with open(SKIPPED_LOG, "a") as f:
            if write_header:
                f.write("target,n_targets,elapsed_s\n")
            for target, n_targets, elapsed in skipped:
                f.write(f"{target},{n_targets},{elapsed:.1f}\n")

    total = sum(times)
    avg = total / len(times) if times else 0.0
    wall = time.perf_counter() - run_t0
    print(f"\n{already_done} already done, {len(times)} generated, {len(skipped)} timed out this run", flush=True)
    print(f"Total GPU compute time (this run): {total:.1f} s over {len(times)} scene(s) (avg {avg:.2f} s/scene)", flush=True)
    print(f"Total wall time (this run): {wall:.1f} s ({wall/60:.1f} min)", flush=True)
    if times and not args.no_save:
        print(f"Timing logged -> {timing_log}", flush=True)
    if skipped:
        print(f"Skipped/timed-out scenes logged -> {SKIPPED_LOG}", flush=True)


if __name__ == "__main__":
    main()
