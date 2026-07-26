#!/usr/bin/env python3
"""Batch-generate echo signals from point_target_location/*.json via the
`gen_echo_signal` C++ binary, for timing/throughput measurement.

Reads N point-target scenes (default 10) from point_target_location/, writes
input_par.json (HRSID sensor params, same as used for P0002_1800_2600_2400_3200
in TestMultiPointTarget/TestMultiPointTarget.py), then runs
`../build/gen_echo_signal <target>` once per scene -> echo_signal/<target>.npy.
Reports per-target and total elapsed time.

Runtime scales with the number of nonzero (post-threshold) scatterer pixels
in the point-target JSON -- measured at ~0.0475 s per scatterer pixel across
10 sample scenes (46-49 s/1000px, consistently). A few scenes keep far more
bright pixels after thresholding (e.g. 32,168 vs the usual ~250-1500) and
would take 20+ minutes each, so scenes whose predicted runtime exceeds
--max-seconds are skipped up front, and a subprocess timeout guards against
the estimate being wrong for some other reason.

Usage:
    python gen_echo_signal_batch.py --n 10
    python gen_echo_signal_batch.py --n 100 --max-seconds 120
"""
import argparse
import json
import os
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
POINT_TARGET_DIR = os.path.join(BASE, "point_target_location")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
GEN_ECHO_SIGNAL_BIN = os.path.join(BASE, "..", "build", "gen_echo_signal")
SKIPPED_LOG = os.path.join(BASE, "skipped_scenes.txt")
TIMING_LOG = os.path.join(BASE, "echo_signal_timing.csv")

SECONDS_PER_NONZERO_PIXEL = 0.0475  # measured across 10 sample scenes, ~46-49 s/1000px

INPUT_PAR = {
    "wavelength_m": 0.1152,
    "pulse_width_sec": 1.25e-5,
    "pulse_rep_freq_hz": 1e3,
    "bandwidth_hz": 50e6,
    "sampling_freq_hz": 64e6,
    "closest_slant_range_m": 4e3,
    "height_m": 0.0,
    "azi_win_en": True,
    "rng_pad_time": 4,
    "noise_en": False,
    "snr_db": 25.0,
    "coherent_scatter_en": False,
}


def write_input_par():
    with open(os.path.join(BASE, "input_par.json"), "w") as f:
        json.dump(INPUT_PAR, f)


def list_targets(n):
    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(POINT_TARGET_DIR)
        if f.endswith(".json")
    )
    return stems[:n]


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
    ap.add_argument("--n", type=int, default=10, help="number of echo signals to generate")
    ap.add_argument("--max-seconds", type=float, default=120,
                     help="skip scenes whose predicted runtime (from scatterer count) exceeds this")
    args = ap.parse_args()

    write_input_par()
    os.makedirs(ECHO_SIGNAL_DIR, exist_ok=True)

    targets = list_targets(args.n)
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
