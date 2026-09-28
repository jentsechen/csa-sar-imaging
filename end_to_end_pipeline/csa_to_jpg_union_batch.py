#!/usr/bin/env python3
"""Run CSA focusing on union_pipeline/echo_signal/*.npy, compute magnitude in
dB, and convert to plain grayscale JPGs -- the union-mask counterpart of
csa_to_jpg_batch.py, kept in its own self-contained union_pipeline/ directory.

Pipeline per target:
    union_pipeline/echo_signal/<target>.npy
      -> [TestMultiPointTarget focus]     -> union_pipeline/focused_image/<target>.npy
      -> [TestMultiPointTarget calc_mag]  -> union_pipeline/focused_image/<target>_mag_db.npy
      -> [crop + 30dB clip + normalize]   -> union_pipeline/csa_jpg/<target>.jpg

Usage:
    python csa_to_jpg_union_batch.py --n 100
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "union_pipeline")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
FOCUSED_IMAGE_DIR = os.path.join(BASE, "focused_image")
CSA_JPG_DIR = os.path.join(BASE, "csa_jpg")
TEST_MULTI_POINT_TARGET_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "build", "TestMultiPointTarget"))

N_ROW, N_COL = 800, 800
DYNAMIC_RANGE_DB = 30


def list_targets(n):
    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(ECHO_SIGNAL_DIR)
        if f.endswith(".npy")
    )
    return stems[:n]


MAX_SUBPROCESS_RETRIES = 8  # TestMultiPointTarget intermittently segfaults (~15-20% per
                            # call, not a race -- reproduces even with OMP_NUM_THREADS=1;
                            # likely marginal hardware). Retrying the same call in-process
                            # avoids restarting this whole script (and its directory listing)
                            # on every crash.


def run_cpp(mode, arg):
    last_stderr = ""
    for attempt in range(MAX_SUBPROCESS_RETRIES):
        proc = subprocess.run(
            [TEST_MULTI_POINT_TARGET_BIN, mode, arg],
            cwd=BASE,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return
        last_stderr = proc.stderr
        print(f"    [retry] TestMultiPointTarget {mode} {arg} exit={proc.returncode} "
              f"(attempt {attempt+1}/{MAX_SUBPROCESS_RETRIES})", flush=True)
    raise RuntimeError(f"TestMultiPointTarget {mode} {arg} failed after "
                       f"{MAX_SUBPROCESS_RETRIES} attempts:\n{last_stderr}")


def mag_db_to_gray_jpg(target):
    mag_db_path = os.path.join(FOCUSED_IMAGE_DIR, target + "_mag_db.npy")
    data = np.load(mag_db_path)[
        int(N_ROW * 3 / 2): int(N_ROW * 5 / 2),
        int(N_COL * 3 / 2): int(N_COL * 5 / 2),
    ]

    max_val = np.max(data)
    clipped = np.clip(data, max_val - DYNAMIC_RANGE_DB, max_val)
    gray = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = np.flipud(gray)

    out_path = os.path.join(CSA_JPG_DIR, target + ".jpg")
    cv2.imwrite(out_path, gray)


def process_target(target):
    t0 = time.perf_counter()
    run_cpp("focus", target)
    t_focus = time.perf_counter()
    run_cpp("calc_mag", f"./focused_image/{target}")
    t_calc_mag = time.perf_counter()
    mag_db_to_gray_jpg(target)
    t_jpg = time.perf_counter()

    return {
        "focus_s": t_focus - t0,
        "calc_mag_s": t_calc_mag - t_focus,
        "jpg_s": t_jpg - t_calc_mag,
        "total_s": t_jpg - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of scenes to process")
    args = ap.parse_args()

    os.makedirs(FOCUSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(CSA_JPG_DIR, exist_ok=True)

    targets = list_targets(args.n)
    todo = [t for t in targets if not os.path.exists(os.path.join(CSA_JPG_DIR, t + ".jpg"))]
    skipped = len(targets) - len(todo)
    print(f"Processing {len(todo)} scene(s) ({skipped} already done, skipped)...")

    totals = []
    failed = []
    for target in todo:
        try:
            timing = process_target(target)
        except RuntimeError as e:
            print(f"  {target}: FAILED after retries, skipping -- {e}", flush=True)
            failed.append(target)
            continue
        totals.append(timing["total_s"])
        print(f"  {target}: focus={timing['focus_s']:.3f}s "
              f"calc_mag={timing['calc_mag_s']:.3f}s jpg={timing['jpg_s']:.3f}s "
              f"total={timing['total_s']:.3f}s")

    total = sum(totals)
    avg = total / len(totals) if totals else 0.0
    print(f"\nTotal (this run): {total:.3f} s over {len(totals)} scene(s)  (avg {avg:.3f} s/scene)")
    if failed:
        failed_log = os.path.join(BASE, "csa_failed_scenes.txt")
        with open(failed_log, "a") as f:
            for t in failed:
                f.write(t + "\n")
        print(f"{len(failed)} scene(s) failed after {MAX_SUBPROCESS_RETRIES} retries each -> {failed_log}")


if __name__ == "__main__":
    main()
