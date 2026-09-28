#!/usr/bin/env python3
"""End-to-end GPU batch: point-target JSON -> echo (CUDA kernel) -> one or more
imaging algorithms -> scene crop -> JPG + float32 linear-power crop + metrics.

The 164 MB echo never leaves the GPU: it is regenerated on demand (~0.14 s)
rather than stored, since /home is a slow SMR disk. Each echo is generated
once per scene and shared by every selected algorithm, so all algorithms see
bit-identical input.

Output layout (under --out-root, default end_to_end_pipeline/union_pipeline/runs):
    <run>/manifest.json                 input_par, git commit, algorithms, formats
    <run>/metrics.csv                   stem, algo, n_targets, echo_s, focus_s, peak_db, entropy
    <run>/<algo>/jpg/<stem>.jpg         8-bit, 30 dB dynamic range (YOLO input)
    <run>/<algo>/power/<stem>.npy       float32 linear |x|^2 of the scene crop
                                        (same orientation as the JPG / labels)

Usage:
    python -m sarsim.pipeline --run baseline --algo csa
    python -m sarsim.pipeline --run cmp --algo csa,rda --n 50
    python -m sarsim.pipeline --run dbg --save-echo P0033_1800_2600_4200_5000
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNION = os.path.join(REPO, "end_to_end_pipeline", "union_pipeline")


def exclude_cpus(spec):
    """Pin this process away from the given CPUs (default: CPU 1, a faulty core on
    this workstation that causes random segfaults). Must run before cupy loads."""
    bad = {int(c) for c in spec.split(",") if c.strip()}
    if bad:
        os.sched_setaffinity(0, os.sched_getaffinity(0) - bad)


def git_commit():
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name (output subdirectory)")
    ap.add_argument("--algo", default="csa", help="comma-separated imaging algorithms")
    ap.add_argument("--n", type=int, default=None, help="only the first N scenes (sorted)")
    ap.add_argument("--input-par", default=os.path.join(UNION, "input_par.json"))
    ap.add_argument("--point-target-dir", default=os.path.join(UNION, "point_target_location"))
    ap.add_argument("--out-root", default=os.path.join(UNION, "runs"))
    ap.add_argument("--no-power", action="store_true", help="skip the float32 power crops")
    ap.add_argument("--save-echo", nargs="*", default=[], metavar="STEM",
                    help="also write the full echo .npy for these scenes (debugging; 164 MB each)")
    ap.add_argument("--overwrite", action="store_true", help="redo scenes whose outputs already exist")
    ap.add_argument("--exclude-cpus", default="1", help="CPUs to avoid ('' for none)")
    args = ap.parse_args()

    exclude_cpus(args.exclude_cpus)
    import cupy as cp
    import cv2
    from .echo import gen_echo_signal
    from .imaging import ALGORITHMS
    from .params import build_imaging_axes, build_point_target_list, load_input_par
    from .postproc import process

    algos = [a.strip() for a in args.algo.split(",") if a.strip()]
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        sys.exit(f"unknown algorithm(s) {unknown}; available: {sorted(ALGORITHMS)}")

    par = load_input_par(args.input_par)
    ax = build_imaging_axes(par)
    run_dir = os.path.join(args.out_root, args.run)
    for a in algos:
        os.makedirs(os.path.join(run_dir, a, "jpg"), exist_ok=True)
        if not args.no_power:
            os.makedirs(os.path.join(run_dir, a, "power"), exist_ok=True)

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "algorithms": algos,
        "input_par": par,
        "point_target_dir": os.path.relpath(args.point_target_dir, REPO),
        "grid": [ax["n_row"], ax["n_col"]],
        "outputs": {
            "jpg": "uint8, 30 dB dynamic range below scene-crop peak, flipped up-down",
            "power": "float32 linear |x|^2 of the scene crop, flipped up-down (same as jpg)",
        },
    }
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        if old["input_par"] != par:
            sys.exit(f"{manifest_path} was made with a different input_par; use a new --run name")
        manifest["created"] = old["created"]
        manifest["algorithms"] = sorted(set(old["algorithms"]) | set(algos))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.point_target_dir) if f.endswith(".json"))
    if args.n is not None:
        stems = stems[: args.n]

    def done(stem):
        return all(os.path.exists(os.path.join(run_dir, a, "jpg", stem + ".jpg")) for a in algos)
    todo = stems if args.overwrite else [s for s in stems if not done(s)]
    print(f"run {run_dir}: algos={algos}, grid={ax['n_row']}x{ax['n_col']}, "
          f"{len(stems) - len(todo)} done, {len(todo)} to process", flush=True)

    imagers = {a: ALGORITHMS[a](cp, ax, par) for a in algos}
    metrics_path = os.path.join(run_dir, "metrics.csv")
    new_file = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    writer = csv.writer(metrics_f)
    if new_file:
        writer.writerow(["stem", "algo", "n_targets", "echo_s", "focus_s", "peak_db", "entropy"])

    failed = []
    run_t0 = time.perf_counter()
    for idx, stem in enumerate(todo):
        try:
            with open(os.path.join(args.point_target_dir, stem + ".json")) as f:
                mask = np.asarray(json.load(f))
            az_off, rg_off, coef = build_point_target_list(
                mask, ax["n_row"], ax["n_col"], ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])

            t0 = time.perf_counter()
            echo, _ = gen_echo_signal(ax, az_off, rg_off, coef)
            cp.cuda.Stream.null.synchronize()
            echo_s = time.perf_counter() - t0
            if stem in args.save_echo:
                np.save(os.path.join(run_dir, stem + "_echo.npy"), cp.asnumpy(echo))

            for a, imager in imagers.items():
                t0 = time.perf_counter()
                focused = imager.focus(echo)
                gray, power, m = process(cp, focused, ax["n_row"], ax["n_col"], mask.shape)
                focus_s = time.perf_counter() - t0  # process() copies to host -> synchronized
                del focused
                cv2.imwrite(os.path.join(run_dir, a, "jpg", stem + ".jpg"), gray)
                if not args.no_power:
                    np.save(os.path.join(run_dir, a, "power", stem + ".npy"), power)
                writer.writerow([stem, a, len(az_off), f"{echo_s:.4f}", f"{focus_s:.4f}",
                                 f"{m['peak_db']:.4f}", f"{m['entropy']:.6f}"])
            del echo
            metrics_f.flush()
        except Exception:
            traceback.print_exc()
            failed.append(stem)
            continue
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(todo)}] {stem} | {(time.perf_counter() - run_t0) / 60:.1f} min", flush=True)

    metrics_f.close()
    wall = time.perf_counter() - run_t0
    print(f"\n{len(todo) - len(failed)} scene(s) in {wall:.1f} s ({wall / 60:.1f} min)", flush=True)
    if failed:
        with open(os.path.join(run_dir, "failed.txt"), "a") as f:
            f.write("".join(s + "\n" for s in failed))
        print(f"{len(failed)} failed -> {run_dir}/failed.txt", flush=True)


if __name__ == "__main__":
    main()
