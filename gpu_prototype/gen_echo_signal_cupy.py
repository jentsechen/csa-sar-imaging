#!/usr/bin/env python3
"""CuPy prototype of ImagingPar::gen_point_target_echo_signal (src/ImagingPar.cpp),
to evaluate GPU-acceleration feasibility for gen_echo_signal.cpp.

Reproduces the exact math from:
  - src/SigPar.cpp               (carrier_freq_hz, range_fm_rate_hz_s)
  - src/ImagingPar.cpp            (axes, calc_slant_range_m, apply_range_window,
                                   gen_point_target_echo_signal)
  - src/apps/gen_echo_signal.cpp  (point-target-list construction from the
                                   masked-image JSON, else-branch)

Assumes azi_win_en=False, noise_en=False, coherent_scatter_en=False, which is
the fixed configuration used everywhere in this project's pipeline.

Validates against a real C++ echo_signal .npy (generated via
end_to_end_pipeline/gen_echo_signal_union_batch.py, kept under
union_pipeline/reference/), then benchmarks NumPy
(CPU, small subset only) vs CuPy (GPU, full target list) for the identical
vectorized computation.

Superseded for production use by the CUDA kernel in sarsim/echo.py; kept as
the straightforward vectorized reference. See README.md in this directory for
the measured results. gen_echo_signal's
former data race (unprotected += into a shared output array across a
target-parallel loop) has since been fixed (parallelizes over azimuth row
instead), so the default multi-threaded build's output is a valid reference
again. Also mirrors ImagingPar's azi_pad_time / apply_azimuth_window (real
reference PRF + genuine azimuth zero-padding), added after the initial
prototype -- see git history.

Usage:
    python gen_echo_signal_cupy.py --target P0033_1800_2600_4200_5000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from sarsim.params import LIGHT_SPEED_M_S, build_imaging_axes, build_point_target_list  # noqa: E402

PIPE = os.path.join(REPO, "end_to_end_pipeline", "union_pipeline")


def gen_echo_signal(xp, ax, azimuth_offset_sec, range_offset_m, scatter_coef, deadline=None):
    """Vectorized (over i and j) port of ImagingPar::gen_point_target_echo_signal,
    for azi_win_en=False, noise_en=False, coherent_scatter_en=False. Includes the
    per-target apply_azimuth_window hard cutoff (see ImagingPar.cpp) -- without
    it, azi_pad_time padding creates no true zero margin.

    If `deadline` (a time.perf_counter() timestamp) is given, checks it once per
    target and bails out early (returning a partial, incomplete result) if
    exceeded. Returns (output, timed_out: bool)."""
    n_row, n_col = ax["n_row"], ax["n_col"]
    range_time_axis_sec = xp.asarray(ax["range_time_axis_sec"])
    azimuth_time_axis_sec = xp.asarray(ax["azimuth_time_axis_sec"])
    closest_ground_range_m = ax["closest_ground_range_m"]
    sensor_speed_m_s = ax["sensor_speed_m_s"]
    height_m = ax["height_m"]
    pulse_width_sec = ax["pulse_width_sec"]
    range_fm_rate_hz_s = ax["range_fm_rate_hz_s"]
    wavelength_m = ax["wavelength_m"]
    synthetic_aperture_time_sec = ax["synthetic_aperture_time_sec"]

    output = xp.zeros((n_row, n_col), dtype=xp.complex128)

    azimuth_offset_sec = xp.asarray(azimuth_offset_sec)
    range_offset_m = xp.asarray(range_offset_m)
    scatter_coef = xp.asarray(scatter_coef)

    n_targets = azimuth_offset_sec.shape[0]
    for t in range(n_targets):
        if deadline is not None and time.perf_counter() > deadline:
            return output, True
        az_off = azimuth_offset_sec[t]
        rg_off = range_offset_m[t]
        coef = scatter_coef[t]

        azimuth_rel_time = azimuth_time_axis_sec + az_off  # (n_row,)
        azimuth_window = (azimuth_rel_time < synthetic_aperture_time_sec / 2.0) & \
                          (azimuth_rel_time > -synthetic_aperture_time_sec / 2.0)

        ground_term = closest_ground_range_m + rg_off
        slant_range_m = xp.sqrt(
            height_m**2 + ground_term**2
            + (sensor_speed_m_s * (azimuth_time_axis_sec + az_off)) ** 2
        )  # shape (n_row,)
        round_trip_time_sec = 2.0 * slant_range_m / LIGHT_SPEED_M_S  # (n_row,)

        rel_time = range_time_axis_sec[None, :] - round_trip_time_sec[:, None]  # (n_row, n_col)
        window = (rel_time < pulse_width_sec / 2.0) & (rel_time > -pulse_width_sec / 2.0)
        window = window & azimuth_window[:, None]

        chirp_term = xp.exp(1j * xp.pi * range_fm_rate_hz_s * rel_time**2)  # (n_row, n_col)
        carrier_term = xp.exp(-1j * xp.pi * 4.0 * slant_range_m / wavelength_m)  # (n_row,)

        sample = chirp_term * carrier_term[:, None] * coef
        output += xp.where(window, sample, 0.0)

    return output, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="P0033_1800_2600_4200_5000",
                     help="stem under union_pipeline/reference/{point_target_location,echo_signal}")
    ap.add_argument("--cpu-subset", type=int, default=30,
                     help="number of targets for the single-core NumPy sanity check "
                          "(the full target list is too slow on 1 CPU core)")
    args = ap.parse_args()

    with open(f"{PIPE}/input_par.json") as f:
        par = json.load(f)
    with open(f"{PIPE}/reference/point_target_location/{args.target}.json") as f:
        mask = json.load(f)

    ax = build_imaging_axes(par)
    print(f"n_row={ax['n_row']} n_col={ax['n_col']}")

    az_off, rg_off, coef = build_point_target_list(
        mask, ax["n_row"], ax["n_col"], ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])
    n_targets = len(az_off)
    print(f"number of point targets: {n_targets}")

    cpp_path = f"{PIPE}/reference/echo_signal/{args.target}.npy"
    cpp_out = np.load(cpp_path)
    print(f"C++ reference output: {cpp_path}, shape={cpp_out.shape}, dtype={cpp_out.dtype}")
    if cpp_out.shape != (ax["n_row"], ax["n_col"]):
        cpp_out = cpp_out.reshape(ax["n_row"], ax["n_col"])

    # ---- Quick independent sanity check on a small subset (single CPU core) ----
    # Confirms the vectorized formula itself (independent of CuPy) matches the
    # C++ math before trusting the full GPU run below.
    n_subset = min(args.cpu_subset, n_targets)
    t0 = time.perf_counter()
    out_cpu_subset, _ = gen_echo_signal(np, ax, az_off[:n_subset], rg_off[:n_subset], coef[:n_subset])
    t_cpu_subset = time.perf_counter() - t0
    print(f"\nNumPy sanity check ({n_subset} targets, 1 core): {t_cpu_subset:.3f} s")

    # ---- GPU (CuPy), full target list ----
    import cupy as cp
    _ = cp.asarray(np.zeros(1)) + 1  # warm up context so timed run excludes CUDA init
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    out_gpu_subset, _ = gen_echo_signal(cp, ax, az_off[:n_subset], rg_off[:n_subset], coef[:n_subset])
    cp.cuda.Stream.null.synchronize()
    print(f"CuPy same {n_subset}-target subset: {time.perf_counter()-t0:.3f} s")
    subset_diff = np.max(np.abs(cp.asnumpy(out_gpu_subset) - out_cpu_subset))
    print(f"NumPy vs CuPy on subset, max abs diff = {subset_diff:.3e} "
          f"(match: {np.allclose(cp.asnumpy(out_gpu_subset), out_cpu_subset, rtol=1e-9, atol=1e-9)})")

    print(f"\nRunning full {n_targets}-target computation on GPU...")
    t0 = time.perf_counter()
    out_gpu, _ = gen_echo_signal(cp, ax, az_off, rg_off, coef)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0
    print(f"CuPy (GPU, full {n_targets} targets): {t_gpu:.3f} s")

    out_gpu_host = cp.asnumpy(out_gpu)
    max_abs = np.max(np.abs(cpp_out))
    diff = np.max(np.abs(out_gpu_host - cpp_out))
    rel_diff = diff / max_abs if max_abs > 0 else diff
    print(f"\nCuPy vs C++ reference: max abs diff = {diff:.3e} (max |C++|={max_abs:.3e}, rel={rel_diff:.3e})")
    print(f"CuPy matches C++ output (rtol=1e-6, atol=1e-6): "
          f"{np.allclose(out_gpu_host, cpp_out, rtol=1e-6, atol=1e-6)}")
    print(f"\nCuPy (GPU) time: {t_gpu:.3f} s")


if __name__ == "__main__":
    main()
