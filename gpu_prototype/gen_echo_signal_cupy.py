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
end_to_end_pipeline/gen_echo_signal_union_batch.py), then benchmarks NumPy
(CPU, small subset only) vs CuPy (GPU, full target list) for the identical
vectorized computation.

See README.md in this directory for the measured results and an important
correctness caveat: gen_echo_signal's default multi-threaded OpenMP build has
a data race (unprotected += into a shared output array across the
parallel-for target loop), so only compare against a single-threaded
(OMP_NUM_THREADS=1) reference run, not the default build's output.

Usage:
    python gen_echo_signal_cupy.py --target P0033_1800_2600_4200_5000
"""
import argparse
import json
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(REPO, "end_to_end_pipeline", "union_pipeline")

LIGHT_SPEED_M_S = 3e8


def build_imaging_axes(par):
    wavelength_m = par["wavelength_m"]
    pulse_width_sec = par["pulse_width_sec"]
    pulse_rep_freq_hz = par["pulse_rep_freq_hz"]
    sampling_freq_hz = par["sampling_freq_hz"]
    closest_slant_range_m = par["closest_slant_range_m"]
    height_m = par["height_m"]
    sensor_speed_m_s = par.get("sensor_speed_m_s", 120.0)
    azimuth_aperture_len_m = par.get("azimuth_aperture_len_m", 1.2)
    rng_pad_time = par["rng_pad_time"]

    range_fm_rate_hz_s = par["bandwidth_hz"] / pulse_width_sec

    closest_ground_range_m = np.sqrt(closest_slant_range_m**2 - height_m**2)
    beamwidth_rad = wavelength_m / azimuth_aperture_len_m
    synthetic_aperture_len_m = beamwidth_rad * closest_slant_range_m
    synthetic_aperture_time_sec = synthetic_aperture_len_m / sensor_speed_m_s

    n_col = int(np.floor(rng_pad_time * pulse_width_sec * sampling_freq_hz / 2)) * 2
    range_time_axis_sec = (np.arange(n_col) - n_col // 2) / sampling_freq_hz + \
        2.0 * closest_slant_range_m / LIGHT_SPEED_M_S

    n_row = int(np.floor(synthetic_aperture_time_sec * pulse_rep_freq_hz / 2)) * 2
    azimuth_time_axis_sec = (np.arange(n_row) - n_row // 2) / pulse_rep_freq_hz

    return dict(
        wavelength_m=wavelength_m, pulse_width_sec=pulse_width_sec,
        pulse_rep_freq_hz=pulse_rep_freq_hz, sampling_freq_hz=sampling_freq_hz,
        range_fm_rate_hz_s=range_fm_rate_hz_s,
        closest_slant_range_m=closest_slant_range_m, height_m=height_m,
        closest_ground_range_m=closest_ground_range_m,
        sensor_speed_m_s=sensor_speed_m_s,
        n_row=n_row, n_col=n_col,
        range_time_axis_sec=range_time_axis_sec,
        azimuth_time_axis_sec=azimuth_time_axis_sec,
    )


def build_point_target_list(mask, n_row, n_col, pulse_rep_freq_hz, sampling_freq_hz):
    """Mirrors gen_echo_signal.cpp's else-branch indexing exactly."""
    mask = np.asarray(mask)
    ii, jj = np.nonzero(mask > 0)
    scatter = mask[ii, jj].astype(np.float64)

    azimuth_offset_sec = ((ii + n_row * 3 // 8) - n_row / 2.0) / pulse_rep_freq_hz
    range_offset_m = ((jj + n_col * 3 // 8) - n_col / 2.0) * LIGHT_SPEED_M_S / 2.0 / sampling_freq_hz
    return azimuth_offset_sec, range_offset_m, scatter


def gen_echo_signal(xp, ax, azimuth_offset_sec, range_offset_m, scatter_coef):
    """Vectorized (over i and j) port of ImagingPar::gen_point_target_echo_signal,
    for azi_win_en=False, noise_en=False, coherent_scatter_en=False."""
    n_row, n_col = ax["n_row"], ax["n_col"]
    range_time_axis_sec = xp.asarray(ax["range_time_axis_sec"])
    azimuth_time_axis_sec = xp.asarray(ax["azimuth_time_axis_sec"])
    closest_ground_range_m = ax["closest_ground_range_m"]
    sensor_speed_m_s = ax["sensor_speed_m_s"]
    height_m = ax["height_m"]
    pulse_width_sec = ax["pulse_width_sec"]
    range_fm_rate_hz_s = ax["range_fm_rate_hz_s"]
    wavelength_m = ax["wavelength_m"]

    output = xp.zeros((n_row, n_col), dtype=xp.complex128)

    azimuth_offset_sec = xp.asarray(azimuth_offset_sec)
    range_offset_m = xp.asarray(range_offset_m)
    scatter_coef = xp.asarray(scatter_coef)

    n_targets = azimuth_offset_sec.shape[0]
    for t in range(n_targets):
        az_off = azimuth_offset_sec[t]
        rg_off = range_offset_m[t]
        coef = scatter_coef[t]

        ground_term = closest_ground_range_m + rg_off
        slant_range_m = xp.sqrt(
            height_m**2 + ground_term**2
            + (sensor_speed_m_s * (azimuth_time_axis_sec + az_off)) ** 2
        )  # shape (n_row,)
        round_trip_time_sec = 2.0 * slant_range_m / LIGHT_SPEED_M_S  # (n_row,)

        rel_time = range_time_axis_sec[None, :] - round_trip_time_sec[:, None]  # (n_row, n_col)
        window = (rel_time < pulse_width_sec / 2.0) & (rel_time > -pulse_width_sec / 2.0)

        chirp_term = xp.exp(1j * xp.pi * range_fm_rate_hz_s * rel_time**2)  # (n_row, n_col)
        carrier_term = xp.exp(-1j * xp.pi * 4.0 * slant_range_m / wavelength_m)  # (n_row,)

        sample = chirp_term * carrier_term[:, None] * coef
        output += xp.where(window, sample, 0.0)

    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="P0033_1800_2600_4200_5000",
                     help="stem under union_pipeline/point_target_location and echo_signal")
    ap.add_argument("--cpu-subset", type=int, default=30,
                     help="number of targets for the single-core NumPy sanity check "
                          "(the full target list is too slow on 1 CPU core)")
    args = ap.parse_args()

    with open(f"{PIPE}/input_par.json") as f:
        par = json.load(f)
    with open(f"{PIPE}/point_target_location/{args.target}.json") as f:
        mask = json.load(f)

    ax = build_imaging_axes(par)
    print(f"n_row={ax['n_row']} n_col={ax['n_col']}")

    az_off, rg_off, coef = build_point_target_list(
        mask, ax["n_row"], ax["n_col"], ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])
    n_targets = len(az_off)
    print(f"number of point targets: {n_targets}")

    cpp_path = f"{PIPE}/echo_signal/{args.target}.npy"
    cpp_out = np.load(cpp_path)
    print(f"C++ reference output: {cpp_path}, shape={cpp_out.shape}, dtype={cpp_out.dtype}")
    print("NOTE: this must be a single-threaded (OMP_NUM_THREADS=1) run -- the default "
          "multi-threaded build has a known data race (see README.md).")
    if cpp_out.shape != (ax["n_row"], ax["n_col"]):
        cpp_out = cpp_out.reshape(ax["n_row"], ax["n_col"])

    # ---- Quick independent sanity check on a small subset (single CPU core) ----
    # Confirms the vectorized formula itself (independent of CuPy) matches the
    # C++ math before trusting the full GPU run below.
    n_subset = min(args.cpu_subset, n_targets)
    t0 = time.perf_counter()
    out_cpu_subset = gen_echo_signal(np, ax, az_off[:n_subset], rg_off[:n_subset], coef[:n_subset])
    t_cpu_subset = time.perf_counter() - t0
    print(f"\nNumPy sanity check ({n_subset} targets, 1 core): {t_cpu_subset:.3f} s")

    # ---- GPU (CuPy), full target list ----
    import cupy as cp
    _ = cp.asarray(np.zeros(1)) + 1  # warm up context so timed run excludes CUDA init
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    out_gpu_subset = gen_echo_signal(cp, ax, az_off[:n_subset], rg_off[:n_subset], coef[:n_subset])
    cp.cuda.Stream.null.synchronize()
    print(f"CuPy same {n_subset}-target subset: {time.perf_counter()-t0:.3f} s")
    subset_diff = np.max(np.abs(cp.asnumpy(out_gpu_subset) - out_cpu_subset))
    print(f"NumPy vs CuPy on subset, max abs diff = {subset_diff:.3e} "
          f"(match: {np.allclose(cp.asnumpy(out_gpu_subset), out_cpu_subset, rtol=1e-9, atol=1e-9)})")

    print(f"\nRunning full {n_targets}-target computation on GPU...")
    t0 = time.perf_counter()
    out_gpu = gen_echo_signal(cp, ax, az_off, rg_off, coef)
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
