#!/usr/bin/env python3
"""Isolate whether the single-point-target range-profile PSLR discrepancy
(-8.5dB measured vs -13.2dB theoretical unweighted-sinc) already exists at
the bare range-compression stage, or is introduced later by the CSA
chirp-scaling / SRC / RCM phase corrections.

Two range profiles are compared, both taken through the point target at its
broadside pulse (azimuth_time = 0, row 1600 of the 3200x3200 grid):

  (a) "pure range compression": an ideal FFT matched filter (no window, no
      chirp-scaling/SRC/RCM phase terms at all) applied directly to the raw
      echo signal at that single pulse -- i.e. what range compression alone
      would produce, before any CSA-specific processing. Same matched-filter
      formula as back_projection_experiment.py's range_compress(), which was
      independently verified against the C++ echo model.

  (b) "full CSA": the actual focused_image output (range compression + SRC +
      RCM + azimuth compression all applied), read straight from the
      existing single_point_target/focused_image/single_point_target.npy.

Both are upsampled 16x via FFT zero-padding (same method as
gen_single_point_target.py) before measuring -3dB IRW and PSLR, so the
comparison isn't an artifact of native pixel quantization.

Does not modify any existing pipeline code -- standalone experiment script.

Usage:
    python isolate_range_compression_test.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")
SPT_DIR = os.path.join(BASE, "single_point_target")

LIGHT_SPEED_M_S = 3e8  # matches src/SigPar.cpp exactly
UPSAMPLE_FACTOR = 16
PEAK_ROW = 1600  # broadside pulse (azimuth_time_axis_sec == 0), see output_par_img.json


def upsample_via_zero_padding(complex_profile, factor):
    n = len(complex_profile)
    spectrum_shifted = np.fft.fftshift(np.fft.fft(complex_profile))
    pad_total = n * (factor - 1)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded_shifted = np.pad(spectrum_shifted, (pad_left, pad_right))
    padded = np.fft.ifftshift(padded_shifted)
    return np.fft.ifft(padded) * factor


def to_mag_db(complex_arr):
    return 10.0 * np.log10(np.abs(complex_arr) ** 2 + 1e-30)


def irw_3db(profile_db, peak_idx):
    peak_val = profile_db[peak_idx]
    threshold = peak_val - 3.0

    left = peak_idx
    while left > 0 and profile_db[left] >= threshold:
        left -= 1
    if left < peak_idx and profile_db[left] < threshold:
        y0, y1 = profile_db[left], profile_db[left + 1]
        frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
        left_sub = left + frac
    else:
        left_sub = float(left)

    right = peak_idx
    while right < len(profile_db) - 1 and profile_db[right] >= threshold:
        right += 1
    if right > peak_idx and profile_db[right] < threshold:
        y0, y1 = profile_db[right - 1], profile_db[right]
        frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
        right_sub = (right - 1) + frac
    else:
        right_sub = float(right)

    return right_sub - left_sub, left_sub, right_sub


def find_mainlobe_nulls(profile_db, peak_idx):
    """Walk outward from the peak to the first local minimum on each side
    (the mainlobe-to-sidelobe null), used to delimit where the mainlobe ends."""
    left = peak_idx
    while left > 0 and profile_db[left - 1] <= profile_db[left]:
        left -= 1
    right = peak_idx
    while right < len(profile_db) - 1 and profile_db[right + 1] <= profile_db[right]:
        right += 1
    return left, right


def measure_pslr(profile_db, peak_idx, search_span):
    """Peak sidelobe ratio: highest sidelobe peak beyond the first mainlobe
    null on either side, relative to the mainlobe peak, within +/-search_span
    samples (native-equivalent, scaled by caller)."""
    null_left, null_right = find_mainlobe_nulls(profile_db, peak_idx)
    lo = max(0, peak_idx - search_span)
    hi = min(len(profile_db), peak_idx + search_span)

    left_region = profile_db[lo:null_left] if null_left > lo else np.array([-np.inf])
    right_region = profile_db[null_right + 1:hi] if null_right + 1 < hi else np.array([-np.inf])

    sidelobe_peak_db = max(left_region.max(), right_region.max())
    return sidelobe_peak_db - profile_db[peak_idx], null_left, null_right


def analyze(label, complex_profile, spacing_m, upsample_factor=UPSAMPLE_FACTOR, native_search_span=60):
    upsampled = upsample_via_zero_padding(complex_profile, upsample_factor)
    up_db = to_mag_db(upsampled)
    up_peak = int(np.argmax(up_db))

    irw_px_up, left_sub, right_sub = irw_3db(up_db, up_peak)
    irw_px_native = irw_px_up / upsample_factor
    irw_m = irw_px_native * spacing_m

    pslr_db, null_left, null_right = measure_pslr(up_db, up_peak, native_search_span * upsample_factor)

    print(f"[{label}] IRW = {irw_px_native:.3f} px ({irw_m:.3f} m), "
          f"PSLR = {pslr_db:.2f} dB, mainlobe null-to-null = "
          f"{(null_right - null_left) / upsample_factor:.3f} px")

    return {
        "label": label,
        "irw_px_native": irw_px_native,
        "irw_m": irw_m,
        "pslr_db": pslr_db,
        "up_db": up_db,
        "up_peak": up_peak,
    }


def range_compress_single_pulse(pulse, range_time_axis, pulse_width_sec, bandwidth_hz, closest_slant_range_m):
    """Ideal FFT matched filter, no window -- same formula as
    back_projection_experiment.py's range_compress(), specialized to one pulse.
    This is the ENTIRE processing chain for case (a): no chirp-scaling, no
    SRC, no RCM, no azimuth compression."""
    n_col = len(range_time_axis)
    Kr = bandwidth_hz / pulse_width_sec
    t_center = 2.0 * closest_slant_range_m / LIGHT_SPEED_M_S
    rel_t = np.array(range_time_axis) - t_center
    window = (rel_t > -pulse_width_sec / 2.0) & (rel_t < pulse_width_sec / 2.0)
    ref_chirp = np.where(window, np.exp(1j * np.pi * Kr * rel_t ** 2), 0.0)

    REF_F = np.conj(np.fft.fft(ref_chirp))
    PULSE_F = np.fft.fft(pulse)
    rc = np.fft.ifft(PULSE_F * REF_F)
    rc = np.fft.fftshift(rc)
    return rc


def main():
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    with open(os.path.join(SPT_DIR, "input_par.json")) as f:
        par = json.load(f)
    with open(os.path.join(SPT_DIR, "output_par_img.json")) as f:
        axes = json.load(f)
    range_time_axis = np.array(axes["range_time_axis_sec"])
    azimuth_time_axis = np.array(axes["azimuth_time_axis_sec"])
    assert abs(azimuth_time_axis[PEAK_ROW]) < 1e-9, \
        f"expected broadside at row {PEAK_ROW}, got azimuth_time={azimuth_time_axis[PEAK_ROW]}"

    range_spacing_m = LIGHT_SPEED_M_S / (2 * par["sampling_freq_hz"])
    theoretical_range_res_m = LIGHT_SPEED_M_S / (2 * par["bandwidth_hz"])

    echo = np.load(os.path.join(SPT_DIR, "echo_signal", "single_point_target.npy"))
    focused = np.load(os.path.join(SPT_DIR, "focused_image", "single_point_target.npy"))
    n_row, n_col = focused.shape
    peak_row_csa, peak_col_csa = np.unravel_index(np.argmax(np.abs(focused)), focused.shape)
    print(f"CSA focused peak at (row={peak_row_csa}, col={peak_col_csa}); "
          f"echo broadside pulse used for (a) is row={PEAK_ROW}")

    # (a) pure range compression, broadside pulse only -- no CSA processing at all
    pulse = echo[PEAK_ROW, :]
    rc_pulse = range_compress_single_pulse(
        pulse, range_time_axis, par["pulse_width_sec"], par["bandwidth_hz"], par["closest_slant_range_m"]
    )

    # (b) full CSA range profile through the actual focused peak
    csa_range_profile = focused[peak_row_csa, :]

    print(f"\nTheoretical unweighted-sinc PSLR = -13.26 dB, "
          f"theoretical range resolution (c/2B) = {theoretical_range_res_m:.3f} m\n")

    result_a = analyze("(a) pure range compression only", rc_pulse, range_spacing_m)
    result_b = analyze("(b) full CSA (RC+SRC+RCM+azimuth)", csa_range_profile, range_spacing_m)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    win_native = 40
    for res, color in [(result_a, "tab:blue"), (result_b, "tab:orange")]:
        up_db = res["up_db"]
        peak = res["up_peak"]
        lo = max(0, peak - win_native * UPSAMPLE_FACTOR)
        hi = min(len(up_db), peak + win_native * UPSAMPLE_FACTOR)
        x = (np.arange(lo, hi) - peak) / UPSAMPLE_FACTOR
        ax.plot(x, up_db[lo:hi] - up_db[peak], color=color,
                label=f"{res['label']}: IRW={res['irw_m']:.2f}m, PSLR={res['pslr_db']:.1f}dB")
    ax.axhline(-13.26, color="gray", linestyle="--", label="theoretical unweighted-sinc PSLR (-13.26dB)")
    ax.axhline(-3.0, color="red", linestyle=":", linewidth=0.8, label="-3dB")
    ax.set_xlabel("range pixel (relative to peak, native units)")
    ax.set_ylabel("normalized magnitude (dB)")
    ax.set_title("Range profile: pure range compression vs. full CSA")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(DIAGRAM_DIR, "isolate_range_compression.png")
    plt.savefig(out_png, dpi=150)

    out = {
        "theoretical_pslr_db": -13.26,
        "theoretical_range_resolution_m": theoretical_range_res_m,
        "pure_range_compression": {"irw_m": result_a["irw_m"], "pslr_db": result_a["pslr_db"]},
        "full_csa": {"irw_m": result_b["irw_m"], "pslr_db": result_b["pslr_db"]},
    }
    out_json = os.path.join(DIAGRAM_DIR, "isolate_range_compression.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_png}")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
