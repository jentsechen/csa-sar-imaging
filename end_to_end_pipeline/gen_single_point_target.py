#!/usr/bin/env python3
"""Simulate a single point target through echo generation + CSA focusing
(no detection), and measure range/azimuth resolution from the -3dB impulse
response width (IRW) of the focused image.

The native pixel spacing is often too coarse to resolve the mainlobe shape
(range here is only ~1.28x oversampled relative to the theoretical
resolution), so before measuring IRW each 1D profile through the peak is
upsampled via frequency-domain zero-padding (sinc interpolation on the
complex focused signal -- NOT on the dB magnitude, which would distort the
peak shape) by UPSAMPLE_FACTOR. This is the standard SAR/radar way to
measure impulse response width accurately from an undersampled image.

Self-contained working directory: single_point_target/
    echo_signal/single_point_target.npy
    focused_image/single_point_target.npy          (complex, used for upsampling)
    focused_image/single_point_target_mag_db.npy    (used only to locate the peak)
    resolution_result.json
    range_azimuth_profile.png

Pixel spacing (see src/ImagingPar.cpp):
    range spacing   = c / (2 * sampling_freq_hz)             (no resampling
                       between echo_signal and focused_image -- same axes)
    azimuth spacing = sensor_speed_m_s / pulse_rep_freq_hz    (sensor_speed_m_s
                       is read from input_par.json, falling back to the
                       airborne default 120 m/s if omitted -- see
                       ImagingPar.h / gen_echo_signal.cpp)

Usage:
    python gen_single_point_target.py
    python gen_single_point_target.py --azimuth-offset-m 0.04 --range-offset-m 30.0
"""
import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "single_point_target")
ECHO_SIGNAL_DIR = os.path.join(BASE, "echo_signal")
FOCUSED_IMAGE_DIR = os.path.join(BASE, "focused_image")
GEN_ECHO_SIGNAL_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "build", "gen_echo_signal"))
TEST_MULTI_POINT_TARGET_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "build", "TestMultiPointTarget"))
TARGET = "single_point_target"

SPEED_OF_LIGHT_M_S = 299792458.0
UPSAMPLE_FACTOR = 16

# same spaceborne sensor params as gen_echo_signal_union_batch.py's INPUT_PAR
# (Sentinel-1B S3-SM-like; wavelength_m/pulse_width_sec/pulse_rep_freq_hz/
# sampling_freq_hz nudged so n_row == n_col == 3200, matching the 800x800
# source-image grid used elsewhere in this pipeline -- see conversation notes)
INPUT_PAR = {
    "wavelength_m": 0.0555042,
    "pulse_width_sec": 11.99e-6,
    "pulse_rep_freq_hz": 6648.15,
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
}
SENSOR_SPEED_M_S = INPUT_PAR["sensor_speed_m_s"]  # must match input_par.json for azimuth_spacing_m below


def run_cpp(binary, args):
    proc = subprocess.run([binary] + list(args), cwd=BASE, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} {args} failed:\n{proc.stderr}")


def upsample_via_zero_padding(complex_profile, factor):
    """Sinc-interpolate a 1D complex signal by zero-padding its spectrum.

    Operates on the full-length native signal (no cropping) to avoid
    truncation/windowing artifacts, then returns the upsampled complex
    profile (same physical extent, `factor` times as many samples).
    """
    n = len(complex_profile)
    spectrum_shifted = np.fft.fftshift(np.fft.fft(complex_profile))
    pad_total = n * (factor - 1)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded_shifted = np.pad(spectrum_shifted, (pad_left, pad_right))
    padded = np.fft.ifftshift(padded_shifted)
    return np.fft.ifft(padded) * factor  # *factor compensates ifft's 1/N normalization


def to_mag_db(complex_arr):
    return 10.0 * np.log10(np.abs(complex_arr) ** 2 + 1e-30)


def irw_3db(profile_db, peak_idx):
    """-3dB impulse response width, in (sub-pixel) pixels, around peak_idx.

    Walks outward to find the last in-mainlobe sample on each side, then
    linearly interpolates (in dB) between that sample and its outside
    neighbor to find the exact fractional-pixel position of the -3dB
    crossing -- avoids quantizing the width to whole pixels. Reliable as
    long as the profile is well-oversampled relative to the mainlobe
    (see upsample_via_zero_padding above).
    """
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


def measure_resolution(complex_profile, spacing_m, factor=UPSAMPLE_FACTOR):
    """Upsample a 1D complex profile, measure -3dB IRW, return (width_m, upsampled_db, up_left, up_right, up_peak)."""
    upsampled = upsample_via_zero_padding(complex_profile, factor)
    up_db = to_mag_db(upsampled)
    up_peak = int(np.argmax(up_db))
    width_up_px, up_left, up_right = irw_3db(up_db, up_peak)
    width_native_px = width_up_px / factor
    width_m = width_native_px * spacing_m
    return width_m, width_native_px, up_db, up_left, up_right, up_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuth-offset-m", type=float, default=0.0)
    ap.add_argument("--range-offset-m", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(ECHO_SIGNAL_DIR, exist_ok=True)
    os.makedirs(FOCUSED_IMAGE_DIR, exist_ok=True)
    with open(os.path.join(BASE, "input_par.json"), "w") as f:
        json.dump(INPUT_PAR, f)

    run_cpp(GEN_ECHO_SIGNAL_BIN, [TARGET, str(args.azimuth_offset_m), str(args.range_offset_m)])
    run_cpp(TEST_MULTI_POINT_TARGET_BIN, ["focus", TARGET])
    run_cpp(TEST_MULTI_POINT_TARGET_BIN, ["calc_mag", f"./focused_image/{TARGET}"])

    mag_db = np.load(os.path.join(FOCUSED_IMAGE_DIR, TARGET + "_mag_db.npy"))
    focused_complex = np.load(os.path.join(FOCUSED_IMAGE_DIR, TARGET + ".npy"))
    n_row, n_col = mag_db.shape
    peak_row, peak_col = np.unravel_index(np.argmax(mag_db), mag_db.shape)

    range_profile_complex = focused_complex[peak_row, :]      # varying column = range direction
    azimuth_profile_complex = focused_complex[:, peak_col]    # varying row = azimuth direction

    range_spacing_m = SPEED_OF_LIGHT_M_S / (2 * INPUT_PAR["sampling_freq_hz"])
    azimuth_spacing_m = SENSOR_SPEED_M_S / INPUT_PAR["pulse_rep_freq_hz"]

    range_res_m, range_width_px, range_up_db, r_left, r_right, r_up_peak = \
        measure_resolution(range_profile_complex, range_spacing_m)
    azimuth_res_m, azimuth_width_px, azimuth_up_db, a_left, a_right, a_up_peak = \
        measure_resolution(azimuth_profile_complex, azimuth_spacing_m)

    theoretical_range_res_m = SPEED_OF_LIGHT_M_S / (2 * INPUT_PAR["bandwidth_hz"])

    result = {
        "image_shape": [int(n_row), int(n_col)],
        "peak_row": int(peak_row),
        "peak_col": int(peak_col),
        "upsample_factor": UPSAMPLE_FACTOR,
        "range_spacing_m": range_spacing_m,
        "azimuth_spacing_m": azimuth_spacing_m,
        "range_irw_px": range_width_px,
        "azimuth_irw_px": azimuth_width_px,
        "range_resolution_m": range_res_m,
        "azimuth_resolution_m": azimuth_res_m,
        "theoretical_range_resolution_m": theoretical_range_res_m,
    }
    with open(os.path.join(BASE, "resolution_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Image shape: {n_row} x {n_col}, peak at (row={peak_row}, col={peak_col})")
    print(f"Upsample factor: {UPSAMPLE_FACTOR}x (zero-padding on complex focused signal)")
    print(f"Range spacing:   {range_spacing_m:.4f} m/px")
    print(f"Azimuth spacing: {azimuth_spacing_m:.4f} m/px")
    print(f"Range -3dB IRW:   {range_width_px:.3f} px (native-equivalent) -> {range_res_m:.3f} m "
          f"(theoretical c/2B = {theoretical_range_res_m:.3f} m)")
    print(f"Azimuth -3dB IRW: {azimuth_width_px:.3f} px (native-equivalent) -> {azimuth_res_m:.3f} m")

    win_native = 40
    win_up = win_native * UPSAMPLE_FACTOR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rc0, rc1 = max(0, r_up_peak - win_up), min(len(range_up_db), r_up_peak + win_up)
    x_range = (np.arange(rc0, rc1) - r_up_peak) / UPSAMPLE_FACTOR
    axes[0].plot(x_range, range_up_db[rc0:rc1])
    axes[0].axhline(range_up_db[r_up_peak] - 3.0, color="red", linestyle="--", label="-3dB")
    axes[0].axvline((r_left - r_up_peak) / UPSAMPLE_FACTOR, color="gray", linestyle=":")
    axes[0].axvline((r_right - r_up_peak) / UPSAMPLE_FACTOR, color="gray", linestyle=":")
    axes[0].set_title(f"Range profile (IRW={range_width_px:.2f}px = {range_res_m:.2f}m)")
    axes[0].set_xlabel("range pixel (relative to peak, native units)")
    axes[0].set_ylabel("magnitude (dB)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    ac0, ac1 = max(0, a_up_peak - win_up), min(len(azimuth_up_db), a_up_peak + win_up)
    x_azimuth = (np.arange(ac0, ac1) - a_up_peak) / UPSAMPLE_FACTOR
    axes[1].plot(x_azimuth, azimuth_up_db[ac0:ac1])
    axes[1].axhline(azimuth_up_db[a_up_peak] - 3.0, color="red", linestyle="--", label="-3dB")
    axes[1].axvline((a_left - a_up_peak) / UPSAMPLE_FACTOR, color="gray", linestyle=":")
    axes[1].axvline((a_right - a_up_peak) / UPSAMPLE_FACTOR, color="gray", linestyle=":")
    axes[1].set_title(f"Azimuth profile (IRW={azimuth_width_px:.2f}px = {azimuth_res_m:.2f}m)")
    axes[1].set_xlabel("azimuth pixel (relative to peak, native units)")
    axes[1].set_ylabel("magnitude (dB)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "range_azimuth_profile.png"), dpi=150)
    print(f"\nSaved -> {os.path.join(BASE, 'resolution_result.json')}")
    print(f"Saved -> {os.path.join(BASE, 'range_azimuth_profile.png')}")


if __name__ == "__main__":
    main()
