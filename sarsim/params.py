"""Imaging geometry / axes, mirroring src/SigPar.cpp and src/ImagingPar.cpp, and
the point-target list construction of src/apps/gen_echo_signal.cpp."""
import json
import os

import cv2
import numpy as np

LIGHT_SPEED_M_S = 3e8
PI = 3.14159265358979323846


def load_input_par(path):
    with open(path) as f:
        return json.load(f)


POINT_TARGET_EXTS = (".png", ".json")


def list_point_target_stems(point_target_dir):
    """Scene stems with a point-target mask (<stem>.png preferred, or legacy <stem>.json)."""
    stems = set()
    for f in os.listdir(point_target_dir):
        stem, ext = os.path.splitext(f)
        if ext in POINT_TARGET_EXTS:
            stems.add(stem)
    return sorted(stems)


def load_point_target_mask(point_target_dir, stem):
    """uint8 scatterer mask: the lossless <stem>.png written by
    mask_outside_gt_pred_union.py, else a legacy <stem>.json (nested list)."""
    png = os.path.join(point_target_dir, stem + ".png")
    if os.path.exists(png):
        mask = cv2.imread(png, cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise ValueError(f"{png}: expected a single-channel image")
        return mask
    with open(os.path.join(point_target_dir, stem + ".json")) as f:
        return np.asarray(json.load(f))


def freq_axis(sampling_freq_hz, n):
    """ImagingPar's gen_freq_axis: (i - n/2) * fs / n, centered."""
    return (np.arange(n) - n // 2) * (sampling_freq_hz / float(n))


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
    azi_pad_time = par.get("azi_pad_time", 1.0)

    range_fm_rate_hz_s = par["bandwidth_hz"] / pulse_width_sec

    closest_ground_range_m = np.sqrt(closest_slant_range_m**2 - height_m**2)
    beamwidth_rad = wavelength_m / azimuth_aperture_len_m
    synthetic_aperture_len_m = beamwidth_rad * closest_slant_range_m
    synthetic_aperture_time_sec = synthetic_aperture_len_m / sensor_speed_m_s

    n_col = int(np.floor(rng_pad_time * pulse_width_sec * sampling_freq_hz / 2)) * 2
    range_time_axis_sec = (np.arange(n_col) - n_col // 2) / sampling_freq_hz + \
        2.0 * closest_slant_range_m / LIGHT_SPEED_M_S

    n_row = int(np.floor(azi_pad_time * synthetic_aperture_time_sec * pulse_rep_freq_hz / 2)) * 2
    azimuth_time_axis_sec = (np.arange(n_row) - n_row // 2) / pulse_rep_freq_hz

    return dict(
        wavelength_m=wavelength_m, pulse_width_sec=pulse_width_sec,
        pulse_rep_freq_hz=pulse_rep_freq_hz, sampling_freq_hz=sampling_freq_hz,
        range_fm_rate_hz_s=range_fm_rate_hz_s,
        closest_slant_range_m=closest_slant_range_m, height_m=height_m,
        closest_ground_range_m=closest_ground_range_m,
        sensor_speed_m_s=sensor_speed_m_s,
        synthetic_aperture_time_sec=synthetic_aperture_time_sec,
        n_row=n_row, n_col=n_col,
        range_time_axis_sec=range_time_axis_sec,
        azimuth_time_axis_sec=azimuth_time_axis_sec,
    )


def build_point_target_list(mask, n_row, n_col, pulse_rep_freq_hz, sampling_freq_hz):
    """Mirrors gen_echo_signal.cpp's else-branch indexing exactly. Targets come
    out in row-major order, i.e. sorted by azimuth offset."""
    mask = np.asarray(mask)
    ii, jj = np.nonzero(mask > 0)
    scatter = mask[ii, jj].astype(np.float64)

    azimuth_offset_sec = ((ii + n_row * 3 // 8) - n_row / 2.0) / pulse_rep_freq_hz
    range_offset_m = ((jj + n_col * 3 // 8) - n_col / 2.0) * LIGHT_SPEED_M_S / 2.0 / sampling_freq_hz
    return azimuth_offset_sec, range_offset_m, scatter
