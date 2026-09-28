"""Focused image -> scene crop -> intensity / dB / 8-bit JPG.

The scene sits at row/col offset n*3//8 of the (n_row, n_col) grid (see
params.build_point_target_list), so the crop recovers exactly the input mask's
footprint. Everything returned here is flipped up-down, matching the
orientation of the source image and its YOLO labels (as in the original
csa_to_jpg_union_batch.py).
"""
import cv2
import numpy as np

DYNAMIC_RANGE_DB = 30


def scene_crop(xp_img, n_row, n_col, crop_shape):
    r0, c0 = n_row * 3 // 8, n_col * 3 // 8
    return xp_img[r0:r0 + crop_shape[0], c0:c0 + crop_shape[1]]


def crop_power(xp, focused_crop):
    """|x|^2 (linear power), float64, same array module as the input."""
    return focused_crop.real ** 2 + focused_crop.imag ** 2


def db_to_gray(mag_db, dynamic_range_db=DYNAMIC_RANGE_DB):
    """Host float64 dB crop -> uint8, identical to csa_to_jpg_union_batch.mag_db_to_gray_jpg."""
    max_val = np.max(mag_db)
    clipped = np.clip(mag_db, max_val - dynamic_range_db, max_val)
    return cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def scene_metrics(power):
    """Cheap per-scene image-quality numbers from the host linear-power crop."""
    total = float(power.sum())
    p = power / total if total > 0 else power
    return {
        "peak_db": float(10.0 * np.log10(power.max())) if total > 0 else float("-inf"),
        "entropy": float(-(p * np.log(p + 1e-12)).sum()),
    }


def process(xp, focused, n_row, n_col, crop_shape):
    """Returns host arrays (gray uint8, power float32) plus metrics, all flipped up-down."""
    power = crop_power(xp, scene_crop(focused, n_row, n_col, crop_shape))
    power_host = power.get() if hasattr(power, "get") else power
    with np.errstate(divide="ignore"):
        mag_db = 10.0 * np.log10(power_host)
    gray = np.flipud(db_to_gray(mag_db))
    return gray, np.flipud(power_host).astype(np.float32), scene_metrics(power_host)
