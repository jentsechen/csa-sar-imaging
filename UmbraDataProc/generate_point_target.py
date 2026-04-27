import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import os
from tqdm import tqdm
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from umbra_denoising import compare_all_methods, DenoisingMethod


S3_BUCKET = "umbra-open-data-catalog"
S3_PREFIX = "sar-data/tasks/ship_detection_testdata/"


def download_tif(uuid, bucket=S3_BUCKET, prefix=S3_PREFIX):
    """Download all .tif/.tiff files matching a UUID from an S3 bucket to the current directory."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    print(f"Searching for .tif files associated with: {uuid}...")

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    found_files = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if uuid in key and key.lower().endswith(('.tif', '.tiff')):
                local_path = os.path.basename(key)
                with tqdm(total=obj['Size'], unit='B', unit_scale=True, desc=local_path) as pbar:
                    s3.download_file(
                        bucket,
                        key,
                        local_path,
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
                found_files += 1

    if found_files == 0:
        print("No matching .tif files found.")
    else:
        print(f"\nDone! Downloaded {found_files} file(s).")


def read_sar_slice(file_name, row_start, row_len, col_start, col_len):
    """Read a rectangular slice from a GeoTIFF SAR image and print its shape and pixel spacing."""
    with rasterio.open(file_name) as dataset:
        sar_data = dataset.read(1)
        transform = dataset.transform
        pixel_spacing_x = -transform[0]
        pixel_spacing_y = transform[4]

    print(f"Data Shape: {sar_data.shape}")
    print(f"Pixel Resolution: {pixel_spacing_x}m x {pixel_spacing_y}m")

    sar_slice = sar_data[row_start:(row_start + row_len), col_start:(col_start + col_len)]
    return sar_slice


def binarize(sar_slice, threshold):
    """Apply a binary threshold to a SAR slice, returning a uint8 array of 0s and 1s."""
    _, sar_bin = cv2.threshold(sar_slice, threshold, 255, cv2.THRESH_BINARY)
    return (sar_bin / 255).astype(np.uint8)


def _to_uint8(sar_slice):
    """Normalize a float SAR slice to uint8 [0, 255]."""
    img = sar_slice.astype(np.float32)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _denoise_bilateral(sar_slice, d=9, sigma_color=75, sigma_space=75):
    """Edge-preserving bilateral filter; averages pixels by spatial and intensity proximity."""
    return cv2.bilateralFilter(_to_uint8(sar_slice), d, sigma_color, sigma_space)


def _denoise_guided(sar_slice, radius=8, eps=1e-4):
    """Self-guided filter: smooths homogeneous regions while preserving sharp SAR edges."""
    I = _to_uint8(sar_slice).astype(np.float32) / 255.0
    win = 2 * radius + 1
    mean_I  = cv2.boxFilter(I,     -1, (win, win))
    mean_II = cv2.boxFilter(I * I, -1, (win, win))
    var_I   = mean_II - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (win, win))
    mean_b = cv2.boxFilter(b, -1, (win, win))
    q = mean_a * I + mean_b
    return np.clip(q * 255, 0, 255).astype(np.uint8)


def _denoise_tv(sar_slice, weight=0.1):
    """Total Variation denoising: flattens clutter while keeping high-return target edges."""
    from skimage.restoration import denoise_tv_chambolle
    img = _to_uint8(sar_slice).astype(np.float32) / 255.0
    denoised = denoise_tv_chambolle(img, weight=weight)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def _denoise_lee(sar_slice, window_size=7):
    """Lee adaptive speckle filter: suppresses speckle using local mean and variance."""
    from scipy.ndimage import uniform_filter
    img = _to_uint8(sar_slice).astype(np.float32)
    local_mean    = uniform_filter(img,       size=window_size)
    local_sq_mean = uniform_filter(img ** 2,  size=window_size)
    local_var     = local_sq_mean - local_mean ** 2
    img_var       = np.var(img)
    k             = local_var / (local_var + img_var + 1e-10)
    result        = local_mean + k * (img - local_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


_DENOISE_FNS = {
    "bilateral": _denoise_bilateral,
    "guided":    _denoise_guided,
    "tv":        _denoise_tv,
    "lee":       _denoise_lee,
}

SEGMENT_METHODS = ("binary", "otsu", "bilateral", "guided", "tv", "lee")


def segment(sar_slice, method="binary", threshold=None,
            morph_close=False, morph_kernel=3, preserve_gray=False, **kwargs):
    """
    Process a SAR slice with the selected method.

    "binary" / "otsu": return a binary 0/1 mask.
    Filter methods ("bilateral", "guided", "tv", "lee"):
      - preserve_gray=False (default): denoise → Otsu → binary 0/1 mask
      - preserve_gray=True: denoise → Otsu mask → keep denoised grayscale for
        detected pixels, zero out background (uint8 0-255)

    Args:
        sar_slice: 2-D numpy array from read_sar_slice
        method: one of "binary", "otsu", "bilateral", "guided", "tv", "lee"
        threshold: fixed intensity threshold (required for method="binary" only)
        morph_close: apply morphological closing after thresholding
        morph_kernel: square kernel size for morphological closing
        preserve_gray: (filter methods only) keep denoised pixel values instead
                       of collapsing to binary
        **kwargs: forwarded to the denoising function (see each _denoise_* helper)
    Returns:
        uint8 array — 0/1 for binary/otsu or preserve_gray=False,
                      0-255 for filter methods with preserve_gray=True
    """
    if method == "binary":
        if threshold is None:
            raise ValueError("threshold is required for method='binary'")
        mask = binarize(sar_slice, threshold)
        if morph_close:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            mask = (cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel) / 255).astype(np.uint8)
        if preserve_gray:
            return cv2.bitwise_and(_to_uint8(sar_slice), mask * 255)
        return mask

    if method == "otsu":
        img8 = _to_uint8(sar_slice)
        _, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = (binary / 255).astype(np.uint8)
        if morph_close:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            result = (cv2.morphologyEx(result * 255, cv2.MORPH_CLOSE, kernel) / 255).astype(np.uint8)
        return result

    if method in _DENOISE_FNS:
        denoised = _DENOISE_FNS[method](sar_slice, **kwargs)
        _, otsu_mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if preserve_gray:
            result = cv2.bitwise_and(denoised, otsu_mask)
        else:
            result = (otsu_mask / 255).astype(np.uint8)
        if morph_close:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        return result

    raise ValueError(f"Unknown method '{method}'. Choose from: {SEGMENT_METHODS}")


def null_area(sar_slice, x_start, x_end, y_start, y_end):
    """Zero out a rectangular region of sar_slice in-place.

    x (columns / range): x_start to x_end (exclusive)
    y (rows / azimuth):  y_start to y_end (exclusive)
    Returns the modified array.
    """
    sar_slice[y_start:y_end, x_start:x_end] = 0
    return sar_slice


def save_scene(scene_name, data):
    """Serialize data to <scene_name>.json, accepting either a list or a numpy array."""
    with open('../TestMultiPointTarget/point_target_location/' + scene_name + '.json', "w") as f:
        json.dump(data if isinstance(data, list) else data.tolist(), f)
    print(f"Saved {scene_name}.json — shape: {np.array(data).shape}")


def count_zeros(data):
    """Count and print zero-valued elements in data; return (count, percentage)."""
    arr = np.array(data)
    num_zero = int((arr == 0).sum())
    total = arr.size
    percentage = (num_zero / total) * 100
    print(f"Number of zeros: {num_zero} ({percentage:.2f}%)")
    return num_zero, percentage


def plot_histogram(scene_name, data):
    """Save a 256-bin intensity histogram of data to <scene_name>_hist.png."""
    flat = np.array(data).flatten()
    counts, _ = np.histogram(flat, bins=256, range=(0, 256))
    plt.plot(counts)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.grid()
    plt.savefig(scene_name + "_hist.png")
    plt.clf()


def plot_scene(scene_name):
    """Load a scene JSON from the point_target_location directory and save it as a viridis imshow PNG."""
    with open('../TestMultiPointTarget/point_target_location/' + scene_name + '.json', 'r') as f:
        data = json.load(f)
    arr = np.flipud(np.array(data))
    print(arr.shape)
    plt.imshow(arr, origin='lower', cmap='viridis')
    plt.xlabel('range')
    plt.ylabel('azimuth')
    plt.colorbar()
    plt.savefig(scene_name + ".png", dpi=300, bbox_inches="tight")
    plt.clf()



def process_scene(scene_name, file_name, row_start, row_len, col_start, col_len,
                  method=None, threshold=None, morph_close=False, morph_kernel=3,
                  null_regions=None, **kwargs):
    """
    Process a SAR scene and save results.

    Args:
        scene_name: output name used for .json and .png files
        file_name: input GeoTIFF path
        row_start, row_len, col_start, col_len: slice bounds
        method: segmentation method passed to segment(); if None, saves raw float data.
                One of: "binary", "otsu", "bilateral", "guided", "tv", "lee"
        threshold: fixed threshold (required when method="binary")
        morph_close: apply morphological closing after thresholding
        morph_kernel: kernel size for morphological closing
        null_regions: list of (x_start, x_end, y_start, y_end) tuples to zero out
                      before segmentation, e.g. [(0, 500, 350, 640)]
        **kwargs: forwarded to the denoising helper (e.g. weight=, window_size=)
    """
    sar_slice = read_sar_slice(file_name, row_start, row_len, col_start, col_len)

    if null_regions:
        for x0, x1, y0, y1 in null_regions:
            null_area(sar_slice, x0, x1, y0, y1)

    if method is None:
        data = sar_slice
    else:
        data = segment(sar_slice, method=method, threshold=threshold,
                       morph_close=morph_close, morph_kernel=morph_kernel, **kwargs)

    save_scene(scene_name, data)
    count_zeros(data)
    plot_histogram(scene_name, data)
    plot_scene(scene_name)


if __name__ == "__main__":
    # download_tif(uuid='c59dab96-3b16-456b-80fe-866f30a3cabe')

    # compare_all_methods(
    #     scene_name='tsoying_naval_base',
    #     file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif',
    #     row_start=4050, row_len=640,
    #     col_start=4180, col_len=720,
    #     threshold=110,
    #     methods=[
    #         DenoisingMethod.RAW,
    #         DenoisingMethod.BINARY,
    #         DenoisingMethod.OTSU,
    #         DenoisingMethod.BILATERAL,
    #         DenoisingMethod.GUIDED,
    #         DenoisingMethod.TV,
    #         DenoisingMethod.LEE,
    #     ],
    # )

    # single-method processing (method="binary" is original behaviour)
    # process_scene(
    #     scene_name='tsoying_naval_base',
    #     file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif',
    #     row_start=4050, row_len=640,
    #     col_start=4180, col_len=720,
    #     method="binary", threshold=110,
    # )

    process_scene(
        scene_name='tsoying_naval_base',
        file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif',
        row_start=4050, row_len=640,
        col_start=4180, col_len=720,
        method="binary", threshold=110, morph_close=True, preserve_gray=True,
        null_regions=[(0, 500, 0, 290)]
    )

    # method="otsu"    — automatic global threshold, no manual threshold needed
    # process_scene(..., method="otsu")

    # method="bilateral" — edge-preserving denoise + Otsu; use morph_close to fill hull gaps
    # process_scene(..., method="bilateral", morph_close=True, d=9, sigma_color=75, sigma_space=75)

    # method="guided"  — guided-filter denoise + Otsu
    # process_scene(..., method="guided", radius=8, eps=1e-4)

    # method="tv"      — Total Variation denoise + Otsu (requires scikit-image)
    # process_scene(..., method="tv", weight=0.1)

    # method="lee"     — Lee adaptive speckle filter + Otsu (requires scipy)
    # process_scene(..., method="lee", window_size=7)

    # method=None      — save raw float data (no segmentation)
    # process_scene(..., method=None)

    # download_tif(uuid='8cd3eeb0-22e2-42d7-969e-030826a3a0c6')
    # process_scene(
    #     scene_name='port_of_kaohsiung',
    #     file_name='2023-06-30-12-56-34_UMBRA-05_GEC.tif',
    #     row_start=1500, row_len=10000,
    #     col_start=0, col_len=10000,
    #     method=None,
    # )