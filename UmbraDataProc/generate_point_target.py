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


S3_BUCKET = "umbra-open-data-catalog"
S3_PREFIX = "sar-data/tasks/ship_detection_testdata/"


def download_tif(uuid, bucket=S3_BUCKET, prefix=S3_PREFIX):
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
    _, sar_bin = cv2.threshold(sar_slice, threshold, 255, cv2.THRESH_BINARY)
    return (sar_bin / 255).astype(np.uint8)


def save_scene(scene_name, data):
    with open(scene_name + '.json', "w") as f:
        json.dump(data if isinstance(data, list) else data.tolist(), f)
    print(f"Saved {scene_name}.json — shape: {np.array(data).shape}")


def count_zeros(data):
    arr = np.array(data)
    num_zero = int((arr == 0).sum())
    total = arr.size
    percentage = (num_zero / total) * 100
    print(f"Number of zeros: {num_zero} ({percentage:.2f}%)")
    return num_zero, percentage


def plot_histogram(scene_name, data):
    flat = np.array(data).flatten()
    counts, _ = np.histogram(flat, bins=256, range=(0, 256))
    plt.plot(counts)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.grid()
    plt.savefig(scene_name + "_hist.png")
    plt.clf()


def plot_scene(scene_name):
    with open(scene_name + '.json', 'r') as f:
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
                  binarize_threshold=None):
    """
    Process a SAR scene and save results.

    Args:
        scene_name: output name used for .json and .png files
        file_name: input GeoTIFF path
        row_start, row_len, col_start, col_len: slice bounds
        binarize_threshold: if set, binarize the slice with this threshold;
                            if None, save raw float data
    """
    sar_slice = read_sar_slice(file_name, row_start, row_len, col_start, col_len)

    data = binarize(sar_slice, binarize_threshold) if binarize_threshold is not None else sar_slice

    save_scene(scene_name, data)
    count_zeros(data)
    plot_histogram(scene_name, data)
    plot_scene(scene_name)


if __name__ == "__main__":
    # download_tif(uuid='c59dab96-3b16-456b-80fe-866f30a3cabe')
    process_scene(
        scene_name='tsoying_naval_base',
        file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif',
        row_start=4050, row_len=640,
        col_start=4180, col_len=720,
        binarize_threshold=110,   # set to None to save raw float data
    )

    # download_tif(uuid='8cd3eeb0-22e2-42d7-969e-030826a3a0c6')
    # process_scene(
    #     scene_name='port_of_kaohsiung',
    #     file_name='2023-06-30-12-56-34_UMBRA-05_GEC.tif',
    #     row_start=1500, row_len=10000,
    #     col_start=0, col_len=10000,
    #     binarize_threshold=None,  # raw float
    # )