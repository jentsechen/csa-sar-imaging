import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

JPG_NAME = "P0002_1800_2600_2400_3200.jpg"
ZERO_THRESHOLD = 150


def jpg_to_point_target(jpg_name, zero_threshold=ZERO_THRESHOLD):
    """Convert a grayscale JPG to a <name>.json point-target file in this directory.

    Pixel values below zero_threshold are set to 0.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jpg_path = os.path.join(base_dir, jpg_name)
    scene_name = os.path.splitext(jpg_name)[0]

    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    img = np.where(img < zero_threshold, 0, img).astype(np.uint8)

    json_path = os.path.join(base_dir, scene_name + ".json")
    with open(json_path, "w") as f:
        json.dump(img.tolist(), f)
    print(f"Saved {json_path} — shape: {img.shape}")

    return scene_name, img


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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    flat = np.array(data).flatten()
    print(f"Mean: {np.mean(flat)}")
    counts, _ = np.histogram(flat, bins=256, range=(0, 256))
    plt.plot(counts)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid()
    plt.savefig(os.path.join(base_dir, scene_name + "_hist.png"))
    plt.clf()


def plot_scene(scene_name, data=None):
    """Load a scene JSON from this directory if data is None, and save it as a viridis imshow PNG."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data is None:
        with open(os.path.join(base_dir, scene_name + ".json"), "r") as f:
            data = json.load(f)
    arr = np.flipud(np.array(data))
    print(arr.shape)
    plt.imshow(arr, origin="lower", cmap="viridis")
    plt.xlabel("range")
    plt.ylabel("azimuth")
    plt.colorbar()
    plt.savefig(os.path.join(base_dir, scene_name + ".png"), dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    scene_name, img = jpg_to_point_target(JPG_NAME)
    count_zeros(img)
    plot_histogram(scene_name, img)
    plot_scene(scene_name, data=img)
