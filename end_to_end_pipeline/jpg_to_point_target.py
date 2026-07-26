import json
import os

import cv2
import numpy as np

IMAGES_DIR = "images"
OUTPUT_DIR = "point_target_location"
THRESHOLD = 60


def apply_threshold(img, threshold=THRESHOLD):
    return np.where(img < threshold, 0, img).astype(np.uint8)


# method name -> processing function; more methods (denoise, otsu, ...) get added here later
METHODS = {
    "threshold": apply_threshold,
}


def jpg_to_point_target(jpg_path, method="threshold", **method_kwargs):
    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    return METHODS[method](img, **method_kwargs)


def convert_all(images_dir=IMAGES_DIR, output_dir=OUTPUT_DIR, method="threshold", **method_kwargs):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, images_dir)
    output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    jpg_names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
    for name in jpg_names:
        stem = os.path.splitext(name)[0]
        img = jpg_to_point_target(os.path.join(images_dir, name), method=method, **method_kwargs)
        with open(os.path.join(output_dir, stem + ".json"), "w") as f:
            json.dump(img.tolist(), f)

    print(f"Converted {len(jpg_names)} images -> {output_dir}")
    return jpg_names


if __name__ == "__main__":
    convert_all(method="threshold", threshold=THRESHOLD)
