import json
import os

import cv2
import numpy as np

SCENE_NAME = "P0002_1800_2600_2400_3200"


def json_to_gray_image(scene_name=SCENE_NAME):
    """Convert <scene_name>.json (raw 0-255 pixel grid) to a plain grayscale JPG
    usable directly with sar_ship_detect/infer.py --image.

    (The .png already in this directory is a matplotlib plot with colorbar/axes
    baked in and isn't suitable for inference.)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, scene_name + ".json")) as f:
        data = json.load(f)

    img = np.array(data, dtype=np.uint8)
    out_path = os.path.join(base_dir, scene_name + "_gray.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved {out_path} — shape: {img.shape}")
    return out_path


if __name__ == "__main__":
    json_to_gray_image()
