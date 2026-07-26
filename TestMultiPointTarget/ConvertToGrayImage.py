import os

import cv2
import numpy as np

IMAGE_NAME = "P0002_1800_2600_2400_3200"
N_ROW, N_COL = 800, 800
DYNAMIC_RANGE_DB = 30


def convert_to_gray_image(image_name=IMAGE_NAME, n_row=N_ROW, n_col=N_COL,
                           dynamic_range_db=DYNAMIC_RANGE_DB):
    """Convert focused_image/<image_name>_mag_db.npy to a plain grayscale JPG
    (no colorbar/axes) usable directly with sar_ship_detect/infer.py --image.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(base_dir, "focused_image", image_name + "_mag_db.npy")

    data = np.load(npy_path)[
        int(n_row * 3 / 2): int(n_row * 5 / 2),
        int(n_col * 3 / 2): int(n_col * 5 / 2),
    ]

    max_val = np.max(data)
    clipped = np.clip(data, max_val - dynamic_range_db, max_val)
    gray = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = np.flipud(gray)  # match origin="lower" orientation used by save_result/plot_scene

    out_dir = os.path.join(os.path.dirname(base_dir), "diagram", "perf_metric", image_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result_csa_gray.jpg")
    cv2.imwrite(out_path, gray)
    print(f"Saved {out_path} — shape: {gray.shape}")
    return out_path


if __name__ == "__main__":
    convert_to_gray_image()
