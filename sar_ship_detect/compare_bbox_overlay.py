#!/usr/bin/env python3
"""Draw GT + predicted bounding boxes on the original HRSID image and on the
CSA-reconstructed grayscale image, each on its own background, for comparison.

Usage:
    python compare_bbox_overlay.py
"""
import os

import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
LABEL_PATH = "HRSID_YOLO/labels/val/P0002_1800_2600_2400_3200.txt"

# name -> (image_path, out_dir)
IMAGES = {
    "original": ("HRSID_YOLO/images/val/P0002_1800_2600_2400_3200.jpg",
                 "../diagram/perf_metric/P0002_1800_2600_2400_3200"),
    "csa_gray": ("../diagram/perf_metric/P0002_1800_2600_2400_3200/result_csa_gray.jpg",
                 "../diagram/perf_metric/P0002_1800_2600_2400_3200"),
    "point_target": ("../gen_point_target_location/P0002_1800_2600_2400_3200_gray.jpg",
                      "../gen_point_target_location"),
}

CONF = 0.25
IOU = 0.45

COLOR_GT = (0, 255, 0)      # green (BGR)
COLOR_PRED = (0, 0, 255)    # red


def load_yolo_labels(label_path, img_w, img_h):
    """Read YOLO-format normalized labels (class cx cy w h) and return xyxy pixel boxes."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            boxes.append([
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
            ])
    return boxes


def draw_boxes(img, boxes, color, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def main():
    model = YOLO(WEIGHTS)

    for name, (image_path, out_dir) in IMAGES.items():
        os.makedirs(out_dir, exist_ok=True)
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_h, img_w = gray.shape
        gt_boxes = load_yolo_labels(LABEL_PATH, img_w, img_h)

        r = model.predict(source=image_path, imgsz=800, conf=CONF, iou=IOU,
                           device="cpu", verbose=False)[0]

        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        draw_boxes(canvas, gt_boxes, COLOR_GT)
        draw_boxes(canvas, r.boxes.xyxy.tolist(), COLOR_PRED)

        for i, (text, color) in enumerate([("GT", COLOR_GT), ("Pred", COLOR_PRED)]):
            cv2.putText(canvas, text, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join(out_dir, f"bbox_overlay_{name}.png")
        cv2.imwrite(out_path, canvas)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
