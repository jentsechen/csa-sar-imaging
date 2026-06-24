#!/usr/bin/env python3
"""Train a YOLO ship detector on HRSID (reproduces the published result).

Defaults match the committed run (YOLO11m, imgsz 800, 100 epochs, batch 32).
Prereqs:  pip install -r requirements.txt
          bash scripts/download_hrsid.sh && python convert_hrsid_to_yolo.py

Usage:    python train.py                      # full run, GPU 0
          python train.py --device cpu --epochs 20
          python train.py --model yolov8m.pt --batch 16
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt", help="base/pretrained weights")
    ap.add_argument("--data", default="HRSID_YOLO/data.yaml")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32, help="-1 = auto-size to VRAM")
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="hrsid_yolo11m")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
