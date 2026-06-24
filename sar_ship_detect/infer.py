#!/usr/bin/env python3
"""Run inference with a trained YOLO model on the HRSID test set.

Usage:
    python infer.py                          # val/test set, default weights
    python infer.py --source path/to/imgs   # arbitrary folder or image
    python infer.py --conf 0.3 --no-save-txt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/hrsid_yolo11m/weights/best.pt")
    ap.add_argument("--source", default="HRSID_YOLO/images/val")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--no-save-txt", action="store_true", help="skip saving YOLO .txt labels")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=not args.no_save_txt,
        project="runs/infer",
        name="hrsid_test",
    )

    n_det = sum(len(r.boxes) for r in results)
    print(f"\nDone — {len(results)} images, {n_det} detections total")
    print(f"Annotated images saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
