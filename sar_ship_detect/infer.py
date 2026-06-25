#!/usr/bin/env python3
"""Run inference with a trained YOLO model on the HRSID test set.

Usage:
    python infer.py                                        # combined val, default weights
    python infer.py --source path/to/imgs                 # arbitrary folder or image
    python infer.py --val                                  # mAP on combined val -> results.md
    python infer.py --val --data HRSID_YOLO/data_val_inshore.yaml   # inshore  -> results.md
    python infer.py --val --data HRSID_YOLO/data_val_offshore.yaml  # offshore -> results.md
    python infer.py --conf 0.3 --no-save-txt
"""
import argparse
import datetime
import os
import time

import yaml
from ultralytics import YOLO


def save_val_markdown(metrics, args, run_dir, subset_label, elapsed_s):
    """Write a markdown summary of val metrics to <run_dir>/results.md."""
    box = metrics.box
    speed = metrics.speed  # dict: preprocess, inference, loss, postprocess (ms/img)
    names = metrics.names  # dict {0: 'ship', ...}

    # number of images: sum GT instance counts across classes as a proxy,
    # or fall back to counting the val image symlinks
    try:
        ydata = yaml.safe_load(open(args.data))
        val_dir = os.path.join(ydata.get("path", ""), ydata.get("val", ""))
        n_images = len([f for f in os.listdir(val_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    except Exception:
        n_images = "—"

    # per-class rows — box.ap_class_index holds the class indices after val
    per_class_rows = ""
    if box.ap_class_index is not None and len(box.ap_class_index):
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = names.get(int(cls_idx), str(cls_idx))
            p      = box.p[i]
            r      = box.r[i]
            f1     = box.f1[i]
            ap50   = box.ap50[i]
            ap5095 = box.ap[i]
            per_class_rows += (
                f"| {cls_name:<12} | {p:>9.3f} | {r:>6.3f} | {f1:>6.3f} "
                f"| {ap50:>8.3f} | {ap5095:>12.3f} |\n"
            )

    inf_ms = speed.get("inference", float("nan"))

    md = f"""# HRSID Val — {subset_label}

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
**Weights:** `{args.weights}`
**Dataset yaml:** `{args.data}`
**Images:** {n_images} · **imgsz:** {args.imgsz} · **conf:** {args.conf} · **IoU:** {args.iou}
**Wall-clock time:** {elapsed_s:.1f} s

## Overall Metrics

| Metric           | Value  |
|------------------|--------|
| Precision        | {box.mp:.4f} |
| Recall           | {box.mr:.4f} |
| mAP@0.5          | {box.map50:.4f} |
| mAP@0.5:0.95     | {box.map:.4f} |
| Inference speed  | {inf_ms:.1f} ms/img |

## Per-Class Metrics

| Class        | Precision | Recall | F1   | AP@0.5   | AP@0.5:0.95 |
|--------------|-----------|--------|------|----------|-------------|
{per_class_rows.rstrip()}

## Plots

| Plot | File |
|------|------|
| PR curve        | [BoxPR_curve.png](BoxPR_curve.png) |
| Precision curve | [BoxP_curve.png](BoxP_curve.png) |
| Recall curve    | [BoxR_curve.png](BoxR_curve.png) |
| F1 curve        | [BoxF1_curve.png](BoxF1_curve.png) |
| Confusion matrix | [confusion_matrix_normalized.png](confusion_matrix_normalized.png) |
"""

    out_path = os.path.join(run_dir, "results.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  Saved -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", default="HRSID_YOLO/images/val")
    ap.add_argument("--data", default="HRSID_YOLO/data.yaml",
                    help="dataset yaml for --val mode; use data_val_inshore.yaml or "
                         "data_val_offshore.yaml for subsets")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--no-save-txt", action="store_true", help="skip saving YOLO .txt labels")
    ap.add_argument("--val", action="store_true",
                    help="run validation and compute mAP instead of predict")
    args = ap.parse_args()

    model = YOLO(args.weights)

    if args.val:
        # derive run name from yaml stem so each subset gets its own results dir
        yaml_stem = os.path.splitext(os.path.basename(args.data))[0]
        name = "hrsid_" + yaml_stem.replace("data_", "")

        # human-readable subset label for the markdown header
        label_map = {
            "data":              "Combined (inshore + offshore)",
            "data_val_inshore":  "Inshore",
            "data_val_offshore": "Offshore",
        }
        subset_label = label_map.get(yaml_stem, yaml_stem)

        t0 = time.perf_counter()
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project="runs/val",
            name=name,
        )
        elapsed_s = time.perf_counter() - t0
        print(f"\nVal completed in {elapsed_s:.1f} s")

        run_dir = str(metrics.save_dir)
        save_val_markdown(metrics, args, run_dir, subset_label, elapsed_s)

    else:
        t0 = time.perf_counter()
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
        elapsed_s = time.perf_counter() - t0

        n_det = sum(len(r.boxes) for r in results)
        print(f"\nDone — {len(results)} images, {n_det} detections total")
        print(f"Elapsed: {elapsed_s:.1f} s  ({elapsed_s / len(results) * 1000:.1f} ms/img)")
        print(f"Annotated images saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
