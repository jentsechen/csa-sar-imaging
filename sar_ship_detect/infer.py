#!/usr/bin/env python3
"""Run inference with a trained YOLO model on the HRSID test set.

Usage:
    python infer.py --val --data HRSID_YOLO/data.yaml --device cpu
    python infer.py                                        # combined val, default weights
    python infer.py --source path/to/imgs                 # arbitrary folder or image
    python infer.py --val                                  # mAP on combined val -> results.md
    python infer.py --val --data HRSID_YOLO/data_val_inshore.yaml   # inshore  -> results.md
    python infer.py --val --data HRSID_YOLO/data_val_offshore.yaml  # offshore -> results.md
    python infer.py --conf 0.3 --no-save-txt
    python infer.py --image HRSID_YOLO/images/val/P0001_0_800_10190_10990.jpg --device cpu
                                                            # single image -> precision/recall vs GT label
    python infer.py --image path/to/img.jpg --label path/to/gt.txt --match-iou 0.6
                                                            # single image, explicit GT label + match threshold
    python infer.py --filter-wrong --source HRSID_YOLO/images/val --device cpu
                                                            # list every image with a FP/FN -> runs/infer/wrong_images.txt
    python infer.py --filter-wrong --source HRSID_YOLO/images/val_inshore --out mywrong.txt
"""
import argparse
import datetime
import os
import time

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


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


def load_yolo_labels(label_path, img_w, img_h):
    """Read YOLO-format normalized labels (class cx cy w h) and return xyxy pixel boxes."""
    boxes = []
    if os.path.exists(label_path):
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
    return torch.tensor(boxes) if boxes else torch.zeros((0, 4))


def label_path_for(image_path):
    return os.path.splitext(image_path.replace("/images/", "/labels/"))[0] + ".txt"


def match_boxes(pred_boxes, pred_conf, gt_boxes, iou_thres):
    """Greedy match: highest-confidence predictions claim a GT box first. Returns tp, fp, fn."""
    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]
    iou = box_iou(pred_boxes, gt_boxes) if n_pred and n_gt else torch.zeros((n_pred, n_gt))

    order = torch.argsort(pred_conf, descending=True).tolist() if n_pred else []
    matched_gt = set()
    tp = 0
    for i in order:
        if n_gt == 0:
            break
        best_j = int(torch.argmax(iou[i]))
        if iou[i, best_j] >= iou_thres and best_j not in matched_gt:
            matched_gt.add(best_j)
            tp += 1

    return tp, n_pred - tp, n_gt - tp


def evaluate_single_image(model, image_path, args, iou_thres=0.5):
    """Run inference on one image and report TP/FP/FN, precision, recall vs ground truth."""
    label_path = args.label or label_path_for(image_path)

    results = model.predict(
        source=image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project="runs/infer",
        name="single_image",
    )
    r = results[0]
    img_h, img_w = r.orig_shape

    pred_boxes = r.boxes.xyxy.cpu()
    pred_conf = r.boxes.conf.cpu()
    gt_boxes = load_yolo_labels(label_path, img_w, img_h)
    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]

    tp, fp, fn = match_boxes(pred_boxes, pred_conf, gt_boxes, iou_thres)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))

    print(f"\nImage: {image_path}")
    print(f"Ground-truth label: {label_path}")
    print(f"GT boxes: {n_gt}   Predicted boxes: {n_pred} (conf>={args.conf}, IoU match>={iou_thres})")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"Annotated image saved to: {r.save_dir}")


def filter_wrong_images(model, source_dir, args, iou_thres=0.5):
    """Scan every image in source_dir and report the ones with any FP or FN (wrong predictions)."""
    exts = (".jpg", ".jpeg", ".png")
    image_paths = sorted(
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if f.lower().endswith(exts)
    )
    if not image_paths:
        print(f"No images found in {source_dir}")
        return

    # predict one image at a time (not a single call over the whole list) — passing the
    # full path list to predict(..., stream=True) was observed to leak memory across a
    # ~2k-image scan and get the process OOM-killed
    wrong = []
    for n, image_path in enumerate(image_paths, 1):
        r = model.predict(
            source=image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        img_h, img_w = r.orig_shape
        pred_boxes = r.boxes.xyxy.cpu()
        pred_conf = r.boxes.conf.cpu()
        gt_boxes = load_yolo_labels(label_path_for(image_path), img_w, img_h)
        n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]

        tp, fp, fn = match_boxes(pred_boxes, pred_conf, gt_boxes, iou_thres)
        if fp or fn:
            wrong.append((image_path, n_gt, n_pred, tp, fp, fn))

        if n % 200 == 0:
            print(f"  ...{n}/{len(image_paths)} scanned")

    print(f"\nScanned {len(image_paths)} images — {len(wrong)} with wrong predictions "
          f"(conf>={args.conf}, IoU match>={iou_thres})\n")
    print(f"{'Image':<55} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}")
    for image_path, n_gt, n_pred, tp, fp, fn in wrong:
        print(f"{os.path.basename(image_path):<55} {n_gt:>4} {n_pred:>5} {tp:>4} {fp:>4} {fn:>4}")

    out_path = args.out or "runs/infer/wrong_images.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("image,gt,pred,tp,fp,fn\n")
        for image_path, n_gt, n_pred, tp, fp, fn in wrong:
            f.write(f"{image_path},{n_gt},{n_pred},{tp},{fp},{fn}\n")
    print(f"\nSaved -> {out_path}")


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
    ap.add_argument("--image", default=None,
                    help="path to a single image; runs per-image precision/recall "
                         "vs its ground-truth label instead of batch predict/val")
    ap.add_argument("--label", default=None,
                    help="ground-truth YOLO .txt for --image (default: mirror images/->labels/ path)")
    ap.add_argument("--match-iou", dest="match_iou", type=float, default=0.5,
                    help="IoU threshold for matching a prediction to a GT box "
                         "in --image / --filter-wrong mode")
    ap.add_argument("--filter-wrong", action="store_true",
                    help="scan --source folder and list every image with a FP or FN")
    ap.add_argument("--out", default=None,
                    help="output path for --filter-wrong report (default runs/infer/wrong_images.txt)")
    args = ap.parse_args()

    model = YOLO(args.weights)

    if args.image:
        evaluate_single_image(model, args.image, args, iou_thres=args.match_iou)

    elif args.filter_wrong:
        filter_wrong_images(model, args.source, args, iou_thres=args.match_iou)

    elif args.val:
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
