#!/usr/bin/env python3
"""Standalone SAR ship detection — runs the trained YOLOv8m model on CPU or GPU.

Two modes (auto-detected from --source):
  * a folder of image patches  -> predict each, annotate, write one CSV
  * a single large GeoTIFF/JPG  -> multi-scale sliding-window, NMS, write CSV

No dependency on the original repo: just ultralytics + opencv + numpy + pandas.
"""
import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def load_gt(txt_path, img_w, img_h):
    """Load YOLO-format .txt labels, return list of [x1,y1,x2,y2] in pixels."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([
                (xc - w / 2) * img_w, (yc - h / 2) * img_h,
                (xc + w / 2) * img_w, (yc + h / 2) * img_h,
            ])
    return boxes


def box_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0


def match_preds(preds, gt_boxes, iou_thresh=0.5):
    """Greedy match predictions (sorted by conf) to GT boxes.
    Returns list of (conf, is_tp) and number of GT boxes."""
    matched_gt = set()
    results = []
    for xc, yc, w, h, conf in sorted(preds, key=lambda x: -x[4]):
        pb = [xc - w/2, yc - h/2, xc + w/2, yc + h/2]
        best_iou, best_j = iou_thresh, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_gt.add(best_j)
            results.append((conf, True))
        else:
            results.append((conf, False))
    return results, len(gt_boxes)


def compute_ap50(all_matches, n_gt_total):
    """Area under the precision-recall curve (envelope) at IoU≥0.5."""
    if n_gt_total == 0:
        return 0.0
    tp_cum = fp_cum = 0
    prec, rec = [], []
    for _, is_tp in sorted(all_matches, key=lambda x: -x[0]):
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        prec.append(tp_cum / (tp_cum + fp_cum))
        rec.append(tp_cum / n_gt_total)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    prec, rec = [1.0] + prec, [0.0] + rec
    return sum((rec[i] - rec[i-1]) * prec[i] for i in range(1, len(rec)))


def to_bgr_uint8(img):
    """Normalize a (possibly 16-bit / single-channel) SAR image to 3-ch uint8."""
    if img is None:
        raise ValueError("could not read image")
    if img.ndim == 2:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def apply_nms(detections, iou_threshold=0.1):
    """detections: list of [name, cls, xc, yc, w, h, conf] in full-image coords."""
    kept = []
    by_cls = {}
    for det in detections:
        _, cls, xc, yc, w, h, conf = det
        by_cls.setdefault(cls, []).append(
            ([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], conf, det)
        )
    for cls, dets in by_cls.items():
        boxes = torch.tensor([d[0] for d in dets], dtype=torch.float32)
        scores = torch.tensor([d[1] for d in dets], dtype=torch.float32)
        for idx in nms(boxes, scores, iou_threshold):
            kept.append(dets[idx][2])
    return kept


def run_patches(model, folder, args):
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )
    print(f"Found {len(files)} images in {folder}")
    rows = []
    annot_dir = os.path.join(args.out, "annotated")
    os.makedirs(annot_dir, exist_ok=True)

    all_matches, n_gt_total, metric_rows = [], 0, []

    for fp in files:
        r = model.predict(fp, imgsz=args.imgsz, conf=args.conf,
                          device=args.device, verbose=False)[0]
        n = len(r.boxes)
        print(f"  {os.path.basename(fp):55s} -> {n} ship(s)")
        cv2.imwrite(os.path.join(annot_dir, os.path.basename(fp)), r.plot())

        preds = []
        for box in r.boxes:
            xc, yc, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            rows.append([os.path.basename(fp), int(box.cls[0]), xc, yc, w, h, conf])
            preds.append([xc, yc, w, h, conf])

        if args.eval:
            img = cv2.imread(fp)
            ih, iw = img.shape[:2]
            txt = os.path.splitext(fp)[0] + ".txt"
            gt = load_gt(txt, iw, ih)
            matches, n_gt = match_preds(preds, gt)
            all_matches.extend(matches)
            n_gt_total += n_gt
            tp = sum(1 for _, t in matches if t)
            fp_n, fn = len(matches) - tp, n_gt - tp
            p  = tp / (tp + fp_n) if (tp + fp_n) > 0 else 0.0
            re = tp / n_gt        if n_gt > 0        else 0.0
            f1 = 2*p*re / (p+re)  if (p + re) > 0    else 0.0
            metric_rows.append([os.path.basename(fp), n_gt, n, tp, fp_n, fn,
                                 round(p, 3), round(re, 3), round(f1, 3)])

    if args.eval and metric_rows:
        ap50   = compute_ap50(all_matches, n_gt_total)
        tp_all = sum(r[3] for r in metric_rows)
        fp_all = sum(r[4] for r in metric_rows)
        fn_all = sum(r[5] for r in metric_rows)
        p_all  = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        r_all  = tp_all / n_gt_total         if n_gt_total > 0         else 0.0
        f1_all = 2*p_all*r_all / (p_all+r_all) if (p_all + r_all) > 0 else 0.0
        print(f"\n--- Eval (IoU≥0.5) ---")
        print(f"  Precision: {p_all:.3f}  Recall: {r_all:.3f}  "
              f"F1: {f1_all:.3f}  AP@50: {ap50:.3f}")
        cols = ["image", "n_gt", "n_pred", "TP", "FP", "FN", "precision", "recall", "f1"]
        df = pd.DataFrame(metric_rows, columns=cols)
        n_pred_all = sum(r[2] for r in metric_rows)
        df.loc[len(df)] = ["OVERALL", n_gt_total, n_pred_all, tp_all, fp_all, fn_all,
                           round(p_all, 3), round(r_all, 3), round(f1_all, 3)]
        metrics_csv = os.path.join(args.out, "metrics.csv")
        df.to_csv(metrics_csv, index=False)
        print(f"  Saved -> {metrics_csv}")

    return rows


def run_scene(model, image_path, args):
    img = to_bgr_uint8(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
    imgsz, stride = args.imgsz, args.stride
    all_det = []
    for scale in args.scales:
        scaled = img if scale == 1 else cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = scaled.shape[:2]
        patches, coords = [], []
        for i in range(0, w, stride):
            i = min(i, w - imgsz) if i + imgsz > w else i
            for j in range(0, h, stride):
                j = min(j, h - imgsz) if j + imgsz > h else j
                patches.append(scaled[j:j + imgsz, i:i + imgsz])
                coords.append((i, j))
                if j + imgsz >= h:
                    break
            if i + imgsz >= w:
                break
        print(f"  scale {scale}: {len(patches)} patches")
        for k in range(0, len(patches), args.batch):
            res = model.predict(source=patches[k:k + args.batch], imgsz=imgsz,
                                conf=args.conf, device=args.device,
                                stream=True, verbose=False)
            for r, (i, j) in zip(res, coords[k:k + args.batch]):
                for box in r.boxes:
                    xc, yc, w_b, h_b = box.xywh[0].tolist()
                    all_det.append([os.path.basename(image_path), int(box.cls[0]),
                                   (xc + i) / scale, (yc + j) / scale,
                                   w_b / scale, h_b / scale, float(box.conf[0])])
    print(f"  raw detections: {len(all_det)} -> NMS")
    return apply_nms(all_det, iou_threshold=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/best.pt")
    ap.add_argument("--source", default="test_data/patches",
                    help="folder of patches OR a single image/GeoTIFF")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cpu", help="cpu | 0 | cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--eval", action="store_true",
                    help="compute P/R/F1/AP@50 against sibling .txt GT labels (patch mode only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.model} on device={args.device} ...")
    t0 = time.time()
    model = YOLO(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s | classes: {model.names}")

    t0 = time.time()
    if os.path.isdir(args.source):
        rows = run_patches(model, args.source, args)
        csv = os.path.join(args.out, "patch_detections.csv")
    else:
        rows = run_scene(model, args.source, args)
        csv = os.path.join(args.out, "scene_detections.csv")

    pd.DataFrame(rows, columns=["image", "cls", "x_center", "y_center",
                                "width", "height", "conf"]).to_csv(csv, index=False)
    print(f"\nDONE: {len(rows)} detections -> {csv}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
