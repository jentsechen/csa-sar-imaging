#!/usr/bin/env python3
"""Per-image detection diagnostic: find scenes where union_csa (no threshold)
and union_csa_t<threshold> disagree on detection outcome (missed/extra GT
matches), and rank candidates for "which image best demonstrates thresholding
recovering toward original performance without exceeding it."

For each scene, GT boxes are matched (IoU>=0.5, greedy) against predictions
from each of the three sets (original, union_csa, union_csa_t<threshold>),
producing per-image TP/FP/FN counts. A scene is a good demo candidate when:
  - original is perfect (TP == n_gt, FP == 0) -- clean reference
  - union_csa misses something (TP < n_gt) or has a false positive
  - union_csa_t<threshold> recovers toward original (higher TP / fewer FP
    than union_csa) but does not exceed original (TP <= n_gt_original,
    FP <= FP_original) -- i.e. thresholding helps without "cheating" past
    the baseline

Usage:
    python find_threshold_diff_images.py --threshold 120 --device cpu
"""
import argparse
import json
import os

import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DIAGRAM_DIR = os.path.join(BASE, "..", "diagram", "thresholding", "union_csa")
ORIGINAL_DIR = os.path.join(BASE, "images")
UNION_CSA_DIR = os.path.join(BASE, "union_pipeline", "csa_jpg")
LABELS_DIR = os.path.join(BASE, "labels")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")
IMG_SIZE = 800


def load_gt_boxes(stem):
    path = os.path.join(LABELS_DIR, stem + ".txt")
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            _, xc, yc, w, h = parts
            xc, yc, w, h = float(xc) * IMG_SIZE, float(yc) * IMG_SIZE, float(w) * IMG_SIZE, float(h) * IMG_SIZE
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return boxes


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Greedy IoU matching; returns (tp, fp, fn, matched_ious)."""
    unmatched_gt = list(range(len(gt_boxes)))
    tp = 0
    matched_ious = []
    for pb in pred_boxes:
        best_iou, best_i = 0.0, -1
        for i in unmatched_gt:
            v = iou(gt_boxes[i], pb)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= iou_thresh:
            tp += 1
            matched_ious.append(best_iou)
            unmatched_gt.remove(best_i)
    fp = len(pred_boxes) - tp
    fn = len(unmatched_gt)
    return tp, fp, fn, matched_ious


def predict_all(model, image_dir, stems, args):
    paths = [os.path.join(image_dir, s + ".jpg") for s in stems]
    results = model.predict(paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                             device=args.device, verbose=False)
    out = {}
    for stem, r in zip(stems, results):
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        out[stem] = boxes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    thresholded_dir = os.path.join(BASE, "union_pipeline", f"csa_jpg_t{args.threshold}")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(UNION_CSA_DIR) if f.endswith(".jpg"))
    print(f"Evaluating {len(stems)} scene(s)\n")

    model = YOLO(WEIGHTS)
    preds = {
        "original": predict_all(model, ORIGINAL_DIR, stems, args),
        "union_csa": predict_all(model, UNION_CSA_DIR, stems, args),
        f"union_csa_t{args.threshold}": predict_all(model, thresholded_dir, stems, args),
    }

    rows = []
    for stem in stems:
        gt = load_gt_boxes(stem)
        n_gt = len(gt)
        per_set = {}
        for name, pred_map in preds.items():
            tp, fp, fn, ious = match(gt, pred_map[stem])
            per_set[name] = {"tp": tp, "fp": fp, "fn": fn, "mean_iou": float(np.mean(ious)) if ious else 0.0}
        rows.append({"stem": stem, "n_gt": n_gt, **{f"{k}": v for k, v in per_set.items()}})

    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    out_json = os.path.join(DIAGRAM_DIR, f"per_image_diff_t{args.threshold}.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved -> {out_json}\n")

    orig_key = "original"
    csa_key = "union_csa"
    t_key = f"union_csa_t{args.threshold}"

    print("Scenes where union_csa and the thresholded set DISAGREE (tp or fp differ):")
    print(f"{'stem':<32} {'n_gt':>5} {'orig(tp/fp)':>12} {'csa(tp/fp)':>11} {'t' + str(args.threshold) + '(tp/fp)':>11}")
    candidates = []
    for r in rows:
        o, c, t = r[orig_key], r[csa_key], r[t_key]
        if (c["tp"], c["fp"]) != (t["tp"], t["fp"]):
            print(f"{r['stem']:<32} {r['n_gt']:>5} "
                  f"{o['tp']}/{o['fp']:>9} {c['tp']}/{c['fp']:>9} {t['tp']}/{t['fp']:>9}")
            # "good demo" candidate: original is clean (tp==n_gt, fp==0),
            # thresholded set moves toward original (closer tp/fp) without
            # exceeding it (t_tp <= o_tp, t_fp <= o_fp -- here o_fp is 0 so
            # t_fp must be 0 too to not "exceed"; also require thresholded
            # to be strictly better than union_csa, i.e. real recovery)
            is_orig_clean = (o["tp"] == r["n_gt"] and o["fp"] == 0)
            csa_worse_than_orig = (c["tp"] < o["tp"]) or (c["fp"] > o["fp"])
            t_recovers = (t["tp"] > c["tp"]) or (t["fp"] < c["fp"])
            t_not_exceed_orig = (t["tp"] <= o["tp"]) and (t["fp"] <= o["fp"])
            if is_orig_clean and csa_worse_than_orig and t_recovers and t_not_exceed_orig:
                gap_before = (o["tp"] - c["tp"]) + (c["fp"] - o["fp"])
                gap_after = (o["tp"] - t["tp"]) + (t["fp"] - o["fp"])
                candidates.append({"stem": r["stem"], "n_gt": r["n_gt"],
                                    "gap_before": gap_before, "gap_after": gap_after,
                                    "improvement": gap_before - gap_after})

    print(f"\n{len(candidates)} candidate(s) matching 'thresholding recovers toward original, without exceeding it':")
    candidates.sort(key=lambda x: (-x["improvement"], x["gap_after"]))
    for c in candidates:
        print(f"  {c['stem']:<32} n_gt={c['n_gt']} gap_before={c['gap_before']} "
              f"gap_after={c['gap_after']} improvement={c['improvement']}")

    if candidates:
        best = candidates[0]
        print(f"\nRecommended scene for cross-section visualization: {best['stem']}")


if __name__ == "__main__":
    main()
