#!/usr/bin/env python3
"""Detection-example figure for the research proposal: the same scene as the
original SAR image, the CSA reconstruction, and CSA + thresholding
(sparse refinement), each with GT (green) and YOLO-predicted (red) boxes.

Uses the reference GPU run (end_to_end_pipeline/union_pipeline/runs/union_png)
and the same hard threshold as end_to_end_pipeline/threshold_union_csa.py.

Usage (from repo root):
    # 1. rank scenes where CSA errs and thresholding recovers the original result
    python document/moe_scholarship/01_research_proposal/figures/make_detection_example.py --search
    # 2. render the chosen scene -> detection_{csa,csa_threshold}.png next to this script
    python document/moe_scholarship/01_research_proposal/figures/make_detection_example.py --stem <stem>

Proposal figure (unmasked scene: the whole original image, sea surface
included, is used as point targets instead of the GT/pred-box union mask, so
the CSA background is realistic sea clutter rather than a black frame):
    mkdir -p /tmp/nomask_pt && python -c "import cv2; cv2.imwrite('/tmp/nomask_pt/P0033_1800_2600_4200_5000.png',
        cv2.imread('end_to_end_pipeline/images/P0033_1800_2600_4200_5000.jpg', cv2.IMREAD_GRAYSCALE))"
    python -m sarsim.pipeline --run nomask --algo csa --point-target-dir /tmp/nomask_pt --out-root /tmp/runs
    python document/moe_scholarship/01_research_proposal/figures/make_detection_example.py --stem P0033_1800_2600_4200_5000 \
        --roi 655,90,775,210 --csa-dir /tmp/runs/nomask/csa/jpg --no-boxes
The crop shows the large diagonal ship. Scene-level detection (tp/fp/fn,
printed to stdout): original 2/0/0, CSA 1/0/1 (misses a small ship at the
bottom edge), CSA + threshold 2/0/0. The --search
candidates (union_png run) all fix errors on small (<35 px) or border-cut
ships, which do not read as ships at figure scale.
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "..", "..", "..", "..", "end_to_end_pipeline")
sys.path.insert(0, PIPE)
from find_threshold_diff_images import load_gt_boxes, match  # noqa: E402

ORIGINAL_DIR = os.path.join(PIPE, "images")
CSA_DIR = os.path.join(PIPE, "union_pipeline", "runs", "union_png", "csa", "jpg")
WEIGHTS = os.path.join(PIPE, "..", "sar_ship_detect", "weights", "best.pt")

COLOR_GT = (0, 200, 0)      # BGR
COLOR_PRED = (0, 0, 255)
UPSCALE = 4                 # nearest-neighbour zoom of the crop, for crisp boxes


def threshold(img, t):
    return np.where(img < t, 0, img).astype(np.uint8)


def write_thresholded(stems, t, out_dir):
    """Threshold each CSA JPG and re-encode as JPG, as threshold_union_csa.py does."""
    for s in stems:
        img = cv2.imread(os.path.join(CSA_DIR, s + ".jpg"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(out_dir, s + ".jpg"), threshold(img, t))


def predict(model, image_dir, stems, args):
    out = {}
    for i in range(0, len(stems), 64):
        batch = stems[i:i + 64]
        rs = model.predict([os.path.join(image_dir, s + ".jpg") for s in batch], imgsz=800,
                           conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        for s, r in zip(batch, rs):
            out[s] = r.boxes.xyxy.cpu().numpy().tolist()
    return out


def search(model, args):
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(CSA_DIR) if f.endswith(".jpg"))
    with tempfile.TemporaryDirectory() as tdir:
        write_thresholded(stems, args.threshold, tdir)
        preds = {"orig": predict(model, ORIGINAL_DIR, stems, args),
                 "csa": predict(model, CSA_DIR, stems, args),
                 "thr": predict(model, tdir, stems, args)}
    rows = []
    for s in stems:
        gt = load_gt_boxes(s)
        r = {k: match(gt, preds[k][s])[:3] for k in preds}   # (tp, fp, fn)
        o, c, t = r["orig"], r["csa"], r["thr"]
        orig_clean = o[0] == len(gt) and o[1] == 0
        if orig_clean and t == o and c != o and len(gt) > 0:
            rows.append((c[1] + c[2], s, len(gt), c, t))
    rows.sort(reverse=True)
    print(f"{len(rows)} scene(s): original perfect, CSA errs, CSA+threshold == original")
    print(f"{'stem':<32} {'n_gt':>4}  csa(tp/fp/fn)  thr(tp/fp/fn)")
    for err, s, n, c, t in rows[:args.top]:
        print(f"{s:<32} {n:>4}  {c[0]}/{c[1]}/{c[2]:<10} {t[0]}/{t[1]}/{t[2]}")


def crop_box(boxes, margin, size=800):
    """Square crop (x0, y0, x1, y1) covering all boxes plus a margin."""
    b = np.array(boxes)
    x0, y0 = b[:, 0].min() - margin, b[:, 1].min() - margin
    x1, y1 = b[:, 2].max() + margin, b[:, 3].max() + margin
    side = int(max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(side, size)
    x0 = int(np.clip(cx - side / 2, 0, size - side))
    y0 = int(np.clip(cy - side / 2, 0, size - side))
    return x0, y0, x0 + side, y0 + side


def render(model, args):
    s = args.stem
    csa = cv2.imread(os.path.join(CSA_DIR, s + ".jpg"), cv2.IMREAD_GRAYSCALE)
    images = {"csa": csa}
    with tempfile.TemporaryDirectory() as tdir:
        write_thresholded([s], args.threshold, tdir)
        images["csa_threshold"] = cv2.imread(os.path.join(tdir, s + ".jpg"), cv2.IMREAD_GRAYSCALE)
        pred_src = {"original": ORIGINAL_DIR, "csa": CSA_DIR, "csa_threshold": tdir}
        preds = {k: predict(model, d, [s], args)[s] for k, d in pred_src.items()}
    print(f"original       {len(preds['original'])} pred, tp/fp/fn={match(load_gt_boxes(s), preds['original'])[:3]} (reference, not rendered)")
    gt = load_gt_boxes(s)
    if args.roi:
        x0, y0, x1, y1 = map(int, args.roi.split(","))
    else:
        x0, y0, x1, y1 = crop_box(gt + sum(preds.values(), []), args.margin)

    for name, img in images.items():
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=UPSCALE, fy=UPSCALE,
                          interpolation=cv2.INTER_NEAREST)
        canvas = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for boxes, color in (() if args.no_boxes else ((gt, COLOR_GT), (preds[name], COLOR_PRED))):
            for bx0, by0, bx1, by1 in boxes:
                p0 = (int((bx0 - x0) * UPSCALE), int((by0 - y0) * UPSCALE))
                p1 = (int((bx1 - x0) * UPSCALE), int((by1 - y0) * UPSCALE))
                cv2.rectangle(canvas, p0, p1, color, 3)
        out = os.path.join(HERE, f"detection_{name}.png")
        cv2.imwrite(out, canvas)
        print(f"{name:<14} {len(preds[name])} pred, tp/fp/fn={match(gt, preds[name])[:3]} -> {out}")
    print(f"crop: x {x0}-{x1}, y {y0}-{y1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--stem")
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--csa-dir", help="CSA JPG directory (default: the union_png run)")
    ap.add_argument("--no-boxes", action="store_true", help="omit GT/pred boxes from the PNGs")
    ap.add_argument("--roi", help="explicit crop x0,y0,x1,y1 (pixels) instead of the box-union crop")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    global CSA_DIR
    if args.csa_dir:
        CSA_DIR = args.csa_dir
    model = YOLO(WEIGHTS)
    if args.search:
        search(model, args)
    elif args.stem:
        render(model, args)
    else:
        ap.error("give --search or --stem")


if __name__ == "__main__":
    main()
