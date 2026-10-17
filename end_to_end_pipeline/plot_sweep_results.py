#!/usr/bin/env python3
"""Figures for the speckle sweep (sarsim/speckle_pipeline.py, levels = SSR) and the
thermal-noise sweep (sarsim/noise_pipeline.py, levels = SNR); --experiment picks one.

    <exp>/<run>/figures/metrics_vs_<lvl>.png   Precision / Recall / mAP@0.5 vs level, one line per method
    <exp>/<run>/figures/metrics_vs_<lvl>_regularization.png   the same without CSA (L0 / L1 / L1/2+TV)
    noise/<run>/figures/metrics_vs_snr_tuning.png   per-level tuning (C, solid) vs tuned at SNR = inf (A, dashed)
    <exp>/<run>/figures/example_grid[_<stem>].png         one test scene: level (rows) x method (cols),
                                                          GT (green) and YOLO (red) boxes
    <exp>/<run>/figures/example_zoom[_<stem>]_ship<n>.png  the same grid cropped around each GT ship

The example scene is chosen by a fixed rule, not by eye: per-scene tp/fp/fn are
computed for every test scene and (level, method) set (cached in per_scene.csv).
Candidates have 2-6 GT ships, the largest at least 40 px long, and CSA misses at
least one ship at every level (the aggregate trend); the one whose CSA recall at
the noise-free level is closest to the test-set CSA recall is used (ties: smallest name).

Usage (from repo root):
    python end_to_end_pipeline/plot_sweep_results.py --run full
    python end_to_end_pipeline/plot_sweep_results.py --run full --stem P0033_1800_2600_4200_5000
    python end_to_end_pipeline/plot_sweep_results.py --experiment noise --run full --stem P0033_1800_2600_4200_5000
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from find_threshold_diff_images import load_gt_boxes, match  # noqa: E402

EXPERIMENTS = {
    "speckle": dict(dir="speckle", key="ssr", set="main", name="SSR", levels=["ssrinf", "ssr10", "ssr5", "ssr0", "ssr-inf"],
                    labels=["∞", "10", "5", "0", "−∞"], weaker="weaker speckle", stronger="stronger speckle",
                    title="1-look speckle level", seed="speckle seed 0"),
    "noise": dict(dir="noise", key="snr", set="C", name="SNR", levels=["snrinf", "snr10", "snr5", "snr0", "snr-5"],
                  labels=["∞", "10", "5", "0", "−5"], weaker="weaker noise", stronger="stronger noise",
                  title="thermal noise level (no speckle, parameters tuned per level)", seed="noise seed 0"),
}
CFG = EXPERIMENTS["speckle"]   # set in main()
METHODS = ["csa", "l0", "l1", "l12tv"]
LABEL = {"csa": "CSA", "l0": "L0", "l1": "L1", "l12tv": "L1/2 + TV"}
# validated categorical slots 1-4 (dataviz reference palette, light surface); contrast
# of slots 3-4 is below 3:1, so every line also gets its own marker and a direct label
COLOR = {"csa": "#2a78d6", "l0": "#eb6834", "l1": "#1baf7a", "l12tv": "#eda100"}
MARKER = {"csa": "o", "l0": "s", "l1": "^", "l12tv": "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_metrics(run_dir, out, methods=METHODS, title_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SSR, key_lv = CFG["levels"], CFG["key"]
    rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "results.csv"))) if r["set"] == CFG["set"]]
    n = rows[0]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, (("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP@0.5"))):
        ax.set_facecolor(SURFACE)
        for m in methods:
            y = [float(next(r[key] for r in rows if r[key_lv] == s and r["method"] == m)) for s in SSR]
            ax.plot(range(len(SSR)), y, color=COLOR[m], marker=MARKER[m], markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[m], zorder=3)
            if key == "map50":  # direct labels at the right end of the headline panel
                ax.annotate(LABEL[m], (len(SSR) - 1, y[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xticks(range(len(SSR)), CFG["labels"])
        ax.set_xlabel(f"{CFG['name']} (dB)   ← {CFG['weaker']} · {CFG['stronger']} →", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    axes[2].set_xlim(-0.3, len(SSR) - 1 + 1.1)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.suptitle(f"YOLO detection vs {CFG['title']} ({n} test scenes, HRSID, unmasked){title_note}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_tuning(run_dir, out):
    """Noise sweep only: per-level tuning (C, solid) vs parameters tuned at SNR = inf (A, dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = list(csv.DictReader(open(os.path.join(run_dir, "results.csv"))))
    lv = CFG["levels"]
    n = next(r["n"] for r in rows if r["set"] == "C")

    def val(tag, s, m, key):
        if tag == "A" and s != lv[0]:
            return float(next(r[key] for r in rows if r["set"] == "A" and r["snr"] == s and r["method"] == m + "_tunedinf"))
        return float(next(r[key] for r in rows if r["set"] == "C" and r["snr"] == s and r["method"] == m))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, (("recall", "Recall"), ("map50", "mAP@0.5"))):
        ax.set_facecolor(SURFACE)
        for m in METHODS[1:]:
            for tag, ls in (("C", "-"), ("A", "--")):
                y = [val(tag, s, m, key) for s in lv]
                ax.plot(range(len(lv)), y, color=COLOR[m], marker=MARKER[m], markersize=6, linewidth=2, linestyle=ls,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            if key == "map50":
                ax.annotate(LABEL[m], (len(lv) - 1, val("C", lv[-1], m, key)), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xticks(range(len(lv)), CFG["labels"])
        ax.set_xlabel(f"SNR (dB)   ← {CFG['weaker']} · {CFG['stronger']} →", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[1].set_xlim(-0.3, len(lv) - 1 + 1.1)
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=2, label=LABEL[m]) for m in METHODS[1:]]
    handles += [Line2D([], [], color=INK2, lw=2, ls="-", label="tuned per level (C)"),
                Line2D([], [], color=INK2, lw=2, ls="--", label="tuned at SNR = ∞ (A)")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle(f"Parameter tuning protocol under thermal noise ({n} test scenes, no speckle)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def per_scene(run_dir, cache):
    """tp/fp/fn of every test scene for every (SSR, method) set of the base seed."""
    if os.path.exists(cache):
        return list(csv.DictReader(open(cache)))
    sys.path.insert(0, os.path.dirname(BASE))
    from sarsim.detect_eval import load_model
    model = load_model()
    rows = []
    base = os.path.join(run_dir, "test", "seed0")
    SSR = CFG["levels"]
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(base, SSR[0], "csa")))
    for s in SSR:
        for m in METHODS:
            paths = [os.path.join(base, s, m, st + ".jpg") for st in stems]
            for i in range(0, len(paths), 64):
                res = model.predict(paths[i:i + 64], imgsz=800, conf=0.25, iou=0.45, device=0, verbose=False)
                for st, r in zip(stems[i:i + 64], res):
                    tp, fp, fn = match(load_gt_boxes(st), r.boxes.xyxy.cpu().numpy().tolist())[:3]
                    rows.append({"stem": st, "ssr": s, "method": m, "tp": tp, "fp": fp, "fn": fn})
            print(f"  per-scene {s} {m}", flush=True)
    with open(cache, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "ssr", "method", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(rows)
    return [{k: str(v) for k, v in r.items()} for r in rows]


def pick_scene(rows, target_recall):
    SSR = CFG["levels"]
    by = {}
    for r in rows:
        by.setdefault(r["stem"], {})[(r["ssr"], r["method"])] = r
    cand = []
    for st, d in by.items():
        n_gt = int(d[(SSR[0], "csa")]["tp"]) + int(d[(SSR[0], "csa")]["fn"])
        if not 2 <= n_gt <= 6 or not all(int(d[(s, "csa")]["fn"]) > 0 for s in SSR):
            continue
        if max(max(b[2] - b[0], b[3] - b[1]) for b in load_gt_boxes(st)) < 40:
            continue
        recall = int(d[(SSR[0], "csa")]["tp"]) / n_gt
        cand.append((abs(recall - target_recall), st))
    return sorted(cand)[0][1], len(cand)


def predictions(run_dir, stem):
    """YOLO boxes on the full JPG for every (SSR, method) set of the base seed."""
    sys.path.insert(0, os.path.dirname(BASE))
    from sarsim.detect_eval import load_model
    model = load_model()
    out = {}
    for s in CFG["levels"]:
        for m in METHODS:
            path = os.path.join(run_dir, "test", "seed0", s, m, stem + ".jpg")
            r = model.predict(path, imgsz=800, conf=0.25, iou=0.45, device=0, verbose=False)[0]
            out[(s, m)] = r.boxes.xyxy.cpu().numpy().tolist()
    return out


def ship_rois(gt, size=800, scale=2.5, min_side=64):
    """Square crop around each GT ship: side = max(scale * longer box side, min_side), kept inside the image."""
    rois = []
    for b in gt:
        side = int(max(scale * max(b[2] - b[0], b[3] - b[1]), min_side))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(np.clip(cx - side / 2, 0, size - side))
        y0 = int(np.clip(cy - side / 2, 0, size - side))
        rois.append((x0, y0, x0 + side, y0 + side))
    return rois


def plot_grid(run_dir, stem, out, preds, roi=None, title=""):
    """SSR level (rows) x [original, methods] (cols); roi = (x0, y0, x1, y1) zooms every panel.
    tp/fp/fn are always for the whole scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    SSR = CFG["levels"]
    gt = load_gt_boxes(stem)
    x0, y0, x1, y1 = roi or (0, 0, 800, 800)
    fig, axes = plt.subplots(len(SSR), len(METHODS) + 1, figsize=(2.3 * (len(METHODS) + 1), 2.35 * len(SSR)),
                             facecolor=SURFACE)

    def show(ax, img, boxes):
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        for bs, c in boxes:
            for b in bs:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c,
                                       lw=1.5 if roi else 1))
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y1 - 0.5, y0 - 0.5)

    orig = cv2.imread(os.path.join(BASE, "images", stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
    for i, s in enumerate(SSR):
        ax = axes[i, 0]
        if i == 0:
            show(ax, orig, ((gt, "#00c800"),))
            ax.set_title("original HRSID image", fontsize=9, color=INK)
        else:
            ax.axis("off")
        for j, m in enumerate(METHODS, start=1):
            ax = axes[i, j]
            img = cv2.imread(os.path.join(run_dir, "test", "seed0", s, m, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
            pred = preds[(s, m)]
            tp, fp, fn = match(gt, pred)[:3]
            show(ax, img, ((gt, "#00c800"), (pred, "#ff2020")))
            ax.text(0.02, 0.97, f"scene tp {tp} fp {fp} fn {fn}", transform=ax.transAxes, va="top", fontsize=7,
                    color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))
            if i == 0:
                ax.set_title(LABEL[m], fontsize=10, color=INK)
            if j == 1:
                ax.set_ylabel(f"{CFG['name']} {CFG['labels'][i]} dB", fontsize=10, color=INK)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{stem}{title}: GT (green) / YOLO (red), 30 dB display, {CFG['seed']}", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="speckle", choices=sorted(EXPERIMENTS))
    ap.add_argument("--run", default="full")
    ap.add_argument("--stem", default=None, help="example scene (default: picked by the rule above)")
    ap.add_argument("--max-ships", type=int, default=3, help="ship close-up figures (largest ships first)")
    ap.add_argument("--metrics-only", action="store_true", help="only the metrics-vs-SSR figures")
    args = ap.parse_args()
    global CFG
    CFG = EXPERIMENTS[args.experiment]
    run_dir = os.path.join(BASE, CFG["dir"], args.run)
    out_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_metrics(run_dir, os.path.join(out_dir, f"metrics_vs_{CFG['key']}.png"))
    # regularizations only: without CSA the y-axes zoom in on their differences
    plot_metrics(run_dir, os.path.join(out_dir, f"metrics_vs_{CFG['key']}_regularization.png"),
                 methods=[m for m in METHODS if m != "csa"], title_note=", regularizations only")
    if args.experiment == "noise":
        plot_tuning(run_dir, os.path.join(out_dir, "metrics_vs_snr_tuning.png"))
    if args.metrics_only:
        print("figures ->", out_dir)
        return
    stem = args.stem
    if stem is None:
        rows = per_scene(run_dir, os.path.join(run_dir, "per_scene.csv"))
        main = list(csv.DictReader(open(os.path.join(run_dir, "results.csv"))))
        target = float(next(r["recall"] for r in main if r["set"] == CFG["set"] and r[CFG["key"]] == CFG["levels"][0]
                            and r["method"] == "csa"))
        stem, n_cand = pick_scene(rows, target)
        print(f"example scene {stem} ({n_cand} candidates; target CSA recall {target:.3f})")
    preds = predictions(run_dir, stem)
    tag = "" if args.stem is None else "_" + stem
    plot_grid(run_dir, stem, os.path.join(out_dir, f"example_grid{tag}.png"), preds)
    # ship close-ups, largest ship first
    gt = load_gt_boxes(stem)
    order = sorted(range(len(gt)), key=lambda k: -(gt[k][2] - gt[k][0]) * (gt[k][3] - gt[k][1]))
    rois = ship_rois(gt)
    for n, k in enumerate(order[: args.max_ships], start=1):
        x0, y0, x1, y1 = rois[k]
        plot_grid(run_dir, stem, os.path.join(out_dir, f"example_zoom{tag}_ship{n}.png"), preds, rois[k],
                  title=f" ship {n}, x {x0}-{x1}, y {y0}-{y1}")
    print("figures ->", out_dir)


if __name__ == "__main__":
    main()
