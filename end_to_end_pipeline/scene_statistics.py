#!/usr/bin/env python3
"""Amplitude distribution of the clean offshore scenes: sea clutter vs ships.

The scene amplitude fed to the echo simulation is the grayscale pixel value of
images/<stem>.jpg (sarsim.params.build_point_target_list; HRSID pixels are linear
amplitude). Pixels are split with the HRSID instance-segmentation polygons, not the
GT boxes (a box is on average only half ship):

    ship = inside a ship polygon
    sea  = farther than GUARD pixels from every ship polygon (the guard band removes
           ship edges, sidelobes and ambiguity streaks next to the ship)

Pixel values are integers 0..255, so per-class 256-bin histograms hold the whole
distribution (all scenes, no subsampling). The sea is compared with a Rayleigh fit
(maximum likelihood on the binned data).

Outputs end_to_end_pipeline/scene_stats/{stats.json, figures/amplitude_hist.png}.
Usage: python3 scene_statistics.py
"""
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(BASE, "images")
ANN = os.path.join(BASE, "..", "sar_ship_detect", "HRSID_JPG", "annotations", "test2017.json")
OUT = os.path.join(BASE, "scene_stats")
GUARD = 10

# dataviz reference palette, categorical slots 1-2 (as in plot_sweep_results.py)
COLOR = {"sea": "#2a78d6", "ship": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_polygons():
    with open(ANN) as f:
        d = json.load(f)
    name = {im["id"]: os.path.splitext(im["file_name"])[0] for im in d["images"]}
    polys = {}
    for a in d["annotations"]:
        for seg in a["segmentation"]:
            polys.setdefault(name[a["image_id"]], []).append(np.array(seg, dtype=np.float64).reshape(-1, 2))
    return polys


def masks(shape, polys):
    """Ship mask (polygon interiors) and distance (px) of every pixel to the nearest ship pixel."""
    ship = np.zeros(shape, np.uint8)
    if polys:
        cv2.fillPoly(ship, [np.round(p).astype(np.int32) for p in polys], 1)
    if ship.any():
        dist = cv2.distanceTransform((1 - ship).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist = np.full(shape, np.inf, np.float32)
    return ship.astype(bool), dist


def collect():
    polys = load_polygons()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    hist = {"all": np.zeros(256, np.int64), "ship": np.zeros(256, np.int64), "sea": np.zeros(256, np.int64)}
    for stem in stems:
        img = cv2.imread(os.path.join(IMAGES, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        ship, dist = masks(img.shape, polys.get(stem, []))
        hist["all"] += np.bincount(img.ravel(), minlength=256)
        hist["ship"] += np.bincount(img[ship], minlength=256)
        hist["sea"] += np.bincount(img[(~ship) & (dist > GUARD)], minlength=256)
    return stems, hist


def fit_rayleigh(h):
    """Binned Rayleigh fit, f(x) = x / s^2 exp(-x^2 / (2 s^2)); bin v = [v, v+1)."""
    e = np.arange(257.0)

    def nll(ls):
        prob = np.clip(np.diff(1.0 - np.exp(-e ** 2 / (2.0 * np.exp(ls) ** 2))), 1e-300, None)
        return -float(np.sum(h * np.log(prob))) / float(h.sum())
    r = minimize_scalar(nll, bounds=(0.0, 6.0), method="bounded")
    return float(np.exp(r.x))


def rayleigh_binned(sigma):
    e = np.arange(257.0)
    return np.diff(1.0 - np.exp(-e ** 2 / (2.0 * sigma ** 2)))


def quantile(h, q):
    return int(np.searchsorted(np.cumsum(h) / h.sum(), q))


def summary(h):
    v = np.arange(256)
    return {"pixels": int(h.sum()), "mean": float(np.sum(v * h) / h.sum()),
            **{f"p{int(q * 1000) / 10:g}": quantile(h, q) for q in (0.5, 0.9, 0.99, 0.999)}}


def plot(hist, sigma, n_scenes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    v = np.arange(256)
    sea, ship = hist["sea"], hist["ship"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.step(v, 100 * sea / sea.sum(), where="post", color=COLOR["sea"], lw=2,
            label=f"sea clutter ({sea.sum() / 1e9:.2f}e9 pixels, guard {GUARD} px)")
    ax.step(v, 100 * ship / ship.sum(), where="post", color=COLOR["ship"], lw=2,
            label=f"ship ({ship.sum() / 1e6:.2f}e6 pixels, segmentation polygons)")
    ax.plot(v + 0.5, 100 * rayleigh_binned(sigma), color=COLOR["sea"], lw=1.2, ls="--",
            label=f"Rayleigh fit to sea, σ = {sigma:.1f}")
    ax.annotate("sea tail heavier\nthan Rayleigh", (85, 100 * sea[85] / sea.sum()), xytext=(103, 2e-3),
                fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"{ship[255] / ship.sum():.0%} of ship pixels clipped at 255", (255, 100 * ship[255] / ship.sum()),
                textcoords="offset points", xytext=(-190, -4), fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 100)
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.set_xlim(0, 255)
    ax.set_xlabel("pixel amplitude (linear, 0-255)")
    ax.set_ylabel("percentage of the class's pixels (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax.set_title(f"Clean offshore scenes: amplitude distribution ({n_scenes} scenes)\n"
                 "each curve normalized to its own class (sums to 100%)", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    stems, hist = collect()
    sigma = fit_rayleigh(hist["sea"])
    sea, ship = hist["sea"], hist["ship"]
    stats = {
        "scenes": len(stems), "guard_px": GUARD,
        "pixel_fraction": {"ship": float(ship.sum() / hist["all"].sum()), "sea": float(sea.sum() / hist["all"].sum())},
        "summary": {"ship": summary(ship), "sea": summary(sea)},
        "ship_clipped_255": float(ship[255] / ship.sum()),
        "sea_rayleigh_sigma": sigma,
        "sea_above_t": {str(t): {"measured": float(sea[t + 1:].sum() / sea.sum()),
                                 "rayleigh": float(np.exp(-(t + 1) ** 2 / (2 * sigma ** 2)))}
                        for t in (40, 47, 60, 80)},
        "hist": {k: v.tolist() for k, v in hist.items()},
    }
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    plot(hist, sigma, len(stems), os.path.join(OUT, "figures", "amplitude_hist.png"))
    print(json.dumps({k: v for k, v in stats.items() if k != "hist"}, indent=1))


if __name__ == "__main__":
    main()
