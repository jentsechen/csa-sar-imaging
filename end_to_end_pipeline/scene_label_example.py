#!/usr/bin/env python3
"""HRSID label example: GT box vs segmentation polygon, and the ship / guard / sea split
used by scene_statistics.py (ship = inside a polygon, sea = farther than GUARD px).

Outputs end_to_end_pipeline/scene_stats/figures/label_example.png.
Usage: python3 scene_label_example.py [stem]
"""
import json
import os
import sys

import cv2
import numpy as np

from scene_statistics import ANN, COLOR, GRID, GUARD, IMAGES, INK, INK2, OUT, SURFACE, masks

STEM = "P0126_1200_2000_3000_3800"


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else STEM
    with open(ANN) as f:
        d = json.load(f)
    image_id = next(im["id"] for im in d["images"] if os.path.splitext(im["file_name"])[0] == stem)
    anns = [a for a in d["annotations"] if a["image_id"] == image_id]
    polys = [np.array(s, np.float64).reshape(-1, 2) for a in anns for s in a["segmentation"]]
    img = cv2.imread(os.path.join(IMAGES, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
    ship, dist = masks(img.shape, polys)

    # class map: sea, guard band (not used), ship
    classes = np.zeros(img.shape + (3,))
    rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    classes[:] = rgb(COLOR["sea"])
    classes[(~ship) & (dist <= GUARD)] = rgb("#c3c2b7")
    classes[ship] = rgb(COLOR["ship"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.9), facecolor=SURFACE)
    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for a in anns:
        x, y, w, h = a["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#eda100", lw=1.5, ls="--"))
    for p in polys:
        ax.add_patch(plt.Polygon(p, closed=True, fill=False, ec=COLOR["ship"], lw=1.5))
    ax.legend(handles=[Line2D([], [], color="#eda100", ls="--", lw=1.5, label="GT box"),
                       Line2D([], [], color=COLOR["ship"], lw=1.5, label="segmentation polygon")],
              loc="lower left", fontsize=8.5, framealpha=0.85)
    ax.set_title("(a) image with HRSID labels", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    ax.imshow(classes)
    ax.legend(handles=[Patch(color=COLOR["ship"], label="ship (inside polygon)"),
                       Patch(color="#c3c2b7", label=f"guard band (≤ {GUARD} px, not used)"),
                       Patch(color=COLOR["sea"], label="sea")],
              loc="upper right", fontsize=8.5, framealpha=0.85)
    ax.set_title("(b) pixel classes", fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "label_example.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    for a in anns:
        print(stem, "box", a["bbox"], "polygon fills", round(a["area"] / (a["bbox"][2] * a["bbox"][3]), 2), "of the box")


if __name__ == "__main__":
    main()
