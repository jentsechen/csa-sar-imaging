"""Render the HRSID YOLO11m training mAP@0.5-vs-epoch curve as a PNG figure
for the group meeting report."""

import matplotlib.pyplot as plt

EPOCH = [1, 6, 16, 26, 41, 56, 71, 86, 100]
MAP50 = [0.029, 0.818, 0.854, 0.875, 0.890, 0.902, 0.907, 0.919, 0.922]

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)

ax.plot(EPOCH, MAP50, marker="o", linewidth=2.2, markersize=6)
ax.axhline(MAP50[-1], linestyle="--", linewidth=1.2, alpha=0.7)
ax.annotate(f"{MAP50[-1]:.3f}", xy=(EPOCH[-1], MAP50[-1]),
            xytext=(-8, 8), textcoords="offset points",
            fontsize=11, fontweight="bold")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("mAP@0.5", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, 105)
ax.tick_params(labelsize=11)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)

fig.tight_layout()
fig.savefig("map_vs_epoch.png")
print("Saved map_vs_epoch.png")
