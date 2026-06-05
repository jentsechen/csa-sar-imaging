import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from generate_point_target import read_sar_slice, SliceParams


FILE_NAME = "2023-04-12-13-00-32_UMBRA-04_GEC.tif"
# FILE_NAME = "2023-10-06-02-14-22_UMBRA-04_GEC.tif"
SCENE_NAME = "tsoying_naval_base"
OUT_DIR = f"{SCENE_NAME}_analysis"


def analyze(sar_slice: np.ndarray, label: str):
    flat = sar_slice.flatten().astype(np.float64)
    nonzero = flat[flat != 0]

    skewness = stats.skew(nonzero)
    kurtosis = stats.kurtosis(nonzero)
    sample = nonzero if len(nonzero) <= 5000 else np.random.choice(nonzero, 5000, replace=False)
    _, p_norm = stats.normaltest(sample)
    k_mom_amp = nonzero.mean() ** 2 / nonzero.var()
    a_mle_amp, _, _ = stats.gamma.fit(nonzero, floc=0)
    intensity = nonzero ** 2
    k_mom_int = intensity.mean() ** 2 / intensity.var()
    a_mle_int, _, _ = stats.gamma.fit(intensity, floc=0)
    log_vals = np.log(nonzero)
    _, p_log_norm = stats.normaltest(np.random.choice(log_vals, min(5000, len(log_vals)), replace=False))

    print(f"\n[{label}] {sar_slice.dtype} {sar_slice.shape}  total={flat.size:,} nonzero={nonzero.size:,}")
    print(f"  min={nonzero.min():.2f}  max={nonzero.max():.2f}  mean={nonzero.mean():.2f}  std={nonzero.std():.2f}  median={np.median(nonzero):.2f}  p1/p99={np.percentile(nonzero,1):.2f}/{np.percentile(nonzero,99):.2f}")
    print(f"  skew={skewness:.3f}  kurt={kurtosis:.3f}  norm-p={p_norm:.2e}  lognorm-p={p_log_norm:.2e}")
    print(f"  amp-gamma  MoM k={k_mom_amp:.3f}  MLE k={a_mle_amp:.3f} (~{round(a_mle_amp)} looks)")
    print(f"  int-gamma  MoM k={k_mom_int:.3f}  MLE k={a_mle_int:.3f} (~{round(a_mle_int)} looks)")

    return nonzero


def _out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


def plot_slice(sar_slice: np.ndarray, scene_name: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.flipud(sar_slice), origin="lower", cmap="viridis")
    ax.set_xlabel("range")
    ax.set_ylabel("azimuth")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = _out(f"{scene_name}_slice.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def check_gamma_fit(nonzero: np.ndarray, label: str) -> dict:
    intensity = nonzero ** 2
    a, loc, scale = stats.gamma.fit(intensity, floc=0)

    fit_mean = a * scale
    fit_var = a * scale ** 2
    sample_mean = intensity.mean()
    sample_var = intensity.var()

    ks_stat, ks_p = stats.kstest(intensity, "gamma", args=(a, loc, scale))

    print(f"  [{label}] k={a:.4f}  θ={scale:.4f}  mean={sample_mean:.2f}/{fit_mean:.2f}={fit_mean/sample_mean:.4f}  var={sample_var:.2f}/{fit_var:.2f}={fit_var/sample_var:.4f}  KS={ks_stat:.4f}  p={ks_p:.2e}")

    return {
        "shape": a, "scale": scale,
        "fit_mean": fit_mean, "sample_mean": sample_mean,
        "fit_var": fit_var, "sample_var": sample_var,
        "ks_stat": ks_stat, "ks_p": ks_p,
    }


def run_scene(sp: SliceParams, label: str) -> dict:
    sar_slice = read_sar_slice(FILE_NAME, sp.row_start, sp.row_len, sp.col_start, sp.col_len)
    plot_slice(sar_slice, label)
    nonzero = analyze(sar_slice, label=label)
    return check_gamma_fit(nonzero, label)


SCENE_DESCS = {
    "scene-A": "Open sea area — no ship objects present",
    "scene-B": "Harbor area — ship objects present",
}


def write_comparison_md(scenes: list):
    (la, spa, _), (lb, spb, _) = scenes

    lines = [
        f"# {SCENE_NAME} — Scene Comparison\n",
        f"File: `{FILE_NAME}`\n",
        "## Overview\n",
        "The Gamma distribution is fitted to **intensity** values, where intensity = amplitude².",
        "SAR amplitude pixels are first squared before fitting, because intensity (not amplitude)",
        "follows the Gamma distribution under the standard speckle model.\n",
        "The KS statistic measures the maximum gap between the empirical and fitted CDFs (0 = perfect).",
        "The KS p-value is omitted from this report: with large SAR images (100k+ pixels),",
        "the p-value almost always rejects the null hypothesis even for a practically good fit,",
        "making it an unreliable indicator at this scale. Use the KS statistic directly.\n",
        "## Scenes\n",
        f"| **{la}** | **{lb}** |",
        "|---|---|",
        f"| {SCENE_DESCS[la]} | {SCENE_DESCS[lb]} |",
        f"| row\_start={spa.row_start}, row\_len={spa.row_len}, col\_start={spa.col_start}, col\_len={spa.col_len} "
        f"| row\_start={spb.row_start}, row\_len={spb.row_len}, col\_start={spb.col_start}, col\_len={spb.col_len} |",
        f"| ![ {la} slice]({la}_slice.png) | ![{lb} slice]({lb}_slice.png) |\n",
        "## Gamma Fit Comparison\n",
        f"| Metric | {la} | {lb} | Reasonable |",
        "|---|---|---|---|",
    ]

    rows = [
        (
            "shape k",
            lambda r: f"{r['shape']:.4f}",
            "~1 (single-look) to ~4 (multi-look)",
        ),
        (
            "scale θ",
            lambda r: f"{r['scale']:.4f}",
            "scene-dependent",
        ),
        (
            "sample mean / fit mean = ratio",
            lambda r: f"{r['sample_mean']:.2f} / {r['fit_mean']:.2f} = {r['fit_mean']/r['sample_mean']:.4f}",
            "ratio ≈ 1.0000",
        ),
        (
            "sample var / fit var = ratio",
            lambda r: f"{r['sample_var']:.2f} / {r['fit_var']:.2f} = {r['fit_var']/r['sample_var']:.4f}",
            "ratio ≈ 1.0000",
        ),
        (
            "KS stat",
            lambda r: f"{r['ks_stat']:.4f}",
            "< 0.05 excellent, < 0.10 good",
        ),
    ]

    for row_name, fmt, reasonable in rows:
        lines.append(f"| {row_name} | " + " | ".join(fmt(r) for _, _, r in scenes) + f" | {reasonable} |")

    path = _out("comparison.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    scene_defs = [
        ("scene-A", SliceParams(row_start=4500, row_len=640, col_start=4500, col_len=720)),
        ("scene-B", SliceParams(row_start=4050, row_len=640, col_start=4180, col_len=720)),
    ]

    scenes = []
    for label, sp in scene_defs:
        result = run_scene(sp, label)
        scenes.append((label, sp, result))

    write_comparison_md(scenes)
