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

    # KL divergence: KL(empirical || Gamma fit) via histogram binning
    n_bins = 200
    upper = np.percentile(intensity, 99.5)
    counts, edges = np.histogram(intensity, bins=n_bins, range=(0, upper))
    bin_width = edges[1] - edges[0]
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    p = counts / counts.sum()                                    # empirical pmf
    q = stats.gamma.pdf(midpoints, a, loc=loc, scale=scale) * bin_width
    q = q / q.sum()                                             # normalise to pmf
    mask = p > 0
    kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))    # KL(P||Q) in nats

    print(f"  [{label}] k={a:.4f}  θ={scale:.4f}  mean={sample_mean:.2f}/{fit_mean:.2f}={fit_mean/sample_mean:.4f}  var={sample_var:.2f}/{fit_var:.2f}={fit_var/sample_var:.4f}  KS={ks_stat:.4f}  KL={kl:.4f}")

    return {
        "shape": a, "scale": scale,
        "fit_mean": fit_mean, "sample_mean": sample_mean,
        "fit_var": fit_var, "sample_var": sample_var,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "kl": kl,
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
        "## Background\n",
        "### Gamma Distribution for SAR Intensity\n",
        "Under the standard SAR speckle model, the complex return of each resolution cell is the",
        "coherent sum of many independent scatterers. For a detected (amplitude) image, the",
        "**intensity** $I = a^2$ (amplitude squared) follows a Gamma distribution:\n",
        "$$I \\sim \\text{Gamma}(k,\\, \\theta)$$\n",
        "with PDF:\n",
        "$$f(I;\\,k,\\theta) = \\frac{I^{k-1}\\,e^{-I/\\theta}}{\\theta^k\\,\\Gamma(k)}, \\quad I > 0$$\n",
        "The shape parameter $k$ equals the **number of looks** $L$, and $\\theta = \\sigma^2 / L$",
        "where $\\sigma^2$ is the mean backscatter power.  Moments of the fit are:\n",
        "$$\\text{Mean} = k\\theta, \\qquad \\text{Variance} = k\\theta^2$$\n",
        "Parameters are estimated by **Maximum Likelihood (MLE)**, maximising:\n",
        "$$\\ell(k,\\theta) = \\sum_{i=1}^{n}\\left[(k-1)\\ln I_i - \\frac{I_i}{\\theta} - k\\ln\\theta - \\ln\\Gamma(k)\\right]$$\n",
        "with the location fixed at zero (`floc=0`) because intensity cannot be negative.\n",
        "### KS Statistic\n",
        "The Kolmogorov–Smirnov statistic measures the largest vertical gap between the empirical CDF $F_n$",
        "and the theoretical CDF $F$:\n",
        "$$D_n = \\sup_x \\left| F_n(x) - F(x) \\right|$$\n",
        "where $F_n(x) = \\frac{1}{n}|\\{i : I_i \\le x\\}|$.  $D_n \\in [0, 1]$;",
        "smaller is a better fit.  The p-value is omitted here: with large SAR images",
        "($n > 10^5$), even a trivially small $D_n$ yields $p < 0.05$, making the",
        "p-value an unreliable indicator at this scale.  Use $D_n$ directly.\n",
        "### KL Divergence\n",
        "The Kullback–Leibler divergence measures how much the empirical distribution $P$",
        "differs from the fitted theoretical distribution $Q$:\n",
        "$$D_{\\mathrm{KL}}(P \\| Q) = \\sum_{i} p_i \\ln \\frac{p_i}{q_i}$$\n",
        "where $p_i$ and $q_i$ are the empirical and Gamma probabilities in each histogram bin",
        "(200 bins over the 0–99.5th percentile range, both normalised to sum to 1).",
        "Units are **nats** (natural units of information: the result of using $\\ln$ instead of $\\log_2$;",
        "1 nat $= \\log_2 e \\approx 1.4427$ bits). $D_{\\mathrm{KL}} = 0$ means perfect agreement;",
        "values below 0.05 nats indicate a good fit.\n",
        "KL is complementary to KS: KS detects the single worst-case gap anywhere in the",
        "distribution, while KL accumulates evidence of mismatch across all bins,",
        "penalising heavy-tailed deviations more strongly.\n",
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
            "KS statistics",
            lambda r: f"{r['ks_stat']:.4f}",
            "< 0.05 excellent, < 0.10 good",
        ),
        (
            "KL divergence (nats)",
            lambda r: f"{r['kl']:.4f}",
            "< 0.05 good, < 0.01 excellent",
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
