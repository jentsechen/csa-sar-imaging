import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from generate_point_target import read_sar_slice, SliceParams


FILE_NAME = "2023-04-12-13-00-32_UMBRA-04_GEC.tif"
SCENE_NAME = "tsoying_naval_base"
OUT_DIR = f"{SCENE_NAME}_analysis"


def analyze(sar_slice: np.ndarray, label: str):
    flat = sar_slice.flatten().astype(np.float64)
    nonzero = flat[flat != 0]

    print(f"\n=== {label} ===")
    print(f"dtype:    {sar_slice.dtype}")
    print(f"shape:    {sar_slice.shape}")
    print(f"total px: {flat.size:,}  non-zero: {nonzero.size:,}")
    print(f"min:      {nonzero.min():.4f}")
    print(f"max:      {nonzero.max():.4f}")
    print(f"mean:     {nonzero.mean():.4f}")
    print(f"std:      {nonzero.std():.4f}")
    print(f"median:   {np.median(nonzero):.4f}")
    print(f"p1/p99:   {np.percentile(nonzero, 1):.4f} / {np.percentile(nonzero, 99):.4f}")

    skewness = stats.skew(nonzero)
    kurtosis = stats.kurtosis(nonzero)
    sample = nonzero if len(nonzero) <= 5000 else np.random.choice(nonzero, 5000, replace=False)
    _, p_norm = stats.normaltest(sample)
    print(f"skewness: {skewness:.4f}  kurtosis: {kurtosis:.4f}  normal-test p: {p_norm:.2e}")

    # Gamma fit — amplitude interpretation (values are amplitude)
    k_mom_amp = nonzero.mean() ** 2 / nonzero.var()
    a_mle_amp, _, _ = stats.gamma.fit(nonzero, floc=0)
    print(f"\nAmplitude Gamma fit  -> MoM k={k_mom_amp:.3f}  MLE k={a_mle_amp:.3f}  (~{round(a_mle_amp)} looks)")

    # Gamma fit — intensity interpretation (values^2 are intensity)
    intensity = nonzero ** 2
    k_mom_int = intensity.mean() ** 2 / intensity.var()
    a_mle_int, _, _ = stats.gamma.fit(intensity, floc=0)
    print(f"Intensity Gamma fit  -> MoM k={k_mom_int:.3f}  MLE k={a_mle_int:.3f}  (~{round(a_mle_int)} looks)")

    # Normality check on log (consistent with dB storage if Gaussian)
    log_vals = np.log(nonzero)
    _, p_log_norm = stats.normaltest(np.random.choice(log_vals, min(5000, len(log_vals)), replace=False))
    print(f"Log-normal test p:   {p_log_norm:.2e}  (high = consistent with dB/log storage)")

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


def plot_histogram(nonzero: np.ndarray, scene_name: str):
    counts, _ = np.histogram(nonzero, bins=256, range=(nonzero.min(), nonzero.max()))
    plt.figure()
    plt.plot(counts)
    plt.xlabel("pixel value")
    plt.ylabel("count")
    plt.title(f"{scene_name} — raw non-zero histogram")
    plt.grid()
    plt.tight_layout()
    path = _out(f"{scene_name}_raw_hist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_gamma_fit(nonzero: np.ndarray, scene_name: str):
    intensity = nonzero ** 2
    a, loc, scale = stats.gamma.fit(intensity, floc=0)

    x = np.linspace(intensity.min(), np.percentile(intensity, 99), 500)
    pdf = stats.gamma.pdf(x, a, loc=loc, scale=scale)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(intensity, bins=256, density=True, alpha=0.7, label="data")
    axes[0].plot(x, pdf, "r-", linewidth=2, label=f"Gamma fit k={a:.2f}")
    axes[0].set_xlim(0, np.percentile(intensity, 99))
    axes[0].set_xlabel("intensity (amplitude²)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Intensity — Gamma fit")
    axes[0].legend()
    axes[0].grid()

    axes[1].hist(intensity, bins=256, density=True, alpha=0.7, label="data")
    axes[1].plot(x, pdf, "r-", linewidth=2, label=f"Gamma fit k={a:.2f}")
    axes[1].set_xlim(0, np.percentile(intensity, 99))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("intensity (amplitude²)")
    axes[1].set_ylabel("density (log)")
    axes[1].set_title("Intensity — Gamma fit (log y)")
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    path = _out(f"{scene_name}_gamma_fit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    sp = SliceParams(row_start=4500, row_len=800, col_start=4500, col_len=800)
    sar_slice = read_sar_slice(FILE_NAME, sp.row_start, sp.row_len, sp.col_start, sp.col_len)

    plot_slice(sar_slice, SCENE_NAME)
    nonzero = analyze(sar_slice, label="raw SAR slice (all pixels)")
    plot_histogram(nonzero, SCENE_NAME)
    plot_gamma_fit(nonzero, SCENE_NAME)
