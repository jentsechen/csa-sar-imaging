import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_skimage
import os
import json
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def replace_neginf_inplace(arr: np.ndarray) -> np.ndarray:
    arr[arr == -np.inf] = 0.0
    return arr


def normalize_matrix(matrix):
    """Normalize a matrix to [0, 1] range (min-max)."""
    mat_min, mat_max = matrix.min(), matrix.max()
    if mat_max - mat_min == 0:
        return matrix - mat_min
    return (matrix - mat_min) / (mat_max - mat_min)


def normalize_peak_db(img, is_db=True, dynamic_range_db=30):
    """
    Normalize to [0, 1] where 1 = peak and 0 = peak - dynamic_range_db.
    Values below peak - dynamic_range_db are clipped to 0.
    """
    img = replace_neginf_inplace(img.copy())
    img_db = img if is_db else 10.0 * np.log10(img + 1e-12)
    peak = img_db.max()
    return np.clip((img_db - (peak - dynamic_range_db)) / dynamic_range_db, 0.0, 1.0)


# ---------------------------------------------------------------------------
# SSIM and its components
# ---------------------------------------------------------------------------

def ssim_components(img1, img2, sigma=1.5, data_range=1.0):
    """
    Compute the three SSIM component maps and return their image-wide means.

    Uses a Gaussian-weighted local window (sigma=1.5, matching skimage's
    default for window_size=11) to compute local statistics.

    Parameters
    ----------
    img1, img2  : 2D ndarray, both normalised to [0, 1]
    sigma       : Gaussian window sigma (default 1.5)
    data_range  : value range of the inputs (default 1.0)

    Returns
    -------
    L, C, S : scalar means of the luminance, contrast, and structure maps
    """
    C1 = (0.01 * data_range) ** 2   # stabilises luminance near zero
    C2 = (0.03 * data_range) ** 2   # stabilises contrast near zero
    C3 = C2 / 2.0                   # stabilises structure near zero

    f = lambda x: gaussian_filter(x.astype(float), sigma=sigma)

    mu1    = f(img1)
    mu2    = f(img2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = f(img1 ** 2) - mu1_sq
    sigma2_sq = f(img2 ** 2) - mu2_sq
    sigma12   = f(img1 * img2) - mu12

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))

    L = (2.0 * mu12   + C1) / (mu1_sq + mu2_sq          + C1)
    C = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    S = (sigma12 + C3)               / (sigma1 * sigma2  + C3)

    return float(L.mean()), float(C.mean()), float(S.mean())


# ---------------------------------------------------------------------------
# Image saving helpers
# ---------------------------------------------------------------------------

def _save_images_and_histograms(norm_mat1, norm_mat2):
    plt.imshow(norm_mat1, origin='lower', cmap='viridis', aspect=1.0)
    plt.colorbar()
    plt.savefig("golden_image.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.imshow(norm_mat2, origin='lower', cmap='viridis', aspect=1.0)
    plt.colorbar()
    plt.savefig("result_image.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.hist(norm_mat1[norm_mat1 != 0.0].flatten(), bins=100, density=True)
    plt.title("Golden Image Histogram")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("golden_histogram.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.hist(norm_mat2[norm_mat2 != 0.0].flatten(), bins=100, density=True)
    plt.title("Result Image Histogram")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("result_histogram.png", dpi=300, bbox_inches="tight")
    plt.clf()


# ---------------------------------------------------------------------------
# Main metric computation
# ---------------------------------------------------------------------------

def calculate_all_metrics(matrix1, matrix2, save_plots=True):
    """
    Calculate SSIM (and its L/C/S decomposition), PSNR, and MSE.

    Both images are converted to dB scale and peak-to-peak−30 dB normalised
    before metric computation, so that the same physical brightness level maps
    to the same value in both images.

    Parameters
    ----------
    matrix1    : 2D ndarray, golden/reference image (linear scale)
    matrix2    : 2D ndarray, result image (dB scale)
    save_plots : save image/histogram PNGs, default True

    Returns
    -------
    dict with keys: ssim, luminance, contrast, structure, psnr_db, mse
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}")

    m1 = replace_neginf_inplace(matrix1.copy())
    m2 = replace_neginf_inplace(matrix2.copy())

    if save_plots:
        _save_images_and_histograms(normalize_matrix(m1), normalize_matrix(m2))

    norm_m1 = normalize_peak_db(m1, is_db=False)
    norm_m2 = normalize_peak_db(m2, is_db=True)

    L, C, S = ssim_components(norm_m1, norm_m2)
    score   = ssim_skimage(norm_m1, norm_m2, data_range=1.0)

    mse = float(np.mean((norm_m1 - norm_m2) ** 2))
    psnr = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    return {
        'ssim':      0.0 if np.isnan(score) else float(score),
        'luminance': L,
        'contrast':  C,
        'structure': S,
        'psnr_db':   psnr,
        'mse':       mse,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_metrics(label, m):
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  {'Metric':<24} {'Result':>12}")
    print(f"  {'-'*36}")
    print(f"  {'SSIM':<24} {m['ssim']:>12.5f}")
    print(f"  {'  Luminance (L)':<24} {m['luminance']:>12.5f}")
    print(f"  {'  Contrast  (C)':<24} {m['contrast']:>12.5f}")
    print(f"  {'  Structure (S)':<24} {m['structure']:>12.5f}")
    print(f"  {'-'*36}")
    print(f"  {'PSNR (dB)':<24} {m['psnr_db']:>12.2f}")
    print(f"  {'MSE [0,1]':<24} {m['mse']:>12.5f}")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def _fmt_tex(val, fmt=".4f"):
    if isinstance(val, float) and np.isnan(val):
        return "---"
    if isinstance(val, float) and np.isinf(val):
        return r"$\infty$"
    return f"{val:{fmt}}"


def write_metrics_tex(metrics_by_label, tex_path="performance_metrics_table.tex"):
    """
    Write (or overwrite) a LaTeX table file for \input{} in simulation_report.tex.

    Parameters
    ----------
    metrics_by_label : dict[label -> calculate_all_metrics() result]
    tex_path         : output .tex file path
    """
    labels   = list(metrics_by_label.keys())
    col_spec = "l" + "c" * len(labels)
    header   = " & ".join(labels)

    def row(name, vals):
        return "        " + name + " & " + " & ".join(vals) + r" \\"

    rows = [
        row(r"SSIM",
            [_fmt_tex(metrics_by_label[l]['ssim'],      ".5f") for l in labels]),
        row(r"Luminance $L$",
            [_fmt_tex(metrics_by_label[l]['luminance'], ".5f") for l in labels]),
        row(r"Contrast $C$",
            [_fmt_tex(metrics_by_label[l]['contrast'],  ".5f") for l in labels]),
        row(r"Structure $S$",
            [_fmt_tex(metrics_by_label[l]['structure'], ".5f") for l in labels]),
        row(r"PSNR (dB)",
            [_fmt_tex(metrics_by_label[l]['psnr_db'],   ".2f") for l in labels]),
        row(r"MSE $[0,1]$",
            [_fmt_tex(metrics_by_label[l]['mse'],       ".5f") for l in labels]),
    ]

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Performance metrics.}\n")
        f.write(f"    \\begin{{tabular}}{{{col_spec}}}\n")
        f.write("        \\toprule\n")
        f.write(f"        Metric & {header} \\\\\n")
        f.write("        \\midrule\n")
        for r in rows:
            f.write(r + "\n")
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[metrics] Written to {tex_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    golden_path  = "./point_target_location/tsoying_naval_base.json"
    result_paths = {
        "result (threshold = 0)": "./focused_image/tsoying_naval_base_mag_db.npy",
        "result (threshold = 1e4)": "./focused_image/tsoying_naval_base_iter_0_mag_db.npy",
    }

    if not os.path.exists(golden_path):
        print(f"Golden file not found: {golden_path}")
        exit(1)

    with open(golden_path, 'r') as f:
        golden_img = np.flipud(np.array(json.load(f), dtype=float))
    n_row, n_col = golden_img.shape

    metrics_by_label = {}
    for label, result_path in result_paths.items():
        if not os.path.exists(result_path):
            print(f"\n[{label}] File not found, skipping: {result_path}")
            continue

        result_img = np.load(result_path).astype(float)[
            int(n_row * 3 / 2):int(n_row * 5 / 2),
            int(n_col * 3 / 2):int(n_col * 5 / 2),
        ]
        save = (label == next(iter(result_paths)))
        m = calculate_all_metrics(golden_img, result_img, save_plots=save)
        metrics_by_label[label] = m
        _print_metrics(label, m)

    if metrics_by_label:
        write_metrics_tex(metrics_by_label,
                          tex_path="../document/simulation_report/performance_metrics_table.tex")

    # --- Side-by-side comparison map ---
    images = {"golden": normalize_matrix(replace_neginf_inplace(golden_img.copy()))}
    for label, result_path in result_paths.items():
        if not os.path.exists(result_path):
            continue
        result_img = np.load(result_path).astype(float)[
            int(n_row * 3 / 2):int(n_row * 5 / 2),
            int(n_col * 3 / 2):int(n_col * 5 / 2),
        ]
        images[label] = normalize_peak_db(result_img, is_db=True)

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images.items()):
        im = ax.imshow(img, origin='lower', cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Image Comparison  (peak to peak−30 dB)", fontsize=12)
    plt.tight_layout()
    map_path = "../diagram/thresholding/comparison_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches="tight")
    plt.clf()
    print(f"[map] Saved {map_path}")