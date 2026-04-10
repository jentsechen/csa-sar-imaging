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
# SCR
# ---------------------------------------------------------------------------

def compute_scr(img, is_db=False, inner_radius=5, outer_radius=15):
    """
    Compute Signal-to-Clutter Ratio (SCR) for a SAR point target image.

    Locates the brightest pixel, treats a circular disk of radius inner_radius
    as the signal zone, and the annular region between inner_radius and
    outer_radius as the clutter zone.

    Parameters
    ----------
    img          : 2D ndarray
    is_db        : True if img is in dB, False if linear amplitude
    inner_radius : radius of signal disk in pixels  (default 5)
    outer_radius : outer radius of clutter annulus in pixels (default 15)

    Returns
    -------
    scr_db : float, SCR in dB
    """
    img = replace_neginf_inplace(img.copy())
    power = 10.0 ** (img / 10.0) if is_db else img ** 2

    r0, c0 = np.unravel_index(np.argmax(power), power.shape)
    rows, cols = np.ogrid[:power.shape[0], :power.shape[1]]
    dist = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2)

    signal_mask  = dist <= inner_radius
    clutter_mask = (dist > inner_radius) & (dist <= outer_radius)

    mean_clutter = power[clutter_mask].mean()
    if mean_clutter == 0:
        return float('inf')
    return 10.0 * np.log10(power[signal_mask].mean() / mean_clutter)


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

def calculate_all_metrics(matrix1, matrix2, save_plots=True,
                          scr_inner_radius=5, scr_outer_radius=15):
    """
    Calculate SSIM (and its L/C/S decomposition), PSNR, MSE, and SCR.

    Both images are converted to dB scale and peak-to-peak−30 dB normalised
    before metric computation, so that the same physical brightness level maps
    to the same value in both images.

    Parameters
    ----------
    matrix1          : 2D ndarray, golden/reference image (linear scale)
    matrix2          : 2D ndarray, result image (dB scale)
    save_plots       : save image/histogram PNGs, default True
    scr_inner_radius : signal disk radius for SCR in pixels (default 5)
    scr_outer_radius : clutter annulus outer radius for SCR in pixels (default 15)

    Returns
    -------
    dict with keys: ssim, luminance, contrast, structure, psnr_db, mse,
                    scr_db_golden, scr_db_result
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

    scr_result = compute_scr(m2, is_db=True,
                             inner_radius=scr_inner_radius,
                             outer_radius=scr_outer_radius)

    return {
        'ssim':          0.0 if np.isnan(score) else float(score),
        'luminance':     L,
        'contrast':      C,
        'structure':     S,
        'psnr_db':       psnr,
        'mse':           mse,
        'scr_db_result': scr_result,
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
    print(f"  {'-'*36}")
    print(f"  {'SCR result (dB)':<24} {m['scr_db_result']:>12.2f}")


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".4f"):
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return "—"
    return f"{val:{fmt}}"


def write_metrics_md(metrics_by_label, md_path="performance_metrics.md"):
    """
    Write (or overwrite) a Markdown file with metric definitions and results table.

    Parameters
    ----------
    metrics_by_label : dict[label -> calculate_all_metrics() result]
    md_path          : output .md file path
    """
    from datetime import datetime
    labels = list(metrics_by_label.keys())

    col_header = " | ".join(labels)
    sep_cols   = " | ".join(["--------"] * len(labels))

    def row(name, vals):
        return f"| {name} | " + " | ".join(vals) + " |"

    table_rows = [
        row("SSIM",
            [_fmt(metrics_by_label[l]['ssim'],           ".5f") for l in labels]),
        row("Luminance (L)",
            [_fmt(metrics_by_label[l]['luminance'],      ".5f") for l in labels]),
        row("Contrast (C)",
            [_fmt(metrics_by_label[l]['contrast'],       ".5f") for l in labels]),
        row("Structure (S)",
            [_fmt(metrics_by_label[l]['structure'],      ".5f") for l in labels]),
        row("PSNR (dB)",
            [_fmt(metrics_by_label[l]['psnr_db'],        ".2f") for l in labels]),
        row("MSE \[0,1\]",
            [_fmt(metrics_by_label[l]['mse'],            ".5f") for l in labels]),
        row("SCR result (dB)",
            [_fmt(metrics_by_label[l]['scr_db_result'],  ".2f") for l in labels]),
    ]

    with open(md_path, "w") as f:
        f.write("# Performance Metrics\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("> All metrics are computed after converting both images to dB scale\n")
        f.write("> and applying **peak-to-peak−30 dB** normalisation to [0, 1].\n\n")

        # --- Definitions ---
        f.write("---\n\n")
        f.write("## Metric Definitions\n\n")

        f.write("### SSIM — Structural Similarity Index\n\n")
        f.write("SSIM measures perceived image similarity by independently comparing local\n")
        f.write("luminance, contrast, and structure within sliding Gaussian-weighted windows\n")
        f.write("(σ = 1.5, equivalent to an 11×11 kernel). The final score is the mean over\n")
        f.write("all window positions. Range: [−1, 1]; **higher is better**.\n\n")
        f.write("$$\\text{SSIM} = L \\cdot C \\cdot S$$\n\n")

        f.write("### L — Luminance\n\n")
        f.write("$$L(x,y) = \\frac{2\\mu_x\\mu_y + C_1}{\\mu_x^2 + \\mu_y^2 + C_1}$$\n\n")
        f.write("Compares the local **mean brightness** μ of each patch.\n")
        f.write("L ≈ 1 when both patches have the same average intensity.\n")
        f.write("C₁ = (0.01 · data\\_range)² stabilises the ratio near zero.\n\n")

        f.write("### C — Contrast\n\n")
        f.write("$$C(x,y) = \\frac{2\\sigma_x\\sigma_y + C_2}{\\sigma_x^2 + \\sigma_y^2 + C_2}$$\n\n")
        f.write("Compares the local **standard deviation** σ (texture energy / variation).\n")
        f.write("C ≈ 1 when both patches have the same spread.\n")
        f.write("A flat patch (σ ≈ 0) against a textured one drives C toward 0.\n")
        f.write("C₂ = (0.03 · data\\_range)².\n\n")

        f.write("### S — Structure\n\n")
        f.write("$$S(x,y) = \\frac{\\sigma_{xy} + C_3}{\\sigma_x\\sigma_y + C_3}$$\n\n")
        f.write("The normalised cross-covariance — equivalent to the **Pearson correlation**\n")
        f.write("between patch values. Measures whether spatial patterns match regardless of\n")
        f.write("brightness or contrast. S = 1 for identical shapes, S = −1 for inverted.\n")
        f.write("C₃ = C₂ / 2.\n\n")

        f.write("### PSNR — Peak Signal-to-Noise Ratio\n\n")
        f.write("$$\\text{PSNR} = 20\\log_{10}\\\left(\\frac{1}{\\sqrt{\\text{MSE}}}\\right) \\text{ dB}$$\n\n")
        f.write("Measures pixel-level fidelity on the normalised images. **Higher is better.**\n\n")

        f.write("### MSE — Mean Squared Error\n\n")
        f.write("$$\\text{MSE} = \\frac{1}{N}\\sum_{i}(x_i - y_i)^2$$\n\n")
        f.write("Computed on the peak−30 dB normalised [0, 1] images. **Lower is better.**\n\n")

        f.write("### SCR — Signal-to-Clutter Ratio\n\n")
        f.write("$$\\text{SCR} = 10\\log_{10}\\\left(\\frac{\\bar{P}_{\\text{signal}}}{\\bar{P}_{\\text{clutter}}}\\right) \\text{ dB}$$\n\n")
        f.write("Computed independently on each image (golden in linear amplitude, result in dB).\n")
        f.write("The brightest pixel is taken as the target centre. An inner disk of radius\n")
        f.write("r₁ = 5 px defines the **signal zone**; an annulus between r₁ and r₂ = 15 px\n")
        f.write("defines the **clutter zone**. Mean power is compared between the two regions.\n")
        f.write("**Higher is better** — a larger SCR means the target stands out more clearly\n")
        f.write("above the background clutter.\n\n")

        # --- Table ---
        f.write("---\n\n")
        f.write("## Results\n\n")
        f.write(f"| Metric | {col_header} |\n")
        f.write(f"|--------|{sep_cols}|\n")
        for r in table_rows:
            f.write(r + "\n")

        f.write("\n")
        f.write("---\n\n")
        f.write("## SSIM Component Interpretation\n\n")
        f.write("| Metric | SAR Interpretation | Role in Object Detection |\n")
        f.write("|:---:|:---:|:---:|\n")
        f.write("| Luminosity (L) | Average Backscatter | Distinguishes overall \"brightness\"; secondary to shape. |\n")
        f.write("| Contrast (C) | Variance/Intensity | Essential for separating target signal from background clutter. |\n")
        f.write("| Structure (S) | Spatial Correlation | Primary metric. Defines the shape and prevents false alarms from noise. |\n")
        f.write("\n")
        f.write("---\n\n")
        f.write("## Image Comparison\n\n")
        f.write("![Image Comparison](../diagram/thresholding/comparison_map.png)\n")

    print(f"\n[metrics] Written to {md_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    golden_path  = "./point_target_location/tsoying_naval_base.json"
    result_paths = {
        "result (threshold = 0)": "./focused_image/tsoying_naval_base_mag_db.npy",
        "result (threshold = 2e5)": "./focused_image/tsoying_naval_base_iter_0_mag_db.npy",
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
        write_metrics_md(metrics_by_label, md_path="../document/performance_metrics.md")

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