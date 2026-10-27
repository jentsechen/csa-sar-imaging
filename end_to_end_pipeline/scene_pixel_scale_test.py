#!/usr/bin/env python3
"""Is an HRSID pixel value linear or logarithmic, amplitude or intensity?

Test data: homogeneous 32x32 sea patches (fully farther than GUARD px from every
HRSID ship polygon, no 255 pixels), all offshore scenes in images/.

(1) Linear vs log. Speckle is multiplicative, so in linear units the patch standard
    deviation grows in proportion to the patch mean (log-log slope k = 1); after a
    log (dB) mapping it becomes additive and the standard deviation does not depend
    on the mean (k = 0). k is the slope of log(std) vs log(mean) over the patches.

(2) Amplitude vs intensity. For L-look speckle (intensity ~ Gamma(L)), two moments of
    the same patch give two estimates of L. Under the right hypothesis they agree:
        amplitude A: E[A^2]/E[A]^2 = L Gamma(L)^2 / Gamma(L+1/2)^2,  Var(ln A) = psi1(L)/4
        intensity I: E[I^2]/E[I]^2 = 1 + 1/L,                        Var(ln I) = psi1(L)
    Sea texture (K-distributed clutter) biases both; the comparison is still made
    against synthetic patches with the same mean level, integer quantization and
    JPEG compression, generated as amplitude, as intensity and as dB.

Writes end_to_end_pipeline/scene_stats/pixel_scale.json. Usage: python3 scene_pixel_scale_test.py
"""
import json
import os

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, polygamma

from scene_statistics import IMAGES, OUT, load_polygons, masks

GUARD = 10
P = 32
JPEG_Q = 90
rng = np.random.default_rng(0)


def l_from_ratio_amp(r):
    f = lambda L: np.exp(np.log(L) + 2 * gammaln(L) - 2 * gammaln(L + 0.5)) - r
    return brentq(f, 0.05, 1e4) if f(0.05) > 0 > f(1e4) else np.nan


def l_from_ratio_int(r):
    return 1.0 / (r - 1.0) if r > 1 else np.nan


def l_from_logvar(v, scale):
    f = lambda L: float(polygamma(1, L)) / scale - v
    return brentq(f, 0.05, 1e4) if f(0.05) > 0 > f(1e4) else np.nan


def patch_stats(x):
    """mean, std, E[x^2]/E[x]^2, Var(ln x) over the nonzero pixels, fraction of zeros."""
    x = x.astype(np.float64)
    m, m2 = x.mean(), (x * x).mean()
    nz = x[x > 0]
    return m, x.std(), m2 / (m * m), np.log(nz).var(), 1.0 - nz.size / x.size


def estimates(rows):
    """Median per-patch moments -> L under each hypothesis."""
    r = np.median([t[2] for t in rows])
    lv = np.median([t[3] for t in rows])
    z = np.mean([t[4] for t in rows])
    return {"ratio": float(r), "var_ln": float(lv), "zero_frac": float(z),
            "amplitude": {"L_from_ratio": float(l_from_ratio_amp(r)), "L_from_log": float(l_from_logvar(lv, 4.0))},
            "intensity": {"L_from_ratio": float(l_from_ratio_int(r)), "L_from_log": float(l_from_logvar(lv, 1.0))}}


def slope(rows):
    m = np.array([t[0] for t in rows])
    s = np.array([t[1] for t in rows])
    k, _ = np.polyfit(np.log(m), np.log(s), 1)
    return float(k)


def real_patches():
    """Patch statistics of the real sea, also grouped by source panorama (P0001, ...)."""
    polys = load_polygons()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    rows = []
    by_pano = {}
    for stem in stems:
        img = cv2.imread(os.path.join(IMAGES, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        ship, dist = masks(img.shape, polys.get(stem, []))
        sea = (~ship) & (dist > GUARD)
        for i in range(0, img.shape[0] - P + 1, P):
            for j in range(0, img.shape[1] - P + 1, P):
                if not sea[i:i + P, j:j + P].all():
                    continue
                x = img[i:i + P, j:j + P]
                # skip saturated patches and no-data areas (image borders filled with 0)
                if x.max() == 255 or np.count_nonzero(x == 0) > x.size // 4:
                    continue
                rows.append(patch_stats(x))
                by_pano.setdefault(stem.split("_")[0], []).append(rows[-1])
    return rows, by_pano


def per_panorama(by_pano, min_patches=200):
    """Which hypothesis gives the more consistent pair of L estimates, panorama by panorama."""
    k, gap_a, gap_i = [], [], []
    for rows in by_pano.values():
        if len(rows) < min_patches:
            continue
        e = estimates(rows)
        a, i = e["amplitude"], e["intensity"]
        k.append(slope(rows))
        gap_a.append(abs(np.log(a["L_from_ratio"] / a["L_from_log"])))
        gap_i.append(abs(np.log(i["L_from_ratio"] / i["L_from_log"])))
    gap_a, gap_i = np.array(gap_a), np.array(gap_i)
    return {"panoramas": len(k), "slope_k_median": float(np.median(k)), "slope_k_min": float(np.min(k)),
            "slope_k_max": float(np.max(k)), "amplitude_more_consistent": int(np.sum(gap_a < gap_i)),
            "median_log_gap_amplitude": float(np.median(gap_a)), "median_log_gap_intensity": float(np.median(gap_i))}


def synthetic(kind, L, real_means, n_img=40):
    """Homogeneous speckle; each patch mean drawn from the real patch means, same filtering."""
    rows = []
    amp_mean = np.exp(gammaln(L + 0.5) - gammaln(L) - 0.5 * np.log(L))   # E[sqrt(I)], E[I] = 1
    for _ in range(n_img):
        level = rng.choice(real_means, size=(800 // P, 800 // P))
        level = np.kron(level, np.ones((P, P)))
        inten = rng.gamma(L, 1.0 / L, size=(800, 800))
        if kind == "amplitude":
            v = level * np.sqrt(inten) / amp_mean
        elif kind == "intensity":
            v = level * inten
        else:                                   # dB of intensity, scaled to the same mean grey level
            db = 10 * np.log10(inten)
            v = level + 4.0 * (db - db.mean())
        img = np.clip(np.rint(v), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        for i in range(0, 800, P):
            for j in range(0, 800, P):
                x = img[i:i + P, j:j + P]
                if x.max() == 255 or np.count_nonzero(x == 0) > x.size // 4:
                    continue
                rows.append(patch_stats(x))
    return rows


def main():
    real, by_pano = real_patches()
    out = {"patches": len(real), "patch": P, "guard_px": GUARD,
           "real": {"slope_k": slope(real), **estimates(real)},
           "per_panorama": per_panorama(by_pano), "synthetic": {}}
    for kind in ("amplitude", "intensity", "db"):
        for L in (1, 2, 4):
            rows = synthetic(kind, L, np.array([t[0] for t in real]))
            out["synthetic"][f"{kind}_L{L}"] = {"slope_k": slope(rows), **estimates(rows)}
    with open(os.path.join(OUT, "pixel_scale.json"), "w") as f:
        json.dump(out, f, indent=1)

    def line(name, d):
        a, i = d["amplitude"], d["intensity"]
        return (f"{name:16s} k={d['slope_k']:5.2f}  ratio={d['ratio']:.3f} varln={d['var_ln']:.3f} zero={d['zero_frac']:.4f} | "
                f"amp L {a['L_from_ratio']:6.2f} vs {a['L_from_log']:6.2f} | int L {i['L_from_ratio']:6.2f} vs {i['L_from_log']:6.2f}")
    print(f"real sea patches: {len(real)}")
    print(line("REAL", out["real"]))
    print("per panorama:", out["per_panorama"])
    for k, d in out["synthetic"].items():
        print(line(k, d))


if __name__ == "__main__":
    main()
