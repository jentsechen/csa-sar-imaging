"""Reproducible 1-look speckle, injected on the scatterer coefficients before
echo generation, graded by the signal-to-speckle ratio (SSR).

Rician (partially developed) speckle, one realization per scatterer (1 look):

    a_k = A_k (c + d s_k),  s_k ~ CN(0, 1),  c = sqrt(K/(K+1)),  d = sqrt(1/(K+1)),
    SSR = 10 log10 K   (coherent-to-diffuse power ratio, the Rician K-factor)

E|a_k|^2 = A_k^2 at every SSR; the intensity contrast std/mean of |c + d s|^2 is
sqrt(1 + 2K) / (K + 1): 0 at SSR = inf (no speckle), 1 at SSR = -inf (fully
developed 1-look speckle, Rayleigh amplitude).

Echo generation and CSA are linear, so the focused image is

    H(SSR) = c * CSA(echo(A)) + d * CSA(echo(A s)),

i.e. one clean image per scene plus one speckle image per realization, mixed
with the SSR weights. Both echoes come from the exact echo model in one pass
of sarsim.echo.gen_echo_signals (clean scene and every speckle realization as
coefficient sets of the same targets).

Seeds: s for a scene comes from SeedSequence([base_seed, crc32(stem)]), so the
pattern differs between scenes, is identical for every SSR level and method of
one scene, and does not depend on processing order or subset.
"""
import zlib

import numpy as np


def ssr_weights(ssr_db):
    """(c, d) for an SSR in dB; inf -> (1, 0), -inf -> (0, 1)."""
    if ssr_db == float("inf"):
        return 1.0, 0.0
    if ssr_db == float("-inf"):
        return 0.0, 1.0
    k = 10.0 ** (ssr_db / 10.0)
    return float(np.sqrt(k / (k + 1.0))), float(np.sqrt(1.0 / (k + 1.0)))


def contrast_theory(ssr_db):
    """std/mean of the speckle intensity |c + d s|^2."""
    c, d = ssr_weights(ssr_db)
    return float(np.sqrt(d ** 4 + 2 * c ** 2 * d ** 2))


def scene_rng(stem, base_seed):
    return np.random.default_rng(np.random.SeedSequence([base_seed, zlib.crc32(stem.encode())]))


def speckle_field(stem, shape, base_seed):
    """s ~ CN(0, 1) on the scene grid (complex128, host)."""
    rng = scene_rng(stem, base_seed)
    re, im = rng.standard_normal((2,) + tuple(shape))
    return (re + 1j * im) / np.sqrt(2.0)


def coef_sets(amp, fields):
    """(1 + len(fields), n_targets) coefficients in build_point_target_list order
    (row-major over amp > 0): the clean scene A, then A s for every speckle field."""
    a = np.asarray(amp, dtype=np.float64)
    nz = a > 0
    return np.vstack([a[nz].astype(np.complex128)] + [a[nz] * f[nz] for f in fields])


def field_stats(amp, s, ssr_db):
    """Scatterer-level check: mean and contrast of |c + d s|^2 over pixels with amp > 0."""
    c, d = ssr_weights(ssr_db)
    n = np.abs(c + d * s[amp > 0]) ** 2
    return {"mean": float(n.mean()), "contrast": float(n.std() / n.mean()),
            "contrast_theory": contrast_theory(ssr_db)}
