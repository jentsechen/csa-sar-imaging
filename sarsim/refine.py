"""Sparse refinements of a CSA image (fully sampled echo).

The raw-echo model 1/2 ||Y - CSA^-1(H)||^2 + R(H) equals 1/2 ||H_CSA - H||^2 + R(H)
because the CSA is unitary (review response, Appendix A), so every method works
on the complex CSA image directly. All thresholds are t = alpha * peak, with
peak = max |H_CSA| over the scene crop, so they do not depend on scene brightness.

    l0     R = lambda ||H||_0       per-pixel hard threshold at t (global optimum)
    l1     R = lambda ||H||_1       per-pixel soft threshold at t
    l12tv  R = lambda1 ||H||_1/2 + lambda2 TV(|H|)   Liu et al., Remote Sens. 16(8)
           1459, 2024, Table 1 with full sampling; TV couples neighbouring pixels.
"""
import numpy as np

CBRT54_4 = 54.0 ** (1.0 / 3.0) / 4.0


def phase(xp, x):
    """x / |x|, 0 where x = 0."""
    mag = xp.abs(x)
    return x / xp.maximum(mag, 1e-300) * (mag > 0)


def l0(xp, h, t):
    """Hard threshold: keep h where |h| > t."""
    return h * (xp.abs(h) > t)


def l1(xp, h, t):
    """Complex soft threshold: h / |h| * max(|h| - t, 0)."""
    return phase(xp, h) * xp.maximum(xp.abs(h) - t, 0.0)


def half_threshold(xp, h, t):
    """L1/2 half threshold (Liu et al. Eqs. (21)-(22); Xu et al. 2012) with cut-off magnitude t:
    the minimizer of |z - h|^2 + mu |z|^(1/2), mu = (t / (54^(1/3)/4))^(3/2); phase kept."""
    mu = (t / CBRT54_4) ** 1.5
    mag = xp.abs(h)
    arg = xp.clip(mu / 8.0 * (xp.maximum(mag, 1e-300) / 3.0) ** -1.5, -1.0, 1.0)
    shrunk = 2.0 / 3.0 * mag * (1.0 + xp.cos(2.0 * np.pi / 3.0 - 2.0 / 3.0 * xp.arccos(arg)))
    return phase(xp, h) * xp.where(mag > t, shrunk, 0.0)


def _grad(xp, f):
    """Forward differences, 0 on the last row / column (Liu et al. Eqs. (11)-(12))."""
    g1 = xp.zeros_like(f)
    g2 = xp.zeros_like(f)
    g1[:-1, :] = f[1:, :] - f[:-1, :]
    g2[:, :-1] = f[:, 1:] - f[:, :-1]
    return g1, g2


def _div(xp, p1, p2):
    """Divergence, the negative adjoint of _grad (Liu et al. Eq. (34))."""
    d = xp.zeros_like(p1)
    d[0, :] = p1[0, :]
    d[1:-1, :] = p1[1:-1, :] - p1[:-2, :]
    d[-1, :] = -p1[-2, :]
    d[:, 0] += p2[:, 0]
    d[:, 1:-1] += p2[:, 1:-1] - p2[:, :-2]
    d[:, -1] += -p2[:, -2]
    return d


def l12tv(xp, h, t, xi, lam_tv, delta=0.125, tv_steps=20, n_iter=300, tol=1e-4):
    """L1/2 & TV, Liu et al. Table 1 with full sampling (R(M(I)) = I), xi1 = xi2 = xi:

        X  = (H + xi z1 + xi z2) / (1 + 2 xi)
        z1 = half_threshold(X, t)                       (paper: cut-off = (K+1)-th largest |X|)
        z2 = sign(X) (|X| - lam_tv div p)               (Chambolle dual projection on |X|)

    Deviations from the paper: fixed cut-off t instead of a fixed sparsity K
    (ship sizes vary per scene); no CAMP Onsager term (no measurement residual
    with full sampling); fixed lam_tv (the paper adapts it to the noise
    variance); tv_steps dual steps per iteration with delta = 1/8 (stable).
    Returns (X, n_iter_run, final relative change); n_iter_run == n_iter with a final change
    >= tol means the iteration cap was hit before convergence."""
    x = h
    z1 = xp.zeros_like(h)
    z2 = xp.zeros_like(h)
    p1 = xp.zeros(h.shape)
    p2 = xp.zeros(h.shape)
    for k in range(n_iter):
        x_new = (h + xi * z1 + xi * z2) / (1.0 + 2.0 * xi)
        mag = xp.abs(x_new)
        z1 = half_threshold(xp, x_new, t)
        for _ in range(tv_steps):
            g1, g2 = _grad(xp, _div(xp, p1, p2) - mag / lam_tv)
            p1 += delta * g1
            p2 += delta * g2
            norm = xp.maximum(xp.sqrt(p1 * p1 + p2 * p2), 1.0)
            p1 /= norm
            p2 /= norm
        z2 = phase(xp, x_new) * xp.maximum(mag - lam_tv * _div(xp, p1, p2), 0.0)
        rel = float(xp.linalg.norm(x_new - x)) / max(float(xp.linalg.norm(x)), 1e-300)
        x = x_new
        if rel < tol:
            return x, k + 1, rel
    return x, n_iter, rel


def apply(xp, method, crop, params, margin):
    """Refine a (scene + 2*margin)^2 complex crop with method in {csa, l0, l1, l12tv};
    returns the scene crop (margin removed) and dict(n_iter, rel, converged): the iteration
    count, the last relative change of X and whether it stopped below the tolerance
    (one-step methods: 0 or 1 iteration, always converged)."""
    scene = (slice(margin, crop.shape[0] - margin), slice(margin, crop.shape[1] - margin))
    one_step = lambda n: {"n_iter": n, "rel": 0.0, "converged": True}
    if method == "csa":
        return crop[scene], one_step(0)
    peak = float(xp.abs(crop[scene]).max())
    t = params["alpha"] * peak
    if method == "l0":
        return l0(xp, crop[scene], t), one_step(1)
    if method == "l1":
        return l1(xp, crop[scene], t), one_step(1)
    if method == "l12tv":
        tol = 1e-4
        out, n, rel = l12tv(xp, crop, t, params["xi"], params["tv"] * peak, tol=tol)
        return out[scene], {"n_iter": n, "rel": rel, "converged": rel < tol}
    raise ValueError(method)
