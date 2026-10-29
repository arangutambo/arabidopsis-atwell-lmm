"""Fast association scans for a quantitative trait on inbred lines: OLS and a linear mixed model (EMMAX-style, P3D).

Model:  y = 1*mu + x*beta + g + e,   g ~ N(0, s_g K),  e ~ N(0, s_e I),  delta = s_g / s_e.
The null variance ratio delta is estimated once by REML (y ~ 1) and then held fixed for every SNP (P3D).
After rotating by the eigenvectors U of K, V is diagonal and each SNP needs only a few weighted sums.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def kinship(Z: np.ndarray) -> np.ndarray:
    """Realised relationship matrix from a (n, m) 0/1 genotype matrix, VanRaden-style (centred and scaled)."""
    p = Z.mean(axis=0)
    keep = (p > 0) & (p < 1)
    W = (Z[:, keep] - p[keep]) / np.sqrt(p[keep] * (1 - p[keep]))
    return (W @ W.T) / W.shape[1]


def reml_delta(y: np.ndarray, lam: np.ndarray, U: np.ndarray) -> float:
    """REML estimate of delta = s_g/s_e under the null y ~ 1 (intercept only)."""
    yt = U.T @ y
    zt = U.T @ np.ones_like(y)
    n = len(y)

    def nll(logd: float) -> float:
        w = 1.0 / (np.exp(logd) * lam + 1.0)
        szz = np.sum(w * zt * zt)
        mu = np.sum(w * zt * yt) / szz
        rss = np.sum(w * (yt - mu * zt) ** 2)
        # profile REML for s_e: 0.5 * [ (n-1) log(rss/(n-1)) + sum log v + log szz ]
        return 0.5 * ((n - 1) * np.log(rss / (n - 1)) - np.sum(np.log(w)) + np.log(szz))

    r = minimize_scalar(nll, bounds=(-8, 8), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def scan_gls(X: np.ndarray, y: np.ndarray, U: np.ndarray, w: np.ndarray, *, chunk: int = 50_000):
    """GLS scan over SNP rows of X (m, n). U: (n, n) rotation, w: (n,) weights 1/(delta*lam+1).

    With U = I and w = 1 this is ordinary least squares. Returns (beta, se, p) arrays of length m.
    """
    n = len(y)
    yt = U.T @ y
    zt = U.T @ np.ones(n)
    szz = np.sum(w * zt * zt)
    szy = np.sum(w * zt * yt)
    syy = np.sum(w * yt * yt)
    wz, wy = w * zt, w * yt
    m = X.shape[0]
    beta, se = np.empty(m), np.empty(m)
    for s in range(0, m, chunk):
        Xt = X[s : s + chunk].astype(np.float64) @ U  # (c, n)
        szx = Xt @ wz
        sxx = (Xt * Xt) @ w
        sxy = Xt @ wy
        det = szz * sxx - szx**2
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (szz * sxy - szx * szy) / det
            mu = (szy - szx * b) / szz
            rss = syy - mu * szy - b * sxy  # weighted RSS of y on [1, x]
            sig2 = rss / (n - 2)
            v = sig2 * szz / det
        beta[s : s + chunk] = b
        se[s : s + chunk] = np.sqrt(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return beta, se, p


def scan_ols(X: np.ndarray, y: np.ndarray, **kw):
    n = len(y)
    return scan_gls(X, y, np.eye(n), np.ones(n), **kw)


def scan_lmm(X: np.ndarray, y: np.ndarray, K: np.ndarray, **kw):
    lam, U = np.linalg.eigh(K)
    lam = np.clip(lam, 0, None)
    delta = reml_delta(y, lam, U)
    w = 1.0 / (delta * lam + 1.0)
    return (*scan_gls(X, y, U, w, **kw), delta)


def genomic_inflation(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


def inverse_normal(y: np.ndarray) -> np.ndarray:
    """Rank-based inverse normal transform (ties get average ranks)."""
    r = stats.rankdata(y)
    return stats.norm.ppf((r - 0.5) / len(y))
