import numpy as np
from scipy import stats

from src.lmm import genomic_inflation, kinship, reml_delta, scan_gls, scan_lmm, scan_ols


def make(n=60, m=40, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((m, n)) < rng.uniform(0.2, 0.5, (m, 1))).astype(np.int8)
    y = rng.normal(size=n)
    return X, y


def test_ols_matches_scipy_linregress():
    X, y = make()
    b, se, p = scan_ols(X, y)
    for i in range(X.shape[0]):
        r = stats.linregress(X[i].astype(float), y)
        assert np.isclose(b[i], r.slope) and np.isclose(se[i], r.stderr) and np.isclose(p[i], r.pvalue)


def test_rotated_gls_matches_brute_force_gls():
    X, y = make(n=50, m=15, seed=1)
    K = kinship(X.T)
    lam, U = np.linalg.eigh(K)
    lam = np.clip(lam, 0, None)
    delta = 0.8
    w = 1 / (delta * lam + 1)
    b, se, _ = scan_gls(X, y, U, w)
    V = delta * K + np.eye(len(y))  # proportional to the true covariance
    Vi = np.linalg.inv(V)
    for i in range(X.shape[0]):
        D = np.column_stack([np.ones(len(y)), X[i]])
        A = D.T @ Vi @ D
        beta = np.linalg.solve(A, D.T @ Vi @ y)
        res = y - D @ beta
        sig2 = res @ Vi @ res / (len(y) - 2)
        assert np.isclose(b[i], beta[1])
        assert np.isclose(se[i], np.sqrt(sig2 * np.linalg.inv(A)[1, 1]))


def test_reml_delta_recovers_known_ratio():
    rng = np.random.default_rng(2)
    n, m = 200, 3000
    Z = (rng.random((n, m)) < 0.3).astype(float)
    K = kinship(Z)
    W = (Z - Z.mean(0)) / np.sqrt(Z.mean(0) * (1 - Z.mean(0)))
    ests = []
    for s in range(40):
        g = W @ rng.normal(size=m) / np.sqrt(m) * np.sqrt(0.5)
        y = g + rng.normal(size=n) * np.sqrt(0.5)
        lam, U = np.linalg.eigh(K)
        d = reml_delta(y, np.clip(lam, 0, None), U)
        ests.append(d / (1 + d))  # heritability scale; delta itself has a skewed sampling distribution
    assert abs(np.mean(ests) - 0.5) < 0.08  # true h2 = 0.5


def test_lmm_controls_structure_that_ols_does_not():
    """Two ancestry groups differ in allele frequency and in mean phenotype; no SNP is causal."""
    rng = np.random.default_rng(3)
    n, m = 160, 6000
    grp = np.repeat([0, 1], n // 2)
    f = rng.uniform(0.1, 0.9, m)
    shift = rng.normal(0, 0.12, m)
    P = np.clip(f[None, :] + np.where(grp[:, None] == 1, shift, -shift), 0.02, 0.98)
    Z = (rng.random((n, m)) < P).astype(np.int8)
    y = grp * 1.5 + rng.normal(size=n)  # phenotype follows ancestry
    X = Z.T
    K = kinship(Z[:, ::2].astype(float))
    lam_ols = genomic_inflation(scan_ols(X[1::2], y)[2])
    lam_lmm = genomic_inflation(scan_lmm(X[1::2], y, K)[2])
    assert lam_ols > 3
    assert lam_lmm < lam_ols / 2
