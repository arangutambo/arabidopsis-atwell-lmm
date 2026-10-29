"""Association scans for all Atwell 2010 phenotypes: OLS, EMMAX-style LMM, and LMM with leave-one-chromosome-out kinship.

    python src/02_scan.py              # all phenotypes (~tens of minutes)
    python src/02_scan.py FT16 FLC     # selected phenotypes

Writes results/per_phenotype.csv, results/candidates.csv and results/scans_<phenotype>.npz (thinned, for plots).
Candidate gene/trait pairs are fixed here, before looking at any result.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lmm import genomic_inflation, inverse_normal, kinship, scan_lmm, scan_ols  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D, R = ROOT / "data", ROOT / "results"
R.mkdir(exist_ok=True)
WIN = 50_000
GENES = {}
for g in ("FLC", "FRI", "FT", "DOG1"):
    j = json.load(open(D / f"gene_{g}.json"))
    GENES[g] = (int(j["seq_region_name"]), int(j["start"]), int(j["end"]))
# Pre-specified pairs: flowering traits vs FLC, FRI, FT; dormancy traits vs DOG1.
FLOWERING = ["FT10", "FT16", "FT22", "LN10", "LN16", "LN22", "FLC", "SDV", "FT GH", "FT Field", "0W GH FT", "8W GH FT"]
DORMANCY = ["Seed Dormancy", "DSDS50"]
PAIRS = [(t, g) for t in FLOWERING for g in ("FLC", "FRI", "FT")] + [(t, "DOG1") for t in DORMANCY]
PLOT = {"FT16", "FLC", "Seed Dormancy", "FT10"}

z = np.load(D / "geno_union.npz")
X, chrom, pos, acc = z["X"], z["chrom"], z["pos"], z["acc"]
M = X.shape[0]
ALPHA = 0.05 / M
thin = np.arange(0, M, 20)
Zt = X[thin].T.astype(np.float64)  # (n_all, m_thin) for kinship
chrom_thin = chrom[thin]
sl = {c: np.where(chrom == c)[0] for c in range(1, 6)}
assert all((np.diff(ix) == 1).all() for ix in sl.values()), "SNPs must be contiguous by chromosome"

pheno = pd.read_csv(D / "values_study1.csv")
cols = [c for c in pheno.columns if c not in ("accession_id", "replicate_id")]
byacc = pheno.groupby("accession_id")[cols].mean().reindex(acc)
want = sys.argv[1:] or cols


rng_w = np.random.default_rng(12345)
chrom_start = {c: (int(pos[sl[c]].min()), int(pos[sl[c]].max())) for c in range(1, 6)}
chrom_w = np.array([sl[c].size for c in range(1, 6)], dtype=float)
NULL_WINDOWS = {}


def null_windows(width, n=2000):
    """Random windows of the same bp width, drawn across the genome in proportion to SNP count; returns SNP index bounds."""
    key = int(width)
    if key not in NULL_WINDOWS:
        cs = rng_w.choice(np.arange(1, 6), size=n, p=chrom_w / chrom_w.sum())
        lo_i, hi_i = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        for k, c in enumerate(cs):
            a, b = chrom_start[c]
            st = int(rng_w.integers(a, max(a + 1, b - key)))
            ix = sl[c]
            lo_i[k] = ix[0] + np.searchsorted(pos[ix], st)
            hi_i[k] = ix[0] + np.searchsorted(pos[ix], st + key, side="right")
        NULL_WINDOWS[key] = (lo_i, hi_i)
    return NULL_WINDOWS[key]


def window_empirical_p(p, w_idx, width):
    """Share of random same-width windows whose minimum p is at least as small as the candidate window's."""
    lo_i, hi_i = null_windows(width)
    obs = np.nanmin(p[w_idx])
    null = np.array([np.nanmin(p[a:b]) if b > a else 1.0 for a, b in zip(lo_i, hi_i)])
    return float((1 + np.sum(null <= obs)) / (1 + len(null)))


def hits(p):
    ok = np.isfinite(p) & (p < ALPHA)
    if not ok.any():
        return 0, 0
    c, s = chrom[ok], pos[ok]
    return int(ok.sum()), len({(int(a), int(b) // 200_000) for a, b in zip(c, s)})


rows, cand = [], []
t0 = time.time()
for name in want:
    y_all = byacc[name].to_numpy()
    idx = np.where(np.isfinite(y_all))[0]
    n = len(idx)
    y = inverse_normal(y_all[idx])
    Xs = X[:, idx]
    K = kinship(Zt[idx])
    p_ols = scan_ols(Xs, y)[2]
    _, _, p_emx, delta = scan_lmm(Xs, y, K)
    p_loco = np.full(M, np.nan)
    for c in range(1, 6):
        Kc = kinship(Zt[idx][:, chrom_thin != c])
        p_loco[sl[c]] = scan_lmm(Xs[sl[c]], y, Kc)[2]
    rec = {"phenotype": name, "n": n, "h2_reml": delta / (1 + delta)}
    for lab, p in (("ols", p_ols), ("emmax", p_emx), ("loco", p_loco)):
        nh, nl = hits(p)
        rec[f"lambda_{lab}"], rec[f"hits_{lab}"], rec[f"loci_{lab}"] = genomic_inflation(p), nh, nl
        rec[f"minp_{lab}"] = float(np.nanmin(p[p > 0])) if np.any(p > 0) else np.nan
    rows.append(rec)
    for t, g in PAIRS:
        if t != name:
            continue
        c, a, b = GENES[g]
        w = np.where((chrom == c) & (pos >= a - WIN) & (pos <= b + WIN))[0]
        for lab, p in (("ols", p_ols), ("emmax", p_emx), ("loco", p_loco)):
            pm = float(np.nanmin(p[w]))
            cand.append({"phenotype": name, "gene": g, "method": lab, "min_p_in_window": pm,
                         "window_empirical_p": window_empirical_p(p, w, (b - a) + 2 * WIN),
                         "n_snps_in_window": len(w)})
    if name in PLOT:
        keep = np.where((p_ols < 1e-3) | (p_loco < 1e-3) | (np.arange(M) % 50 == 0))[0]
        np.savez_compressed(R / f"scans_{name.replace(' ', '_')}.npz", idx=keep, chrom=chrom[keep], pos=pos[keep],
                            ols=p_ols[keep], emmax=p_emx[keep], loco=p_loco[keep], alpha=ALPHA)
    print(f"{name:<24} n={n:3d} lambda ols {rec['lambda_ols']:.2f} emmax {rec['lambda_emmax']:.2f} loco {rec['lambda_loco']:.2f} "
          f"hits {rec['hits_ols']}/{rec['hits_emmax']}/{rec['hits_loco']}  [{time.time() - t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(R / "per_phenotype.csv", index=False)
    pd.DataFrame(cand).to_csv(R / "candidates.csv", index=False)
print(f"M = {M:,} SNPs; Bonferroni alpha = {ALPHA:.2e}")
