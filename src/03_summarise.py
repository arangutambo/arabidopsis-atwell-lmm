"""Summaries and figures from results/per_phenotype.csv, results/candidates.csv and the thinned scans."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
R, F, D = ROOT / "results", ROOT / "figures", ROOT / "data"
F.mkdir(exist_ok=True)
INK, RUST, GREEN, BROWN, MUTED = "#2A2521", "#8C4B34", "#53624A", "#5C4736", "#5C5448"
COL = {"ols": RUST, "emmax": BROWN, "loco": GREEN}
LAB = {"ols": "OLS (no correction)", "emmax": "mixed model (all-SNP kinship)", "loco": "mixed model (leave-one-chromosome-out)"}
per = pd.read_csv(R / "per_phenotype.csv")
cand = pd.read_csv(R / "candidates.csv")
out = {"n_phenotypes": int(len(per)), "n_min": int(per.n.min()), "n_max": int(per.n.max()), "n_median": float(per.n.median())}

# Dataset shapes, so the report quotes them from here rather than from typed values.
# ATWELL_* are facts about the original paper (Atwell et al. 2010), not computed here.
ATWELL_LINES, ATWELL_SNPS = 199, 216_130
_g = np.load(D / "geno_union.npz")
prev = json.load(open(R / "summary.json")) if (R / "summary.json").exists() else {}
_h5 = D / "1001_SNP_MATRIX" / "imputed_snps_binary.hdf5"
if _h5.exists():
    import h5py
    with h5py.File(_h5, "r") as f:
        release = {"n_snps": int(f["snps"].shape[0]), "n_accessions": int(f["snps"].shape[1])}
else:  # the matrix is large and not redistributed; reuse what a previous run recorded
    release = prev.get("data", {}).get("release", {})
out["data"] = {
    "release": release,
    "n_snps_analysed": int(_g["chrom"].shape[0]),
    "n_accessions_analysed": int(_g["acc"].shape[0]),
    "atwell_lines": ATWELL_LINES,
    "atwell_snps": ATWELL_SNPS,
}
for m in COL:
    l = per[f"lambda_{m}"]
    out[m] = {"lambda_median": round(float(l.median()), 3), "lambda_q1": round(float(l.quantile(.25)), 3), "lambda_q3": round(float(l.quantile(.75)), 3),
              "lambda_max": round(float(l.max()), 2), "frac_lambda_gt_1.2": round(float((l > 1.2).mean()), 3),
              "frac_lambda_lt_0.9": round(float((l < 0.9).mean()), 3),
              "phenotypes_with_hit": int((per[f"hits_{m}"] > 0).sum()), "total_hit_snps": int(per[f"hits_{m}"].sum()),
              "total_loci_200kb": int(per[f"loci_{m}"].sum())}
w = stats.wilcoxon(per.lambda_emmax, per.lambda_loco)
out["wilcoxon_emmax_vs_loco_lambda_p"] = float(w.pvalue)
out["ols_hit_but_no_lmm_hit"] = int(((per.hits_ols > 0) & (per.hits_loco == 0)).sum())
out["lmm_hit_but_no_ols_hit"] = int(((per.hits_ols == 0) & (per.hits_loco > 0)).sum())
out["h2_median"] = round(float(per.h2_reml.median()), 3)
out["h2_frac_near_zero_lt_0.01"] = round(float((per.h2_reml < 0.01).mean()), 3)
out["h2_frac_near_one_gt_0.99"] = round(float((per.h2_reml > 0.99).mean()), 3)
pairs = cand.pivot_table(index=["phenotype", "gene"], columns="method", values="window_empirical_p")
out["n_candidate_pairs"] = int(len(pairs))
out["pairs_empirical_p_lt_0.05"] = {m: int((pairs[m] < 0.05).sum()) for m in COL}
out["pairs_expected_by_chance"] = round(0.05 * len(pairs), 1)
out["pairs_lt_0.05_names_loco"] = [f"{a}~{b}" for (a, b), v in pairs["loco"].items() if v < 0.05]
out["pairs_lt_0.05_names_ols"] = [f"{a}~{b}" for (a, b), v in pairs["ols"].items() if v < 0.05]
out["hit_table_top"] = per.sort_values("hits_ols", ascending=False)[["phenotype", "n", "lambda_ols", "lambda_loco", "hits_ols", "hits_loco"]].head(8).round(2).to_dict("records")
json.dump(out, open(R / "summary.json", "w"), indent=2)
pairs.round(4).to_csv(R / "candidate_pairs_empirical_p.csv")

# --- figure 1: lambda per phenotype by method
fig, ax = plt.subplots(figsize=(6.4, 3.4))
rng = np.random.default_rng(0)
for i, m in enumerate(COL):
    v = per[f"lambda_{m}"].to_numpy()
    ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=9, color=COL[m], alpha=.7)
    ax.hlines(np.median(v), i - .3, i + .3, color=INK, lw=2)
ax.axhline(1, ls="--", color=MUTED, lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["OLS", "mixed model\n(all-SNP kinship)", "mixed model\n(leave-one-chrom-out)"], fontsize=8)
ax.set_ylabel("genomic inflation $\\lambda$"); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
plt.tight_layout(); plt.savefig(F / "lambda.png", dpi=170); plt.close()

# --- figure 2/3: Manhattan and QQ for chosen phenotypes (thinned scans)
GENES = {g: json.load(open(D / f"gene_{g}.json")) for g in ("FLC", "FRI", "FT", "DOG1")}
offs, tot = {}, 0
chrom_len = {1: 30.4e6, 2: 19.7e6, 3: 23.5e6, 4: 18.6e6, 5: 26.9e6}
for c in range(1, 6):
    offs[c] = tot; tot += chrom_len[c]
def manhattan(name, methods=("ols", "loco")):
    z = np.load(R / f"scans_{name.replace(' ', '_')}.npz")
    fig, axes = plt.subplots(len(methods), 1, figsize=(7.2, 1.9 * len(methods) + .3), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        x = np.array([offs[int(c)] for c in z["chrom"]]) + z["pos"]
        y = -np.log10(np.clip(z[m], 1e-300, 1))
        ax.scatter(x / 1e6, y, s=2, color=np.where(z["chrom"] % 2 == 0, BROWN, MUTED), alpha=.6, lw=0)
        sig = y > -np.log10(float(z["alpha"])); ax.scatter(x[sig] / 1e6, y[sig], s=4, color=RUST, lw=0)
        ax.axhline(-np.log10(float(z["alpha"])), ls="--", color=RUST, lw=.8)
        for g, j in GENES.items():
            ax.axvline((offs[int(j["seq_region_name"])] + int(j["start"])) / 1e6, color=GREEN, lw=.8, alpha=.8)
        ax.set_ylabel(f"-log10 p\n{LAB[m].split(' (')[0]}" if m == "ols" else "-log10 p\nmixed (LOCO)", fontsize=7)
        [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
    np.atleast_1d(axes)[-1].set_xlabel("genome position (Mb, chromosomes 1-5 concatenated); green lines: FLC, FRI, FT, DOG1", fontsize=7)
    fig.suptitle(name, fontsize=9, x=0.02, ha="left"); plt.tight_layout(); plt.savefig(F / f"manhattan_{name.replace(' ', '_')}.png", dpi=170); plt.close()
for nm in ("FT16", "FLC", "Seed Dormancy"):
    if (R / f"scans_{nm.replace(' ', '_')}.npz").exists():
        manhattan(nm)
print(json.dumps({k: out[k] for k in ("n_phenotypes", "ols", "emmax", "loco", "pairs_empirical_p_lt_0.05", "pairs_expected_by_chance")}, indent=1))
