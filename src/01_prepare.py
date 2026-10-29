"""Select the Atwell-study accessions from the 1001 Genomes imputed matrix and filter SNPs.

Reads data/1001_SNP_MATRIX/imputed_snps_binary.hdf5 in row chunks (the file is 10.7M SNPs x 1135 accessions),
keeps the accessions that have at least one Atwell phenotype, drops SNPs with minor-allele frequency < 5% in
that set, and writes a compact int8 matrix to data/geno_union.npz (not committed to git).
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
MAF_MIN = 0.05

pheno = pd.read_csv(D / "values_study1.csv")
acc_pheno = pheno["accession_id"].astype(int).unique()
h = h5py.File(D / "1001_SNP_MATRIX" / "imputed_snps_binary.hdf5", "r")
acc_geno = np.array([int(a) for a in h["accessions"][:]])
keep_acc = np.array(sorted(set(acc_pheno) & set(acc_geno)))
col = np.array([np.where(acc_geno == a)[0][0] for a in keep_acc])
order = np.argsort(col)  # columns are taken from a full chunk, so any order works; keep file order for clarity
col, keep_acc = col[order], keep_acc[order]
n = len(keep_acc)
print(f"accessions with phenotypes: {len(acc_pheno)}; in genotype file: {len(keep_acc)}")

pos_all = h["positions"][:]
regions = h["positions"].attrs["chr_regions"]
chrom_all = np.zeros(len(pos_all), dtype=np.int8)
for i, (a, b) in enumerate(regions):
    chrom_all[a:b] = i + 1

snps, chrom, pos = [], [], []
CH = 200_000
total = len(pos_all)
for s in range(0, total, CH):
    block = h["snps"][s : s + CH][:, col]  # (rows, n) int8
    f = block.mean(axis=1)
    ok = (f >= MAF_MIN) & (f <= 1 - MAF_MIN)
    if ok.any():
        snps.append(block[ok])
        chrom.append(chrom_all[s : s + CH][ok])
        pos.append(pos_all[s : s + CH][ok])
    if (s // CH) % 10 == 0:
        print(f"  {s / total:5.1%} read, kept {sum(len(c) for c in chrom):,}", flush=True)
X = np.concatenate(snps)
chrom = np.concatenate(chrom)
pos = np.concatenate(pos)
print(f"SNPs kept (MAF >= {MAF_MIN} among {n} accessions): {len(pos):,} of {total:,}")
np.savez(D / "geno_union.npz", X=X, chrom=chrom, pos=pos, acc=keep_acc)
print("saved", (D / "geno_union.npz").stat().st_size // 1_000_000, "MB")
