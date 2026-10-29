#!/usr/bin/env bash
# Public data only; nothing here is committed to git (no licence statement was found for the genotype matrix).
#  - Phenotypes: AraPheno study 1 (Atwell et al. 2010, 107 phenotypes)
#  - Genotypes: 1001 Genomes imputed SNP matrix (HDF5)
#  - Gene coordinates: Ensembl Plants REST (TAIR10 / Araport annotation)
set -euo pipefail
cd "$(dirname "$0")/../data"
curl -fsSL -o phenotypes_study1.json "https://arapheno.1001genomes.org/rest/study/1/phenotypes.json"
curl -fsSL -o values_study1.csv      "https://arapheno.1001genomes.org/rest/study/1/values.csv"
curl -fL -C - --retry 10 --retry-delay 3 --retry-all-errors -O \
  "https://1001genomes.org/data/GMI-MPI/releases/current/SNP_matrix_imputed_hdf5/1001_SNP_MATRIX.tar.gz"
tar -xzf 1001_SNP_MATRIX.tar.gz
for g in FLC FRI FT DOG1; do
  curl -fsSL -H "Content-Type: application/json" \
    "https://rest.ensembl.org/lookup/symbol/arabidopsis_thaliana/$g?expand=0" -o "gene_$g.json" || echo "no Ensembl record for $g"
done
ls -lh
