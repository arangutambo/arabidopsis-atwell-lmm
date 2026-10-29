# Does a mixed model keep a plant GWAS calibrated? A re-analysis of Atwell et al. (2010)

Re-analysis of the 107 *Arabidopsis thaliana* phenotypes of [Atwell et al. 2010](https://doi.org/10.1038/nature08800)
with 1.83 million imputed SNPs from the 1001 Genomes, comparing ordinary regression with an EMMAX-style mixed model
(all-SNP kinship, and leave-one-chromosome-out kinship). Python, NumPy, SciPy.

**Report:** [`report/report.pdf`](report/report.pdf) (source `report/report.qmd`).

## Questions
The original paper already used a mixed model (EMMA), so this is not "applying a mixed model where there was none".
It asks whether (1) the calibration finding replicates with a different implementation and denser SNPs, (2) LOCO kinship
changes anything, and (3) four loci chosen in advance (*FLC*, *FRI*, *FT* for flowering traits; *DOG1* for dormancy) localise.

## Result in brief (107 phenotypes, 51 to 135 lines each)
- **Calibration replicates.** OLS: median genomic inflation 1.46 (76% of phenotypes above 1.2, maximum 3.0). Mixed models: 1.01 (all-SNP) and 1.04 (LOCO), none above 1.13.
- **Hits.** OLS: 52 phenotypes with at least one Bonferroni hit, 48,631 SNPs. Mixed models: 15 to 16 phenotypes, about 2,300 SNPs. 37 phenotypes have OLS hits and no LOCO hit.
- **LOCO vs all-SNP kinship:** nearly identical hits; LOCO's median inflation is slightly higher (1.04 vs 1.01).
- **Known loci:** 8 of 38 pre-specified trait-locus pairs have a window empirical p < 0.05 under the LOCO mixed model (about 1.9 expected by chance; 2 under OLS), mostly flowering traits at *FLC* and *FT*; none at *DOG1*. The pairs are correlated, and only one (FT10 at *FLC*, all-SNP kinship, p = 0.001) clears a Bonferroni correction over 38 pairs.
- **Not an improvement on the paper.** I did not compare hit lists with the paper's supplementary tables. Only 137 of the paper's 199 lines are in the genotype release, so power is lower.

## Reproduce
```bash
micromamba create -y -f environment.yml
bash src/00_download.sh          # AraPheno study 1, 1001 Genomes imputed SNP matrix (~330 MB), Ensembl gene coordinates
python src/01_prepare.py         # select accessions, filter to MAF >= 5%
python src/02_scan.py            # all 107 phenotypes, ~40 min on a laptop
python src/03_summarise.py       # summary.json, figures
python -m pytest                 # tests for the scan code
quarto render report/report.qmd
```

## Data and licence
All data are public. I found **no licence statement for the 1001 Genomes genotype matrix**, so it is not included here;
`src/00_download.sh` fetches it from the source. Code: MIT.

## Provenance
Written and run with the assistance of Claude Code (Anthropic). I am responsible for checking and understanding the results.
No part derives from university coursework or assessed material.
`report/apa.csl` is the APA style from the [CSL styles repository](https://github.com/citation-style-language/styles) (CC BY-SA 3.0).
