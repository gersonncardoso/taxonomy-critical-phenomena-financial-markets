# Taxonomy of Critical Phenomena in Financial Markets

Reproducibility repository for Paper 1:

> Taxonomy of Critical Phenomena in Financial Markets

## Authors

- Gerson Nassor Cardoso, ORCID: https://orcid.org/0000-0002-8499-3390
- Renato Cesar Sato, ORCID: https://orcid.org/0000-0002-9902-9086

## Scope

This repository contains only the Paper 1 pipeline, manuscript source, figures, configuration, and reproducibility instructions for the descriptive B3 rolling-network study. It excludes Paper 2 materials, including GDELT, news sentiment, MRQAP, and cross-layer news-network outputs.

## Contents

- `paper/`: active Paper 1 LaTeX source, tables, bibliography, and highlights.
- `figures/`: active vector artwork and publication figures cited by Paper 1.
- `scripts/`: scripts used to generate the active validation outputs and figures.
- `src/`: network metric implementations used by the Paper 1 pipeline.
- `configs/`: configuration required for the B3 event calendar and pipeline.
- `data/`: small, publication-relevant validation tables only.
- `zenodo/`: instructions and metadata for the complete large-data archive.

## Large data archive

The complete matrices, 360 PMFG edge lists, filtered edge lists, and large rolling outputs are not stored in this GitHub repository because they exceed practical GitHub repository limits. They are packaged for Zenodo in `deliverables/zenodo_paper1.zip` and should be published there to obtain a persistent DOI.

After Zenodo publication, add the DOI here and in the manuscript Data Availability statement.

## Reproduction

1. Use Python with dependencies from `requirements.txt`.
2. Run the validation and figure scripts in `scripts/` from the repository root.
3. Compile the manuscript from `paper/paper1/` with:

```text
pdflatex systemic_risk_brazill_p1.tex
bibtex systemic_risk_brazill_p1
pdflatex systemic_risk_brazill_p1.tex
pdflatex systemic_risk_brazill_p1.tex
```

## Data source

Raw adjusted B3/COTAHIST data are publicly distributed by B3:
https://bvmf.bmfbovespa.com.br/InstDados/SerHist/

## Citation

Cardoso, G. N., & Sato, R. C. (2026). *Taxonomy of Critical Phenomena in Financial Markets*. Reproducibility repository. GitHub: `taxonomy-critical-phenomena-financial-markets`.
