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
- `pipeline/`: the six executable Paper 1 stages plus `pipeline_paper1.py`,
	which runs data preparation, rolling correlations, validation, PMFG networks,
	metrics, and figures.
- `main/`: stage-specific entry points called by the Paper 1 runner.
- `scripts/`: scripts used to generate the active sensitivity outputs and figures.
- `src/`: network, statistical, rolling-window, and visualization implementations
	used by the Paper 1 pipeline.
- `configs/`: configuration required for the B3 event calendar and pipeline.
- `requirements.txt`: dependencies for the Paper 1 pipeline only.
- `data/`: small, publication-relevant validation tables only.
- `zenodo/`: instructions and metadata for the complete large-data archive.

The active artwork is duplicated under `paper/paper1/figures/` because the
LaTeX source resolves paths relative to `paper/paper1/`, while the Python
plotting scripts resolve paths relative to the repository root. Both copies
come from the same active artwork set and must remain synchronized.

## Large data archive

The complete matrices, 360 PMFG edge lists, filtered edge lists, and large rolling outputs are not stored in this GitHub repository because they exceed practical GitHub repository limits. They are packaged for Zenodo in `deliverables/zenodo_paper1.zip` and should be published there to obtain a persistent DOI.

After Zenodo publication, add the DOI here and in the manuscript Data Availability statement.

## Reproduction

1. Use Python with dependencies from `requirements.txt`.
2. From the repository root, run the six-stage Paper 1 pipeline:

```text
python -m pipeline.pipeline_paper1
```

Use `--force` to recompute existing intermediate outputs. The runner executes
only the B3 financial-network stages 1--6: data preparation, rolling
correlations, heatmaps/dendrograms, KS/MP/ADF validation, PMFG construction,
network metrics, and final figures. It intentionally excludes GDELT, news
sentiment, MRQAP, Spark sentiment stages, and all Paper 2 code.

3. Run the sensitivity and figure scripts in `scripts/` from the repository root
	after the six-stage pipeline has produced the required rolling outputs.
4. Compile the manuscript from `paper/paper1/` with:

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
