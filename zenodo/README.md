# Zenodo deposit package: Paper 1

This package contains the data and reproducibility materials supporting *Taxonomy of Critical Phenomena in Financial Markets*. It is prepared for upload to Zenodo and is not a published DOI record yet.

## Dedicated GitHub repository

The dedicated Paper 1 repository should use the article title as its GitHub slug:

`https://github.com/gersonncardoso/taxonomy-critical-phenomena-financial-markets`

This repository is intended to contain only the Paper 1 pipeline, code,
configuration, documentation, and reproducibility instructions. It must not
include the Paper 2 GDELT, sentiment, or MRQAP materials.

## Authors

- Gerson Nassor Cardoso, ORCID: https://orcid.org/0000-0002-8499-3390
- Renato Cesar Sato, ORCID: https://orcid.org/0000-0002-9902-9086

## Scope

- 360 monthly rolling windows of B3 equity networks, 1995-2025.
- Four configured core events: 1997, 1998, 2008, and 2020.
- One additional 1999 -3 sigma trigger and six fallback sensitivity proxies.
- PMFG network edge lists, filtered edge lists, consolidated metrics, centralities, correlation data, validation outputs, heatmaps, dendrograms, rolling plots, and active Paper 1 artwork.
- No GDELT, news-sentiment, MRQAP, or Paper 2 artifacts are included.

## Directory layout

- `data/network_edge_lists/`: 360 PMFG edge lists and 360 filtered edge lists.
- `data/metrics/`: consolidated metrics, centralities, group-size nulls, and correlation matrices.
- `data/validation/`: validation and sensitivity CSV outputs used by the manuscript.
- `figures/`: heatmaps, dendrograms, network plots, validation figures, and active multipart atlas artwork.
- `code/`: selected scripts and configuration needed to reproduce the included outputs.
- `metadata.json`: Zenodo upload metadata.
- `MANIFEST.sha256`: checksums for every package file.

## Citation after publication

Cardoso, G. N., & Sato, R. C. (2026). B3 rolling-window systemic-risk data and reproducibility materials for *Taxonomy of Critical Phenomena in Financial Markets* [dataset]. Zenodo. DOI: to be assigned after publication.

Do not replace the placeholder with a DOI until Zenodo has published the record and the DOI resolves.
