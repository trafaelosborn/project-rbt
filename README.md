# Project RBT

**A non-neural, target-conditioned corpus transformation system for typological projection.**

This repository accompanies the paper *Target-Conditioned Corpus Transformation: A Non-Neural Typological Projection* (Osborn, 2026). It contains the engine code, post-hoc analysis scripts, validator and control outputs, and the manuscript source for the v5 paper-run reported in that work.

## What this is

Project RBT projects modern-language corpora onto a low-dimensional structural fingerprint space and runs a target-conditioned stochastic search that iteratively transforms a source corpus under explicit pressure from a target reference. The paper run reported here transforms a Modern French corpus under Latin reference, runs to a native plateau after 614,000 proposals, and is then compared post hoc against a held-out validator bank of six medieval Romance corpora and a control bank including a typologically unrelated language (Sumerian) as a falsifier.

The headline finding is interpretive rather than reconstructive: the structural metric used here functions as a **typological proximity detector** (sensitive to morphological richness, suffix-profile entropy, and inflectional vocabulary breadth) rather than a chronological one. Sumerian, a non-Indo-European agglutinative language with no historical relationship to Romance, lands at fourth-rank structural proximity to Latin under an expanded 15-feature analysis — diagnostic evidence that the metric is not lineage-tracking.

## Paper

The manuscript source is at `docs/manuscript/french_to_latin_retrodiction.tex`. The compiled PDF is built locally with the standard LaTeX toolchain:

```
cd docs/manuscript
pdflatex french_to_latin_retrodiction.tex
bibtex   french_to_latin_retrodiction
pdflatex french_to_latin_retrodiction.tex
pdflatex french_to_latin_retrodiction.tex
```

When the paper is published, the DOI and full citation will appear here.

## Repository contents

```
src/retrodiction/      Engine: candidate generation, mutation, acceptance, scoring path.
src/validation/        Held-out validator-bank and control-bank comparison code.
docs/manuscript/       LaTeX source for the paper (+ references.bib).
docs/reports/          Dimensional robustness check and Latin self-similarity calibration reports.
tmp_dimcheck/          Post-hoc dimensional robustness analysis script (cited in §3.7).
tmp_self_similarity/   Post-hoc Latin self-similarity calibration script (cited in Limitations).
data/processed/        Input corpora (tokenized, structured).
data/retrodiction/french/v5_fortran_c16_seed45_paper_run/   v5 paper-run manifest + per-block lightweight summaries.
data/validation/       Validator-bank, control-bank, and dimensional-robustness output files.
tests/                 Unit tests for the engine and scoring path.
```

Excluded from this repo (intentionally; see `.gitignore`):

- v1–v4 historical development runs (not cited in the paper).
- Non-paper v5 experiments (not cited in the paper).
- Per-block bulk artifacts inside the v5 paper run (corpora dumps, dense matrices, previews, per-block records, live event logs — ~6.7 GB total; lightweight per-block summaries are kept).
- Reference n-gram matrices (~3.4 GB; derivable from `data/processed/` and `src/`).
- Standard build/cache artifacts (`__pycache__/`, `*.pyd`, `.pytest_cache/`, etc.).

## Reproducing the paper run

The engine is CPU-only and fully deterministic given a seed. The paper run consumed approximately **135 hours of wall-clock time** at ~4,534 proposals/hour.

The configuration that produced the reported numbers is documented exhaustively in §2.3 of the manuscript and is also frozen in the run manifest at `data/retrodiction/french/v5_fortran_c16_seed45_paper_run/manifest.json`. A replay audit (`docs/reports/2026-04-24_v5_paper_run_seed_audit.md`) records the seed-discrepancy reconciliation: the manifest's launch field records requested seed 45, while the engine seed used by the search was 42.

## Post-hoc analyses

Both post-hoc analyses cited in the paper are reproducible from the data here:

- **Dimensional robustness check** — `tmp_dimcheck/dim_robustness.py`. Expands the 4D structural feature space to 15D and re-ranks corpora under the expanded metric. Report: `docs/reports/2026-05-19_dimensional_robustness_check.md`.
- **Latin self-similarity calibration** — `tmp_self_similarity/calibration.py`. Establishes the matched-slice ceiling for Latin under the structural and family-alignment scores. Report: `docs/reports/2026-05-19_latin_self_similarity_calibration.md`.

## Data availability

The full validator-bank, control-bank, dimensional-robustness, and self-similarity output files are included in this repository under `data/validation/`. The v5 paper-run lightweight summaries are under `data/retrodiction/french/v5_fortran_c16_seed45_paper_run/`. Heavy intermediate artifacts (per-block corpora, dense matrices, previews) are excluded from this repository for size reasons; the lightweight summaries are sufficient for the comparisons reported in the paper.

A frozen snapshot of this repository at the moment of paper submission is archived at Zenodo: **DOI: [pending]**.

## Citation

If you use code or data from this repository, please cite the paper:

```
Osborn, T. (2026). Target-Conditioned Corpus Transformation: A Non-Neural
Typological Projection. [Journal/DOI pending.]
```

## License

The source code in this repository is released under the MIT License (see `LICENSE`). The manuscript and analysis reports are released under CC-BY 4.0.

## Use of AI tools

The author designed the methodology, experiments, and overall research program. Claude Code (Anthropic) drafted most of the implementation, including substantial portions of the engine and both post-hoc analysis scripts. Manuscript writing, restructuring, and editing were assisted by Claude (Anthropic) in the Octave document-chat workstation. The author retains full responsibility for all design choices, implementation correctness, analysis, and conclusions. See the *Use of AI Tools* section of the manuscript for the full disclosure.

## About the name

"RBT" is left unexpanded in the paper — it's just the project name. Unofficially, though, it stands for **Romantic Baby Talk**: the engine babbles its way semi-randomly from one corpus state to another under the pressure of a target reference, and what comes out at the plateau is, structurally speaking, the babble that most resembles the target.

The honest description is the unofficial one. The engine is a stochastic-acceptance search over a feature-space neighborhood; it does not learn, infer, or reconstruct anything. The interesting question — addressed in the paper — is what intermediate states a search like that produces when the target is a typologically distant ancestor language, and what the metric's behavior reveals about the geometry of the feature space rather than the history of the languages.
