# OpenADMET CYP Inhibition Challenge — submission reproduction kit

Minimal, runnable code to reproduce our submission to the [OpenADMET CYP Inhibition
Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge): direct
CYP1A2/2C9/2D6/3A4 inhibition (pIC50 regression) and CYP2D6/3A4 time-dependent
inhibition (TDI classification).

This is a trimmed-down version of a larger internal research repo — see
[What we tried but didn't include](#what-we-tried-but-didnt-include) for
everything else we explored. The goal here is a small number of scripts that get
you from a clean checkout to a validated submission file, not the full research trail.

## Results

| Track | Metric | This kit (OOF CV) | Full research ensemble | Notes |
|---|---|---|---|---|
| Direct inhibition (regression) | Macro Soft-Threshold RAE ↓ | **0.614** | 0.604 | mean-predictor floor = 1.0 |
| TDI (classification) | Macro MCC ↑ | **0.283** | 0.312 | chance floor = 0.0 |

This kit's 5+2 models get within ~2% of the full research ensemble's ~15+7 models on
regression, and a real but larger gap on TDI (the full ensemble's TDI pool has more
genuinely diverse candidates — see the research repo if closing that gap matters to
you). Exact numbers depend on Butina-fold randomness and TabPFN/chemprop training
stochasticity — expect small run-to-run variation.

## Approach

Two things, repeated across every model family: **freeze, don't fine-tune** pretrained
encoders on this challenge's own labels (fine-tuning consistently *hurt* relative to
using the frozen encoder as a feature extractor — a population-mismatch effect, see
below), and **combine everything with Caruana bagged ensemble selection**, not a plain
mean (it's structurally robust to correlated candidates and doesn't destructively
reallocate weight the way a continuous optimizer like NNLS can on true blind data).

**Regression models** (5, then Caruana-combined):
1. LightGBM on ECFP4 + RDKit2D descriptors — classical baseline.
2. TabPFN on frozen `chemprop_medium` (D-MPNN pretrained on public CYP/ADME data)
   embeddings — best single model in our runs.
3. TabPFN on frozen `chemprop_chemeleon` embeddings — a second checkpoint from the
   same pretraining project, CheMeleon-foundation-initialized; a genuinely
   decorrelated second cluster from (2), not a duplicate.
4. Multitask ChemProp fine-tune, warm-started from `chemprop_medium`: 4 primary
   heads (this challenge's own pIC50) + 4 single-concentration log2fc heads (real
   auxiliary targets, same compounds) + 4 ChEMBL pIC50 heads + 5 PubChem qHTS
   pIC50-like heads (see below) = 17 heads total. Its *raw* calibrated score isn't
   the strongest, but it's consistently the best-ranking chemprop-family model and
   the single largest ensemble contributor we found.

**TDI models** (2, then Caruana-combined): LightGBM baseline, and TabICL on the
same frozen `chemprop_medium` embeddings (TabICL specifically beat TabPFN on this
track).

**Auxiliary data — PubChem AID 1851.** `adme_pretrain`'s own ChEMBL-sourced CYP
data is medicinal-chemistry-biased — it runs ~1–1.65 log units more potent on
average than this challenge's own diversity-library screening population, which is
exactly what makes *fine-tuning* on it hurt (the encoder gets pulled toward a
population it won't see at test time). PubChem AID 1851 is an NCATS qHTS
cytochrome panel (CYP1A2/2C9/2C19/2D6/3A4, ~16.5k substances from NCATS's
diversity-oriented "Biodiverse" collection) — much closer in population shape to
this challenge's own screen. Fetched via PubChem's bulk per-SID CSV export
(batched under its 10k-SID cap; the simpler "concise" export only has binary
outcomes, not the continuous `Fit_LogAC50` dose-response fit this uses).

## Quickstart

```bash
pip install -e .
python scripts/01_download_data.py                 # HF train/test + PubChem AID 1851
python scripts/02_baseline_lgb.py                   # LightGBM, both tracks
python scripts/03_embed_frozen_encoders.py          # frozen chemprop_medium + chemeleon embeddings
python scripts/04_run_tabpfn.py                     # TabPFN on those embeddings (regression)
python scripts/05_run_tabicl_tdi.py                 # TabICL on adme_pretrain embeddings (TDI)
python scripts/06_finetune_chemprop_multitask.py    # 17-head multitask fine-tune
python scripts/07_build_submission.py               # Caruana ensemble -> results/submission_{activity,tdi}.csv
```

`tabpfn` and `tabicl` commonly need separate environments (they conflict on shared
dependencies in practice) — scripts 04 and 05 are split for exactly that reason; run
each with whichever environment has the matching package installed.

Needs a CUDA GPU for reasonable runtime on scripts 03–06 (chemprop/TabPFN/TabICL all
run on CPU too, just much slower). Total runtime roughly 2–3 hours on a single GPU,
dominated by script 06's fine-tune (~1.5h).

`checkpoints/chemprop_medium.pt` and `checkpoints/chemprop_chemeleon.pt` are
included directly (D-MPNN checkpoints from a sibling pretraining project, trained on
public ChEMBL/TDC/Polaris data plus a public Novartis-surrogate CYP panel described
in an OpenADMET community talk — see [References](#references)).
`data/external/cyp_chembl.csv` (public ChEMBL bioactivity data, pre-extracted) is
also included; PubChem data is fetched fresh by script 01.

## What we tried but didn't include

From a larger internal exploration, kept out of this kit either because they didn't
help or because they add complexity/dependencies disproportionate to their benefit:

- **Monroe** (a GRIT graph transformer pretrained on quantum-chemistry + PubChem
  bioassay data, frozen + TabPFN) — a real, decorrelated contributor in our full
  ensemble, but its own package + checkpoint live in a separate project with a
  torch-geometric dependency; left out here for a lighter dependency footprint.
- **AutoGluon** and **plain LightGBM on the frozen embeddings** — both weaker than
  TabPFN on the identical features, and net-neutral-to-negative once added to the
  ensemble.
- **3D shape/polarity descriptors** (Jazzy polarity + RDKit USR/USRCAT/PMI) — a
  genuinely different (non-graph) modality, but the weakest standalone model we
  tried and net-negative for the ensemble.
- **"Vanilla" CheMeleon** (the public foundation checkpoint with zero CYP/ADME
  adaptation) — turned out redundant with classical descriptors, not the
  more-decorrelated source we expected.
- **Domain-adapting `chemprop_medium` on PubChem qHTS data alone** (no ChEMBL, then
  freeze + embed) — a real, validated win in our full ensemble (best-ever
  single-model CYP2D6 score), left out of this kit purely for script count; see the
  full research repo if you want the extra edge on CYP2D6 specifically.
- **Self-supervised contrastive pretraining** (SimCLR-style domain-adaptation on
  ~34k unlabeled molecules, no labels at all) — converged too easily to be useful;
  the worst embedding source we tried.
- **ChemProp multitask v1/v2** (flat and inverse-count task weighting without the
  PubChem heads) — superseded by the 17-head version included here.
- **Pseudo-labeling missing pIC50 rows from the single-concentration screen** — hurt
  badly (assay saturation at high concentration); using the same data as a genuine
  auxiliary multitask target (as done here, `log2fc` heads) is fine, treating it as
  ground truth for missing rows is not.
- **TabICL / TabPFN on classical ECFP4+RDKit2D descriptors** — highly correlated
  with the LightGBM baseline already included (r ≈ 0.9), redundant.

## References

- Challenge home: <https://huggingface.co/spaces/openadmet/cyp-challenge>
- Data: <https://huggingface.co/datasets/openadmet/cyp-challenge-train-test>
- Official evaluation/validation code (vendored here under `reference/`, unmodified):
  <https://github.com/OpenADMET/CYP-Challenge-Tutorial>
- Background: <https://openadmet.github.io/octant-cyp-inhib-blog-post/>,
  <https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/>
- PubChem AID 1851 (NCATS qHTS cytochrome panel):
  <https://pubchem.ncbi.nlm.nih.gov/bioassay/1851>
- ChEMBL database (`data/external/cyp_chembl.csv`'s source — a local ChEMBL 37
  SQLite dump, queried directly, not a web API client): Zdrazil et al.,
  *"The ChEMBL Database in 2023,"* Nucleic Acids Research, 2024,
  doi:10.1093/nar/gkad1004, EMBL-EBI. Molecule standardization via RDKit's
  `MolStandardize` module (<https://www.rdkit.org>).
- Monroe (not included in this kit, but informed our "freeze the encoder" approach):
  Banaszewski & Fitzgibbon, arXiv:2608.18982
- CheMeleon foundation checkpoint: Zenodo record 15460715, arXiv:2506.15792
- Caruana ensemble selection: Caruana, Niculescu-Mizil, Crew & Ksikes,
  *"Ensemble Selection from Libraries of Models,"* ICML 2004
- The current leaderboard team SuperCowPowers/briford's public write-up and repo
  (their "placement correction" calibration idea, and the multi-task-with-auxiliary-
  heads recipe this kit's approach is closest to):
  <https://supercowpowers.github.io/workbench/blogs/cyp_challenge/>,
  <https://github.com/SuperCowPowers/workbench>
- A different OpenADMET-PXR-challenge team's public write-up (RyeCatcher/BioInfo) —
  informed the auxiliary-head strategy and the "analog-expansion test sets contain
  structurally-close-but-inactive compounds" caution:
  <https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026>
- OpenADMET PXR challenge `LESSONS_LEARNED.md` (a prior internal write-up from the
  same team on the previous OpenADMET challenge) — Butina scaffold CV, Caruana over
  NNLS/plain-mean, and several "tried, didn't help" leads this kit avoided repeating.
