# PXR Challenge — Training Data Distance from Test: Experiments & Findings

Comprehensive record of all experiments involving training set filtering, applicability domain,
covariate shift characterisation, and correctors based on structural distance to the test set.

---

## 1. The Core Problem: Severe Train–Test Covariate Shift

**Adversarial AUC ≈ 0.95** — a LightGBM binary classifier trained to distinguish training from
test compounds (ECFP4 + RDKit2D features) achieves AUC 0.95. A value this high means the two sets
are nearly linearly separable in chemical space. This is not a random split; the test set was
deliberately designed to probe distinct chemical scaffolds.

Practical consequences:
- Local structural analogues in the training set are unreliable predictors for test compounds.
- Post-processing the predictions to match the training marginal distribution makes things worse.
- Training set variance compression in model output is *correct*, not a bug to fix.
- Test set predicted std ≈ 0.633 vs training std ≈ 0.963 — models compress correctly.

---

## 2. Early AD Filtering Experiments (sub4–sub6, script: `02b_ad_analysis.py`)

Tested four strategies for removing training compounds that are structurally "far" from the test set,
hoping to reduce noise from irrelevant chemistry.

| Strategy | Threshold | Compounds removed | Effect on blind test |
|---|---|---|---|
| Baseline (sub3) | none | 0 | rank 44, RAE 0.614 |
| Rare atom types (sub4) | P/Si/I/B/H absent from test | 30 (0.7%) | rank 46, RAE 0.626 (**worse**) |
| Rare atoms + property 3σ (sub5) | MW/LogP/TPSA/HBD/HBA outliers | 632 (15.5%) | rank TBD |
| Rare atoms + Mahalanobis (sub6) | RDKit2D PCA Mahalanobis > χ²(0.95) | 910 (22.3%) | rank 54, RAE 0.631 (**worse**) |

**Finding**: Removing even 0.7% of training compounds hurt all metrics. Hypothesis: those 30
rare-atom compounds contain substructures useful for test interpolation beyond just their rare
atoms. More aggressive removal (22%) only made things worse.

**Lesson**: AD filtering that removes training diversity is net negative on this dataset. The
overlap is structurally sparse but the model generalises better by retaining all diversity.

---

## 3. ChEMBL Enrichment (sub54, script: `02_chembl_enrichment.py`)

Attempted to bridge the train–test gap by adding public PXR EC50 data from ChEMBL (CHEMBL3401).

| Dataset | Compounds | Blind rank | RAE | R² |
|---|---|---|---|---|
| train_final (standard) | 4,083 | 20 | 0.5546 | 0.548 |
| train_challenge_chembl (enriched) | ~4,400 | 26 | 0.5639 | 0.559 |

**Finding**: ChEMBL enrichment made things 0.009 RAE worse. The ~300 extra ChEMBL compounds
add distributional noise rather than signal, likely because:
1. ChEMBL EC50 mixes heterogeneous assay types (cell line, reporter, endpoint)
2. The challenge assay is a standardised PXR activation protocol; public data diverges
3. Activity context (agonism vs partial agonism) is hard to classify reliably from text comments

**Lesson**: Use `train_final.csv` as-is; external data from ChEMBL does not help.

---

## 4. kNN Residual Corrector (sub44, script: `60_knn_residual_corrector.py`)

The global NNLS ensemble systematically under-predicts high-actives (bias −0.70 for pEC50 > 6)
and over-predicts low-actives (bias +1.44 for pEC50 < 2). Hypothesis: local training analogues
could provide corrective signal.

**Architecture:**
- Combined embedding index: chemeleon_hts_mtl + UniMol + ECFP4 (standardised, concatenated)
- k = 10 nearest training neighbours per test compound
- Features: neighbour pEC50 stats (mean, median, std, weighted), distance profile, classifier signals
- Target: OOF residual = y_true − ensemble_oof_pred
- Corrector: LightGBM with Butina OOF on residuals

| Metric | Sub33 (baseline) | Sub44 (kNN corrector) | Δ |
|---|---|---|---|
| Rank | 15 | 35 | −20 (**much worse**) |
| RAE | 0.550 | 0.580 | +0.030 |
| R² | 0.551 | 0.515 | −0.036 |

**Root cause of failure**: With adversarial AUC 0.95, test compounds are systematically shifted
from the training distribution. The corrector pushes predictions toward local training analogues'
pEC50 values — but those analogues are in a different region of chemical space than the test
compound actually inhabits. The correction amplifies bias rather than reducing it.

Additionally, the OOF correction was itself leaky: the embedding index was built on the full
training set, meaning fold-k training neighbours were included in the kNN reference for OOF
compound k.

---

## 5. GP-Gated kNN Corrector (sub46, script: `64_gp_gated_knn.py`)

Attempted fix for sub44: gate the kNN correction by Tanimoto GP posterior variance.

**Logic:**
- Low GP variance → compound well-covered by training → trust kNN correction
- High GP variance → structurally novel → suppress correction toward zero
- Gate function: gate(σ²) = exp(−σ² / τ), τ tuned on OOF

**Result**: Sub46 not confirmed in the available submit log — likely either not submitted or
merged into another experiment. The fundamental issue (test compounds are novel) means low-variance
compounds with reliable kNN corrections are rare in the test set, limiting the approach's impact.

---

## 6. Training Set Pruning Experiments (sub47–sub49, script: `65_pruning_experiment.py`)

Targeted removal of specific compound types hypothesised to push the model away from test
distribution. Strategy: one hypothesis per submission, leaderboard as sole signal.

| Sub | Pruning | n removed | Rank | RAE | R² | Spearman |
|---|---|---|---|---|---|---|
| Baseline (sub50) | none | 0 | 21 | 0.5546 | 0.546 | 0.834 |
| sub47 | pEC50 < 2.0 (extreme inactives) | 24 | 22 | 0.5546 | 0.548 | 0.834 |
| sub48 | acrylamide warheads (C=CC(=O)N) | ~30 | 20 | 0.5559 | 0.549 | 0.832 |
| sub49 | pEC50 < 2 + acrylamides combined | ~54 | 19 | 0.5559 | 0.549 | 0.832 |

**Finding**: None of the targeted pruning experiments made a significant difference — all landed
within ±2 rank positions and ±0.001 RAE of the unpruned baseline. The pruning hypotheses were
sound (extreme inactives and warhead artefacts are legitimate concerns) but the effect size is
too small relative to rank noise.

**Lesson**: With a structural train–test gap this large, removing small subsets of "problematic"
training compounds doesn't materially change what the model learns or how it generalises.
The gap is global, not localised to a removable fraction of training data.

---

## 7. Post-Processing / Distribution Matching (sub30–sub32, script: `44_postprocess.py`)

Attempted to expand compressed test predictions to match the training pEC50 distribution.

| Sub | Transform | Rank | RAE | R² | Spearman |
|---|---|---|---|---|---|
| sub28 (baseline) | none | 18 | 0.561 | 0.519 | 0.831 |
| sub30 | Quantile transform → match training CDF | 54 | 0.608 | 0.523 | 0.831 |
| sub31 | Linear rescale to match training mean/std | 49 | 0.601 | 0.536 | 0.831 |
| sub32 | Asymmetric expand below 4.5 only | 26 | 0.571 | **0.545** | 0.831 |

Spearman is rank-preserving so it was unchanged by all transforms.

**Finding**: Any distribution expansion hurt RAE, even the targeted one (sub32) that only
expanded the low-activity tail. Blind test compounds are drawn from a narrower, shifted activity
range than training. The model's variance compression is appropriate for the test distribution.

**Lesson**: The test marginal distribution is not the training marginal. Post-processing that
forces predictions toward the training distribution is directionally wrong.

---

## 8. High-Active Deep Dive (script: `47_highactive_analysis.py`)

Investigated the sub-population of training compounds with pEC50 > 6 to understand why the
ensemble under-predicts this region.

Key findings:
- Training pEC50 > 6: ~180 compounds (~4.4% of training set)
- Scaffold overlap with test: minimal — high-active scaffolds are concentrated in training
- Physicochemical profile of high-actives: lower MW (mean ~350), more lipophilic (LogP ~4.5)
- Embedding distance: test compounds in the chemeleon_hts_mtl space are systematically
  far from training high-actives
- Sub34 gate A (blend MolE/UniMol for predicted pEC50 > 5) improved blind R² — the only
  successful mitigation of the high-active bias

---

## 9. Summary: What Works vs What Doesn't

### Confirmed failures (made predictions worse on blind test)

| Approach | Why it fails |
|---|---|
| AD Tanimoto filtering | Removes useful structural diversity |
| Rare-atom filtering | Removes diversity, minimal risk reduction |
| Mahalanobis filtering | Removes too many compounds (22%) with no payoff |
| ChEMBL enrichment | Distribution mismatch between ChEMBL and challenge assay |
| kNN residual corrector (ungated) | Pushes predictions toward wrong local analogues |
| Quantile/linear distribution matching | Test distribution ≠ training distribution |
| Asymmetric tail expansion | Still wrong direction, smaller penalty |

### Marginal / null effects

| Approach | Result |
|---|---|
| Remove 24 extreme inactives (pEC50 < 2) | Within rank noise; no meaningful effect |
| Remove acrylamide warheads | Within rank noise; no meaningful effect |
| GP-gated kNN correction | Insufficient test compounds in low-variance region |

### What helped at the margin

| Approach | Gain | Mechanism |
|---|---|---|
| Gate A: boost MolE/UniMol weight above pEC50=5 | +1 rank | Reduces high-active under-prediction |
| Retain all training data (no pruning) | — | Full diversity retained for embedding learning |

### Core conclusion

The adversarial AUC of 0.95 means no post-hoc adjustment to the training set or predictions
can bridge the structural gap. Performance gains come exclusively from better representations
(HTS-pretrained CheMeleon, MolE 3D pretraining) that generalise across the structural divide,
not from manipulating which training compounds to include or how to rescale outputs.

---

## Scripts reference

| Script | Purpose |
|---|---|
| `02b_ad_analysis.py` | AD filtering strategies: Tanimoto, rare atoms, property space, Mahalanobis |
| `02_chembl_enrichment.py` | ChEMBL CHEMBL3401 enrichment pipeline |
| `60_knn_residual_corrector.py` | kNN residual corrector (sub44) |
| `64_gp_gated_knn.py` | GP-gated kNN corrector (sub46) |
| `65_pruning_experiment.py` | pEC50 < 2 and acrylamide pruning (sub47–49) |
| `44_postprocess.py` | Distribution matching post-processing (sub30–32) |
| `34_residual_analysis.py` | OOF residual binning — exposes calibration bias by pEC50 range |
| `46_lowactivity_analysis.py` | Test compound similarity to low-activity training set |
| `47_highactive_analysis.py` | High-active (pEC50 > 6) scaffold and embedding analysis |
| `61_gp_uncertainty.py` | GP posterior variance → structural novelty estimate |
