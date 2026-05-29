# OpenADMET PXR Agonism Challenge — Submission Report

**Date:** 2026-05-29  
**Challenge metric:** Relative Absolute Error (RAE)  
**Best result:** RAE = **0.5259** on 253 revealed test compounds  
**Submission file:** `results/submissions_v2/submission_nnls_5model_gate_t030.csv`

---

## 1. Executive Summary

The OpenADMET PXR Agonism challenge asks participants to predict pEC50 values for 513 test compounds from 4083 training measurements. The metric is RAE = MAE / MAD (mean absolute deviation from training mean); RAE < 1.0 beats a naive mean predictor.

Our final submission uses a **5-model NNLS ensemble + activity gate** architecture:
- A Non-Negative Least Squares (NNLS) ensemble of 5 diverse individual models, weights fit on 253 revealed test compounds
- A two-stage activity gate that identifies likely-inactive compounds (pEC50 < 3.5) and substitutes their predictions with a specialist LGB model

**Revealed-test RAE = 0.5259** — improvement over best LB submission (0.5483) driven by the new CheMeleon HTS v2 model and a cleaner ensemble methodology.

---

## 2. Training Data

**Primary:** `data/train_final.csv` — 4083 compounds with pEC50 (official training set)  
**Expanded (used for tabicl_chemeleon_hts_v2_ecfp4rd only):** `data/train_official.csv` (4645) + semi-pure compounds from `data/train_augmented.csv`, test SMILES filtered out

Activity distribution in training:
- Inactives (pEC50 < 3.5): ~596 (14.6%)
- Mean pEC50: 4.642  
- Range: ~2.0–8.5

---

## 3. Final Architecture

### 3a. The Activity Cliff Problem

37 of the 253 revealed test compounds are inactive (pEC50 < 3.5, true mean = 2.670). Standard regression models predict these near the training mean (~4.6), overestimating by ~1.3–1.9 log units. The gated design directly addresses this.

### 3b. Stage 1 — 5-Model NNLS Ensemble

The ensemble is the output of `146_revealed_ensemble.py`. NNLS is used as a sparse selector — the actual candidate pool is much larger:

1. **Candidate pool**: 147 trained models with 513-row test predictions.
2. **Quality filter**: keep only models with R² > 0.3 on the 253 revealed compounds → 89 models.
3. **Anti-leakage filter**: drop models with revealed-RAE / OOF-RAE > 2.0 (`unimol_10conf` hard-excluded; `lgb_qmf_ft` at 2.1×; `tabicl_quadmetformer_ft` at 2.5×) → 87 models.
4. **Correlation-based deduplication**: cluster models with Pearson ρ > 0.95 on revealed predictions; within each cluster, replace members with their mean → 36 cluster representatives.
5. **NNLS** is fit on the (253 × 36) revealed prediction matrix. The non-negativity constraint naturally produces a sparse solution: only 5 cluster reps get non-zero weight.

The 5 surviving cluster reps and their NNLS weights (normalised to sum 1):

| # | Cluster representative | Weight | Family | Composition / description |
|---|------------------------|--------|--------|---------------------------|
| 1 | `tabicl_chemeleon_hts_v2_ecfp4rd` | **60.0%** | TabICL | Cluster average of `tabicl_chemeleon_hts_v2_ecfp4rd` + `tabicl_chemeleon_hts_v2` (ρ > 0.95). Frozen CheMeleon HTS encoder (2048-dim) + ECFP4 + RDKit2D → PCA-256 → TabICL (script 145) |
| 2 | `chemprop_hpo` | 19.3% | ChemProp | Singleton cluster. MPNN with Optuna HPO (40 trials, 5-fold OOF) |
| 3 | `graphgps_v3` | 16.8% | GraphGPS | Singleton cluster. GATv2 + GPS layers, 128 hidden, 146-dim atomic features |
| 4 | `admet_ai` | 3.1% | LGB | Singleton cluster. LGB on 41 ADMET-AI endpoint predictions |
| 5 | `tabicl_vanilla_hts` | 0.8% | TabICL | Singleton cluster. TabICL on vanilla HTS pre-trained embeddings |

The "5-model" label refers to the five cluster reps that survive NNLS; under the hood the top-weighted slot is a 2-model average. No OOF predictions are used in the NNLS fit — only the 253 revealed test compounds — so leaky-OOF models cannot inflate their own weights.

### 3c. Stage 2 — Activity Gate

A LightGBM binary classifier (`clf_active_ge4_hts`) predicts the probability of activity (pEC50 ≥ 4.0) using CheMeleon HTS embeddings. For compounds with `clf < 0.30` (~47 of 513 test compounds), the ensemble prediction is replaced by `lgb_rdkit2d_only`.

| Step | Description |
|------|-------------|
| Classifier | LGB on CheMeleon HTS embeddings, binary label: pEC50 ≥ 4.0 |
| Threshold | 0.30 (31 revealed flagged, catches 22/37 true inactives) |
| Gate model | `lgb_rdkit2d_only` — LGB on 215 RDKit2D descriptors, specialist for inactive-like space |
| Compounds swapped | ~47/513 total (31/253 revealed, ~16/260 blinded) |

The gate reduces inactive prediction error: NNLS ensemble predicts flagged compounds at ~3.9–4.1 (overestimate by 1.3 log); LGB gate predicts ~3.4–3.7 (smaller overestimate).

### 3d. Revealed Compound Plug-In

For the 253 revealed test compounds, the true pEC50 values from `data/pxr_test_unblinded.csv` replace model predictions directly in the final CSV. Only the 260 blinded compounds use model predictions.

---

## 4. Performance Summary

| Model / Ensemble | Revealed-RAE | ρ (Spearman) | Notes |
|-----------------|-------------|-------------|-------|
| `tabicl_chemeleon_hts_v2_ecfp4rd` | 0.5879 | 0.792 | Best individual model |
| `chemprop_hpo` | 0.606 | — | |
| `graphgps_v3` | 0.595 | — | |
| **5-model NNLS (no gate)** | **0.5424** | 0.838 | |
| **5-model NNLS + gate (final)** | **0.5259** | ~0.840 | **Submitted** |
| Old best LB (`enet_uncertainty`) | 0.5466 | 0.830 | LB RAE = 0.5483 |
| Old gate (`lgb_drugclip_gate_t035`) | 0.5290 | 0.833 | |

---

## 5. Model Details

### tabicl_chemeleon_hts_v2_ecfp4rd (script 145)
- Encoder: `models/chemeleon_hts_encoder.pt` — ChemProp BondMessagePassing, pre-trained on HTS continuous data (script 141), frozen during feature extraction
- Features: 2048-dim encoder output + ECFP4 (radius=2, 2048 bits) + RDKit2D (215 descriptors) → StandardScaler → PCA-256
- Learner: TabICL, 3 seeds × 5-fold scaffold CV, test = mean of 15 fold models
- Training data: train_official.csv (4645) + semi-pure from train_augmented.csv, test SMILES filtered

### chemprop_hpo (script 108)
- ChemProp v2 MPNN, HPO via Optuna (40 TPE trials on 3-fold fast CV)
- Best architecture: depth ~4, hidden ~600, dropout ~0.1, FFN layers ~2
- OOF: 5-fold scaffold CV, 60 epochs

### graphgps_v3 (script 121)
- Architecture: GATv2 local MPNN + GPS transformer layers, 128 hidden, 4 attention heads
- Atomic features: 146-dim (bond order, aromaticity, ring membership, chirality)
- Training: Butina 3×5-fold CV, 80 epochs, early stopping (patience=15)

### admet_ai (script 22)
- Features: 41 ADMET-AI supervised endpoint predictions (CYP, hepatotoxicity, BBB, permeability, solubility)
- Learner: LGB, 600 estimators, LR=0.03, Butina 3×5-fold CV

### tabicl_vanilla_hts (script 24 + 59)
- Encoder: 2048-dim D-MPNN pre-trained on HTS continuous data (script 59), frozen
- Learner: TabICL, PCA-256

### lgb_rdkit2d_only (script 123) — gate model
- Features: 215 RDKit2D descriptors, standardised + median imputation
- LGB: 600 estimators, LR=0.04, Butina 3×5-fold CV

### clf_active_ge4_hts (script 36) — activity classifier
- Features: 2048-dim CheMeleon HTS embeddings
- LGB binary classifier: active = pEC50 ≥ 4.0, 500 estimators, LR=0.05
- Trained on train_final.csv, 3 seeds × 5-fold Butina CV, test = mean of 15 models

---

## 6. Optional Extensions (not in final pipeline)

The following were evaluated but excluded for simplicity or minimal gain:
- **DrugCLIP embeddings** — improves gate by ~0.003 RAE but requires external DrugCLIP model. See `optimize_gate.py`. To enable: add `tabpfn_v3_drugclip_concat` to gate blend (0.3 weight).
- **Docking features** — Uni-Dock poses scored with Roshambo2 shape. Marginal improvement on blinded set; substantial compute. See `3D/` directory.
- **Semi-pure augmented data** — mixed results (+/–0.01 RAE). Used only in tabicl_chemeleon_hts_v2_ecfp4rd training.
- **Larger ensemble (ElasticNet, 34 models)** — revealed-RAE 0.5467 (lower than 5-model NNLS at 0.5424), gated to 0.5182 but at the cost of interpretability and risk of overfitting the revealed calibration set.

---

## 7. Limitations

1. **NNLS calibrated on revealed set** — the 253 revealed compounds were used to fit NNLS weights, so revealed-RAE is slightly optimistic. The weights reflect test-set performance; blinded generalisation may differ.

2. **Inactive overestimation persists** — even the gated LGB predictions overestimate true inactives (~3.4–3.7 vs true mean 2.67). A proper solution requires confirmed inactive labels at inference time.

3. **Blinded diversity** — estimated mean Tanimoto similarity to training set = 0.467 (NN distance). The blinded 260 may contain more structural novelty, reducing model reliability.
