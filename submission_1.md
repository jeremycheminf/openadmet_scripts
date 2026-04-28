# PXR pEC50 Prediction — Submission 1
Using Claude for script building

**Date:** 2026-04-28  
**File:** `results/submission_pEC50.csv` (513 rows, no NaN)  
**Prediction range:** 2.898 – 6.018 | mean 4.955 ± 0.552

---

## Pipeline

### Data
- **Train:** 4,139 raw → 3,764 curated (after SMILES standardisation, salt removal, quality & AD filters)
  - Quality filter: removed 357 compounds with `pEC50_std.error > 0.5`
  - AD filter: removed 17 compounds with max Tanimoto to test < 0.15
- **ChEMBL enrichment (CHEMBL3401):** 319 novel PXR agonist EC50 records merged → **4,083 final training compounds**
- **Test:** 513 blinded molecules

### Feature sets
| Name | Dimensions | Description |
|------|-----------|-------------|
| `ecfp4` | 2048 | Morgan r=2, binary |
| `rdkit2d` | 217 | RDKit 2D descriptors, RobustScaler |
| `mordred2d` | 1443 | Mordred 2D, RobustScaler |
| `rdkit3d_pharm` | 901 | WHIM+GETAWAY+RDF+MORSE+AUTOCORR3D (ETKDG+MMFF conformers) |
| `mordred3d` | 213 | Mordred 3D-only descriptors |
| `ecfp4_rdkit` | 2265 | ecfp4 + rdkit2d |
| `ecfp4_mordred` | 3491 | ecfp4 + mordred2d |
| `ecfp4_rdkit_3dqsar` | 3166 | ecfp4 + rdkit2d + rdkit3d_pharm |
| `ecfp4_mordred3d` | 3704 | ecfp4 + mordred2d + mordred3d |
| `all_combined` | 3329 | ecfp4 + rdkit2d + Avalon/AtomPair/RDKpath |

3D conformers cached to SDF (ETKDG+MMFF, 4082/4083 train, 513/513 test).

### Cross-validation
3 × 5-fold **Butina clustering** CV (scaffold-diverse splits, Tanimoto cutoff 0.4).  
Splits pre-computed once and reused across all models.

---

## CV Results (20 models, 15 folds each)

| Model | RMSE mean | RMSE std | Spearman |
|-------|-----------|----------|----------|
| **LGB_all** | **0.5954** | 0.0162 | 0.716 |
| XGB_all | 0.5970 | 0.0177 | 0.715 |
| **ChemProp_Chemeleon** | **0.5977** | 0.0176 | **0.725** |
| XGB_ecfp4_rdkit | 0.5990 | 0.0176 | 0.713 |
| CAT_all | 0.5993 | 0.0194 | 0.712 |
| XGB_ecfp4_mordred | 0.6003 | 0.0178 | 0.713 |
| LGB_ecfp4_mordred3d | 0.6014 | 0.0178 | 0.709 |
| CAT_ecfp4_mordred | 0.6020 | 0.0193 | 0.715 |
| CAT_ecfp4_mordred3d | 0.6027 | 0.0188 | 0.713 |
| LGB_ecfp4_rdkit | 0.6027 | 0.0188 | 0.710 |
| LGB_ecfp4_mordred | 0.6031 | 0.0201 | 0.710 |
| CAT_ecfp4_rdkit | 0.6034 | 0.0185 | 0.708 |
| XGB_ecfp4_mordred3d | 0.6049 | 0.0172 | 0.708 |
| LGB_3dqsar | 0.6101 | 0.0174 | 0.704 |
| XGB_3dqsar | 0.6102 | 0.0167 | 0.706 |
| CAT_3dqsar | 0.6107 | 0.0176 | 0.706 |
| RF_ecfp4_rdkit | 0.6309 | 0.0197 | 0.689 |
| EN_ecfp4 | 0.6815 | 0.0150 | 0.656 |
| LGB_extra_fps | 0.6885 | 0.0166 | 0.633 |
| XGB_extra_fps | 0.6954 | 0.0185 | 0.631 |

Note: TabPFN not included (failed during CV run).

### Statistical comparison
Pairwise Wilcoxon signed-rank tests with BH FDR correction (190 pairs, 15 observations each).  
- Top 6–8 models (LGB_all → XGB_ecfp4_mordred) are **statistically indistinguishable** from each other (p_adj > 0.05)  
- 3D QSAR (rdkit3d_pharm) models significantly worse than the 2D top cluster (p_adj < 0.01)  
- Extra fps (Avalon+AP+RDK), EN, RF all significantly worse (p_adj < 0.001)

---

## Ensemble Analysis

Evaluated on OOF predictions (4,083 compounds, repeat-0 folds):

| Strategy | OOF RMSE | vs best single |
|----------|----------|----------------|
| **NNLS** (10 models) | **0.56696** | **−4.8%** |
| Greedy Caruana (4 models) | 0.56747 | −4.7% |
| ElasticNetCV | 0.56793 | −4.6% |
| Mean top-3 | 0.57110 | −4.1% |
| Mean top-5 | 0.57492 | −3.4% |
| Softmax-weighted | 0.58313 | −2.0% |
| Mean all | 0.58456 | −1.8% |
| *Best single (LGB_all)* | *0.59536* | — |

### Final ensemble weights (NNLS)

| Weight | Model |
|--------|-------|
| 41.6% | ChemProp_Chemeleon |
| 15.8% | XGB_ecfp4_mordred |
| 15.0% | XGB_all |
| 11.3% | XGB_ecfp4_rdkit |
|  5.4% | LGB_all |
|  3.8% | EN_ecfp4 |
|  2.4% | LGB_3dqsar |
|  1.9% | CAT_ecfp4_mordred3d |
|  1.7% | XGB_3dqsar |
|  1.0% | LGB_ecfp4_rdkit |

ChemProp_Chemeleon dominates because its graph message-passing captures structural features complementary to ECFP4-based fingerprints. All three smart weighting methods (NNLS, greedy, ElasticNet) independently converge on ~40% weight for ChemProp.

---

## Key Observations

- **3D QSAR adds marginal diversity** in the ensemble (small weights) but hurts as standalone — conformer-based 3D descriptors not well-suited to activation data with flexible active site
- **Extra fingerprints (Avalon/AtomPair/RDKit-path)** consistently underperform ECFP4 — no added value here
- **Mordred 2D** gives small improvement over RDKit 2D alone (~0.001 RMSE), not worth the compute unless using the ensemble
- **ChemProp+Chemeleon** is the key performance driver in the ensemble — strongly recommended for future iterations

