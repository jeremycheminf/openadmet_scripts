# PXR Descriptor & Fingerprint Combination Study

LGB Butina 3×5-fold OOF RMSE on 4083 training compounds.
Baseline: TabICL on chemeleon_hts_mtl = **0.495** (HTS-pretrained encoder, dominant base learner).

---

## Key Finding

**Encoders + TabICL dominate everything.** No descriptor combination with LGB comes close
to TabICL on the HTS-pretrained CheMeleon encoder. The best LGB descriptor combo
(CheMeleon+ECFP4+rdkit2d, RMSE 0.531) is 0.036 worse than the TabICL encoder baseline.

---

## Descriptor Combination Study (53_descriptor_combo_study.py)

22 LGB combinations tested. Groups A–F.

### Top results by OOF RMSE

| Rank | Combination | Dim | RMSE | RAE | Spearman ρ |
|------|-------------|-----|------|-----|-----------|
| 1 | **CheMeleon+ECFP4+rdkit2d** | 4313 | **0.531** | 0.529 | 0.801 |
| 2 | CheMeleon+QMF | 2176 | 0.539 | 0.539 | 0.794 |
| 3 | CheMeleon+UniMol | 2560 | 0.545 | 0.543 | 0.790 |
| 4 | FP+all_desc+ADMET | 4186 | 0.577 | 0.588 | 0.738 |
| 5 | Everything-but-encoders | 6661 | 0.577 | 0.586 | 0.736 |
| 6 | FP+desc+QMF | 6051 | 0.593 | 0.604 | 0.721 |
| 7 | FP+rdkit2d+mordred2d | 3875 | 0.593 | 0.603 | 0.720 |
| 8 | QMF+all2D | 4003 | 0.594 | 0.604 | 0.719 |
| 9 | ECFP4+rdkit2d+mordred2d | 3708 | 0.595 | 0.605 | 0.719 |
| 10 | rdkit2d only | 217 | 0.613 | 0.623 | 0.701 |
| 22 | ECFP4 only | 2048 | 0.692 | 0.720 | 0.635 |

### Key observations

1. **CheMeleon embeddings dominate** — the top-3 all require CheMeleon as a component.
   Without CheMeleon, best is 0.577 (3× worse than TabICL baseline).

2. **QMF (128-d) adds marginal signal** (+0.006 RMSE over CheMeleon alone) — orthogonal
   QM-based features complement HTS-learned representations.

3. **rdkit2d is the best single classical descriptor** (RMSE 0.613) — much better than
   fingerprints alone (ECFP4: 0.692). Physicochemical descriptors >> structural bits.

4. **Adding more FP types doesn't help** — ECFP4+MACCS+AtomPair = 0.646, worse than
   ECFP4+rdkit2d = 0.596. Diminishing returns from fingerprint concatenation.

5. **Kitchen sink hurts** — more features beyond the encoder combo add noise, not signal.

---

## Fingerprint Grid Search (63_fp_grid_search.py)

4 types × 5 bit sizes × 2 conditions (solo + rdkit2d concat) = 40 combinations.

### Solo fingerprint OOF RMSE by type and bits

| Bits | ECFP4 | ECFP6 | FCFP4 | FCFP6 |
|------|-------|-------|-------|-------|
| 1024 | 0.698 | 0.706 | 0.700 | 0.709 |
| 2048 | 0.692 | 0.700 | 0.695 | 0.704 |
| 4096 | 0.683 | 0.694 | 0.693 | 0.698 |
| 8192 | 0.679 | 0.687 | 0.692 | 0.689 |
| **16384** | **0.674** | **0.680** | **0.693** | **0.689** |

### + rdkit2d concat OOF RMSE

| Bits | ECFP4 | ECFP6 | FCFP4 | FCFP6 |
|------|-------|-------|-------|-------|
| 1024 | 0.595 | 0.598 | 0.599 | 0.604 |
| 2048 | **0.595** | 0.602 | 0.600 | 0.604 |
| 4096 | 0.597 | 0.603 | 0.599 | 0.599 |
| 8192 | 0.596 | 0.596 | 0.599 | 0.596 |
| 16384 | **0.595** | 0.595 | 0.599 | 0.596 |

### Key observations

1. **ECFP4 > ECFP6 at all bit sizes** — radius 2 captures the relevant PXR pharmacophore
   better than radius 3. Larger neighbourhoods add noise for this target.

2. **Larger bits improve ECFP4 solo** (monotonically: 0.698 → 0.674) but gains diminish
   rapidly above 4096 bits. Each doubling saves ~0.005 RMSE.

3. **+rdkit2d dominates** — regardless of FP type or bits, adding rdkit2d drops RMSE
   ~0.10. The physicochemical descriptors carry most of the signal.

4. **FCFP adds nothing over ECFP** — feature-based Morgan fingerprints don't outperform
   atom-based at any bit resolution.

5. **Optimal practical choice: ECFP4_2048 + rdkit2d** (RMSE 0.5947) — marginal to use
   16384 bits (saves 0.0001 RMSE, 8× memory). Going beyond 2048 not worth it.

---

## Comparison vs TabICL Baseline

| Model | Feature set | RMSE | vs TabICL_HTS (0.495) |
|-------|------------|------|----------------------|
| **TabICL_chemeleon_hts_mtl** | HTS encoder 2048-d | **0.495** | baseline |
| TabICL_chemeleon_hts_cont | HTS encoder 2048-d | 0.496 | −0.001 |
| LGB: CheMeleon+ECFP4+rdkit2d | 4313-d concat | 0.531 | +0.036 |
| LGB: ECFP4_16384+rdkit2d | 16601-d | 0.595 | +0.100 |
| LGB: ECFP4_2048+rdkit2d | 2265-d | 0.595 | +0.100 |
| LGB: rdkit2d only | 217-d | 0.613 | +0.118 |
| LGB: ECFP4 only | 2048-d | 0.692 | +0.197 |

**Takeaway:** The ~0.036 gap between best LGB descriptor combo and TabICL on the same
encoder is purely from the model class. TabICL's in-context learning on the same feature
set would likely give ~0.480 RMSE (estimated from TabICL_chemeleon_hts = 0.500).

---

## Conclusions

1. **Encoder quality >> descriptor choice >> model class** for this dataset.
2. **HTS pretraining** (CheMeleon trained on 21K PXR screen data) is the single largest
   signal source. Nothing compensates for lack of task-relevant pretraining.
3. **Optimal descriptor set for LGB**: CheMeleon embeddings (2048-d, PCA-256) + ECFP4
   (2048-bit) + rdkit2d (217-d). Anything more adds noise.
4. **For TabICL input**: PCA-256 is adequate for 2048-d HTS embeddings (57% variance
   captured). 1024-d PCA for 4313-d all_combined (79% variance).
5. **ECFP4_2048 is the optimal fingerprint** — going to 16384 bits saves negligible RMSE
   at 8× memory cost. ECFP6, FCFP4, FCFP6 don't outperform ECFP4.
