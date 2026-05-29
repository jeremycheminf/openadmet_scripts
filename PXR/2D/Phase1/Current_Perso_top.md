# PXR Challenge — Scripts for Best Submission

**Best known blind-test result: rank 20, RAE 0.5546, R² 0.548 (sub47/50/56)**
NNLS inner-CV OOF RMSE ≈ 0.472 | Calibration ratio: blind RAE ≈ OOF RMSE × 1.17

---

## NNLS-selected base learners (nonzero weight)

| Model | NNLS weight | OOF RMSE | Script(s) |
|---|---|---|---|
| TabPFN_v3_chemeleon_hts_cont | 32% | 0.489 | `39_hts_continuous_pretrain.py` → `69_tabpfn_v3_oof.py` |
| TabPFN_v3_chemeleon_hts_mtl  | 31% | 0.491 | `43_hts_mtl_encoder.py` → `69_tabpfn_v3_oof.py` |
| MolE_e50                     | 21% | 0.546 | `21_mole_finetune.py` |
| UniMol_FT_e50                |  9% | 0.574 | `19_unimol_e50.py` + `19b_unimol_e50_aggregate.py` |
| TabICL_chemeleon_hts         |  5% | 0.500 | `23_chemeleon_hts_pretrain.py` → `24_tabicl_base.py` |
| TabICL_drugclip_concat       |  2% | 0.726 | `75_docking_unidock_full.py` → `78_drugclip_embeddings.py` → `24_tabicl_base.py` |

**Stacking**: `26_submission13.py` (NNLS meta-learner, run with `python 26_submission13.py 47`)

---

## Full dependency chain (copy these to public repo)

### Data & curation
- `00_download_data.py` — fetch train/test CSVs from HuggingFace
- `01_curation_run.py` — SMILES canonicalization, salt stripping, quality filters → `data/train_final.csv`
- `02_chembl_enrichment.py` — optional ChEMBL supplement (did not improve; excluded from best run)

### Base features
- `utils.py` — shared helpers: `butina_kfold`, descriptor utils, metrics
- `03_feature_generation.py` — ECFP4, RDKit2D, Mordred2D → `features/`

### Encoder pretraining (CheMeleon family)
- `23_chemeleon_hts_pretrain.py` — binary HTS → `features/chemeleon_hts_*.npy` (needs `cheminf_utils` env + GPU)
- `39_hts_continuous_pretrain.py` — continuous log2_fc + concentration descriptor → `features/chemeleon_hts_cont_*.npy`
- `43_hts_mtl_encoder.py` — continuous + soft-binary dual-head → `features/chemeleon_hts_mtl_*.npy`

### Foundation model fine-tuning
- `21_mole_finetune.py` — MolE (DeBERTa-style graph, GuacaMol pretrained) 50-epoch PXR fine-tune → `results/mole_e50_{oof,test}_preds.npy` (needs `mole_env`)
- `19_unimol_e50.py` — UniMol 50-epoch PXR fine-tune, 3 seeds × 5 folds (needs `unimol_env`)
- `19b_unimol_e50_aggregate.py` — aggregates seed OOFs → `results/unimol_ft_e50_oof_preds.npy`

### Docking + DrugCLIP
- `75_docking_unidock_full.py` — UniDock GPU docking, all 4596 compounds × 3 PXR receptors (2O9I, 8R81, 8EQZ)
- `78_drugclip_embeddings.py` — extract 128-d DrugCLIP embeddings from docked poses (needs `drugclip_env`)

### Foundation regressor
- `24_tabicl_base.py` — TabICL on all feature sets incl. chemeleon_hts + drugclip_concat (needs `tabicl_env`)
- `69_tabpfn_v3_oof.py` — TabPFN v3 Butina 3×5-fold OOF on chemeleon_hts_cont and _mtl (needs `TABPFN_TOKEN` + GPU)

### Stacking
- `26_submission13.py` — NNLS ensemble; pass sub number as CLI arg

---

## Conda environments needed

| Env | Key packages | Used by |
|---|---|---|
| `cheminf_utils` | chemprop≥2.2, tabpfn≥8.0.2, scikit-learn | most scripts |
| `tabicl_env` | tabicl (soda-inria), torch | `24_tabicl_base.py` |
| `mole_env` | mole_public (Python 3.10) | `21_mole_finetune.py` |
| `unimol_env` | unimol (Python 3.9) | `19_unimol_e50.py` |
| `drugclip_env` | unicore, torch 2.8+cuda | `78_drugclip_embeddings.py` |

---

## What was tried but excluded (zeroed by NNLS)

All other models (LGB/XGB on ECFP/Mordred/MQN, SVM, ChemBERTa, GP models, MIST fine-tuned,
MTL ChemProp, ADMET-AI standalone) received 0% NNLS weight when the HTS encoders + MolE +
UniMol were available. They are correlated with and weaker than the selected six.

**Meta-learner experiments**: Ridge, ElasticNet (positive), XGB stacker all gave higher blind RAE
than NNLS. Non-negativity constraint is load-bearing — prevents scale distortions from weaker models.

**Post-processing**: Quantile matching and linear rescaling to match training distribution both
hurt blind RAE (train std 0.963 vs test std 0.633 — compression is correct, not a bug to fix).
