"""
Submission 2: enhanced ensemble with:
  - HPO-tuned GBMs (from 07_optuna_hpo.py)
  - Multi-seed ChemProp+Chemeleon (3 seeds, 50 epochs each)
  - SMILES test-time augmentation for ChemProp (5 random SMILES per molecule)
  - SVM Tanimoto (best variant from 09_cv_additions.py)
  - NNLS re-fitted on full OOF file

Saves results/submission_pEC50_2.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import FEATURE_DIR, DATA_DIR

warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "results"
OOF_DIR     = RESULTS_DIR / "oof_predictions"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CHEMPROP_SEEDS = [0, 1, 2]   # average over these seeds
TTA_N_AUGMENTS = 5            # random SMILES per molecule for ChemProp TTA


# -----------------------------------------------------------------------
# SMILES test-time augmentation helpers
# -----------------------------------------------------------------------

def enumerate_smiles(smi: str, n: int = 5, seed: int = 42) -> list[str]:
    """Return up to n distinct random SMILES enumerations (canonical first)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return [smi] * n
    canonical = Chem.MolToSmiles(mol)
    variants = [canonical]
    seen = {canonical}
    rng = np.random.default_rng(seed)
    for _ in range(n * 20):
        rand_smi = Chem.MolToSmiles(mol, doRandom=True,
                                    randomSeed=int(rng.integers(0, 100_000)))
        if rand_smi not in seen:
            variants.append(rand_smi)
            seen.add(rand_smi)
        if len(variants) >= n:
            break
    return variants


def chemprop_predict_tta(model, smiles_list: list[str],
                         n_augments: int = 5) -> np.ndarray:
    """Batch-predict all SMILES augmentations, then average per molecule."""
    all_variants = [enumerate_smiles(s, n=n_augments) for s in smiles_list]
    flat_smiles  = [s for variants in all_variants for s in variants]
    counts       = [len(v) for v in all_variants]

    df_flat = pd.DataFrame({"SMILES": flat_smiles,
                             "pEC50": [0.0] * len(flat_smiles)})
    all_preds = model.predict(df_flat)

    result = []
    idx = 0
    for c in counts:
        result.append(float(np.mean(all_preds[idx: idx + c])))
        idx += c
    return np.array(result)


# -----------------------------------------------------------------------
# Multi-seed ChemProp
# -----------------------------------------------------------------------

def train_chemprop_multiseed(smiles_train: list[str], y: np.ndarray,
                              smiles_test: list[str],
                              seeds: list[int],
                              num_epochs: int = 50,
                              tta: bool = True,
                              n_augments: int = 5) -> np.ndarray:
    """Train ChemProp with each seed; average test predictions."""
    import lightning.pytorch as pl
    from cheminf_utils.chemprop_helpers import ChemPropChemeleonWrapper

    train_df = pd.DataFrame({"SMILES": smiles_train, "pEC50": y})
    test_df  = pd.DataFrame({"SMILES": smiles_test,  "pEC50": [0.0] * len(smiles_test)})

    seed_preds = []
    for seed in seeds:
        print(f"  ChemProp seed={seed}  epochs={num_epochs}  TTA={tta} ...")
        pl.seed_everything(seed, workers=True)
        model = ChemPropChemeleonWrapper(y_name="pEC50")
        model.fit(train_df, num_epochs=num_epochs, accelerator="auto")

        if tta:
            preds = chemprop_predict_tta(model, smiles_test, n_augments=n_augments)
        else:
            preds = model.predict(test_df)

        seed_preds.append(preds)
        print(f"    seed={seed}  mean={preds.mean():.3f}  std={preds.std():.3f}")

    ensemble = np.mean(seed_preds, axis=0)
    print(f"  Multi-seed mean: {ensemble.mean():.3f}  std={ensemble.std():.3f}")
    return ensemble


# -----------------------------------------------------------------------
# NNLS re-fit on full OOF (all models available)
# -----------------------------------------------------------------------

def fit_nnls(model_names: list[str]) -> dict[str, float]:
    oof_path = OOF_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        return {}
    oof_df = pd.read_csv(oof_path)
    y_true = oof_df["y_true"].values
    cols = [m for m in model_names if m in oof_df.columns]
    if not cols:
        return {}
    X = oof_df[cols].values
    mask = ~np.isnan(X).any(axis=1)
    w_raw, _ = nnls(X[mask], y_true[mask])
    if w_raw.sum() == 0:
        w_raw = np.ones(len(cols))
    w = w_raw / w_raw.sum()
    return dict(zip(cols, w.round(6)))


# -----------------------------------------------------------------------
# Helpers to instantiate base models
# -----------------------------------------------------------------------

def instantiate_from_hparams(model_name: str, params: dict):
    if model_name.startswith("XGB"):
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    if model_name.startswith("LGB"):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    if model_name.startswith("CAT"):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params, random_seed=42, verbose=0)
    raise ValueError(f"Unknown model type: {model_name}")


def feat_name_for(model_name: str) -> str:
    if "3dqsar" in model_name:
        return "ecfp4_rdkit_3dqsar"
    if "mordred3d" in model_name:
        return "ecfp4_mordred3d"
    if "mordred" in model_name:
        return "ecfp4_mordred"
    if "_all" in model_name:
        return "all_combined"
    if "ecfp4_rdkit" in model_name:
        return "ecfp4_rdkit"
    if "ecfp4" in model_name:
        return "ecfp4"
    return "ecfp4_rdkit"


def tanimoto_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    XY   = X @ Y.T
    denom = X.sum(1, keepdims=True) + Y.sum(1, keepdims=True).T - XY
    return np.where(denom > 0, XY / denom, 0.0)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== Submission 2: HPO + Multi-seed ChemProp + SMILES TTA ===\n")

    # -- Load data -------------------------------------------------------
    train_path = (DATA_DIR / "train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                  else DATA_DIR / "train_curated.csv")
    test_path  = (DATA_DIR / "test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
                  else DATA_DIR / "test_raw.csv")
    train_df   = pd.read_csv(train_path)
    test_df    = pd.read_csv(test_path)

    y            = np.load(FEATURE_DIR / "y_train.npy")
    smiles_train = train_df["SMILES"].tolist()
    smiles_test  = test_df["SMILES"].tolist()

    test_preds: dict[str, np.ndarray] = {}

    # -- 1. Multi-seed ChemProp + SMILES TTA ----------------------------
    print("\n[1/3] Multi-seed ChemProp+Chemeleon (seeds={}, TTA={} augments)"
          .format(CHEMPROP_SEEDS, TTA_N_AUGMENTS))
    try:
        from cheminf_utils.chemprop_helpers import ChemPropChemeleonWrapper  # noqa
        cp_preds = train_chemprop_multiseed(
            smiles_train, y, smiles_test,
            seeds=CHEMPROP_SEEDS, num_epochs=50,
            tta=True, n_augments=TTA_N_AUGMENTS,
        )
        test_preds["ChemProp_Chemeleon_multiseed"] = cp_preds
    except Exception as e:
        print(f"  ChemProp multi-seed failed: {e}")
        import traceback; traceback.print_exc()

    # -- 2. HPO-tuned GBMs ----------------------------------------------
    print("\n[2/3] HPO-tuned GBM models")
    hparams_path = RESULTS_DIR / "best_hparams.json"
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)
        for model_name, info in hparams.items():
            feat_name = info["feat_name"]
            params    = info["params"]
            feat_tr   = FEATURE_DIR / f"{feat_name}_train.npy"
            feat_te   = FEATURE_DIR / f"{feat_name}_test.npy"
            if not feat_tr.exists() or not feat_te.exists():
                print(f"  SKIP {model_name}: features not found")
                continue
            print(f"  Retraining {model_name} (CV RMSE={info['cv_rmse']:.5f}) ...")
            try:
                X_tr = np.load(feat_tr)
                X_te = np.load(feat_te)
                model = instantiate_from_hparams(model_name, params)
                model.fit(X_tr, y)
                preds = model.predict(X_te)
                test_preds[model_name] = preds
                print(f"    preds: mean={preds.mean():.3f}  std={preds.std():.3f}")
            except Exception as e:
                print(f"  ERROR in {model_name}: {e}")
    else:
        print("  No best_hparams.json — run 07_optuna_hpo.py first")

    # -- Fallback: submission-1 base models (always include for stability)
    print("\n  Adding submission-1 base models as fallback ...")
    BASE_MODELS = [
        ("XGB_ecfp4_mordred", "ecfp4_mordred"),
        ("XGB_all",           "all_combined"),
        ("XGB_ecfp4_rdkit",   "ecfp4_rdkit"),
        ("LGB_all",           "all_combined"),
        ("EN_ecfp4",          "ecfp4"),
    ]
    for model_name, feat_name in BASE_MODELS:
        if model_name in test_preds:
            continue
        try:
            X_tr = np.load(FEATURE_DIR / f"{feat_name}_train.npy")
            X_te = np.load(FEATURE_DIR / f"{feat_name}_test.npy")
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.linear_model import ElasticNetCV
            if model_name.startswith("XGB"):
                m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8 if "all" not in model_name else 0.5,
                                 random_state=42, n_jobs=-1, verbosity=0)
            elif model_name.startswith("LGB"):
                m = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.5,
                                  random_state=42, n_jobs=-1, verbose=-1)
            else:
                m = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0], cv=5,
                                 max_iter=5000, random_state=42, n_jobs=-1)
            m.fit(X_tr, y)
            test_preds[model_name] = m.predict(X_te)
            print(f"    {model_name}: mean={test_preds[model_name].mean():.3f}")
        except Exception as e:
            print(f"  SKIP {model_name}: {e}")

    # -- 3. Best SVM from OOF results ------------------------------------
    print("\n[3/3] SVM with Tanimoto kernel (best C from OOF)")
    oof_path = OOF_DIR / "oof_predictions.csv"
    best_svm = None
    if oof_path.exists():
        oof_df  = pd.read_csv(oof_path)
        y_oof   = oof_df["y_true"].values
        svm_cols = [c for c in oof_df.columns if c.startswith("SVM_tanimoto")]
        if svm_cols:
            # Pick SVM variant with best OOF RMSE
            svm_rmses = {}
            for col in svm_cols:
                mask = ~np.isnan(oof_df[col].values)
                if mask.sum() > 100:
                    r = float(np.sqrt(((y_oof[mask] - oof_df[col].values[mask])**2).mean()))
                    svm_rmses[col] = r
            if svm_rmses:
                best_svm = min(svm_rmses, key=svm_rmses.get)
                print(f"  Best SVM: {best_svm}  OOF RMSE={svm_rmses[best_svm]:.5f}")

    if best_svm is not None:
        C_val = float(best_svm.split("_C")[1])
        X_tr_ecfp4 = np.load(FEATURE_DIR / "ecfp4_train.npy")
        X_te_ecfp4 = np.load(FEATURE_DIR / "ecfp4_test.npy")
        K_tr = tanimoto_kernel(X_tr_ecfp4, X_tr_ecfp4)
        K_te = tanimoto_kernel(X_te_ecfp4, X_tr_ecfp4)
        from sklearn.svm import SVR
        svm = SVR(kernel="precomputed", C=C_val, epsilon=0.05)
        svm.fit(K_tr, y)
        svm_preds = svm.predict(K_te)
        test_preds["SVM_tanimoto"] = svm_preds
        print(f"  SVM preds: mean={svm_preds.mean():.3f}  std={svm_preds.std():.3f}")
    else:
        print("  No SVM OOF data found — skipping (run 09_cv_additions.py first)")

    # -- NNLS ensemble ---------------------------------------------------
    print(f"\nFitting NNLS over {len(test_preds)} models ...")
    available = list(test_preds.keys())
    weights = fit_nnls(available)

    if weights:
        print("NNLS weights (from OOF):")
        for m, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "#" * int(w * 40)
            print(f"  {w:.4f}  {bar}  {m}")
    else:
        print("  OOF NNLS unavailable — using equal weights")
        weights = {m: 1.0 / len(available) for m in available}

    pred_matrix = np.column_stack([test_preds[m] for m in available])
    w_vec = np.array([weights.get(m, 0.0) for m in available])
    if w_vec.sum() > 0:
        w_vec /= w_vec.sum()
    final_preds = np.clip(pred_matrix @ w_vec, 1.0, 9.0)

    print(f"\nFinal: range=[{final_preds.min():.3f}, {final_preds.max():.3f}]  "
          f"mean={final_preds.mean():.3f}  std={final_preds.std():.3f}")

    # -- Save ------------------------------------------------------------
    submission = test_df[["SMILES", "Molecule Name"]].copy()
    submission["pEC50"] = final_preds

    out_path = RESULTS_DIR / "submission_pEC50_2.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(submission)} rows)")
    assert submission["pEC50"].isna().sum() == 0
    assert submission.columns.tolist() == ["SMILES", "Molecule Name", "pEC50"]
    print("All checks passed.")


if __name__ == "__main__":
    main()
