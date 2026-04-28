"""
Train final ensemble on full training set and predict test set.

Run AFTER 04_cv_benchmark.py + 06_ensemble_analysis.py.

Strategy (from OOF analysis):
  - Greedy Caruana ensemble: ChemProp_Chemeleon(40%) + XGB models(20% each)
  - Fallback: NNLS weights if ChemProp fails, then mean top-3
  - Retrain each base model on full training set, combine with OOF-derived weights
"""
from __future__ import annotations

import pickle
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


# -----------------------------------------------------------------------
# Helpers to instantiate models (mirrors 04_cv_benchmark.py)
# -----------------------------------------------------------------------

def instantiate_model(model_name: str):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV
    from xgboost import XGBRegressor

    registry = {
        "XGB_ecfp4_rdkit":    XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                                            n_jobs=-1, verbosity=0),
        "LGB_ecfp4_rdkit":    LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                                            n_jobs=-1, verbose=-1),
        "CAT_ecfp4_rdkit":    CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                random_seed=42, verbose=0),
        "RF_ecfp4_rdkit":     RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                    random_state=42, n_jobs=-1),
        "EN_ecfp4":           ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0], cv=5,
                                          max_iter=5000, random_state=42, n_jobs=-1),
        "XGB_ecfp4_mordred":  XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                                           n_jobs=-1, verbosity=0),
        "LGB_ecfp4_mordred":  LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                                            n_jobs=-1, verbose=-1),
        "CAT_ecfp4_mordred":  CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                random_seed=42, verbose=0),
        "XGB_3dqsar":          XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.5, random_state=42,
                                            n_jobs=-1, verbosity=0),
        "LGB_3dqsar":          LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                             subsample=0.8, colsample_bytree=0.5, random_state=42,
                                             n_jobs=-1, verbose=-1),
        "CAT_3dqsar":          CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                 random_seed=42, verbose=0),
        "XGB_ecfp4_mordred3d": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.5, random_state=42,
                                            n_jobs=-1, verbosity=0),
        "LGB_ecfp4_mordred3d": LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                             subsample=0.8, colsample_bytree=0.5, random_state=42,
                                             n_jobs=-1, verbose=-1),
        "CAT_ecfp4_mordred3d": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                 random_seed=42, verbose=0),
        "XGB_all":            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.5, random_state=42,
                                           n_jobs=-1, verbosity=0),
        "LGB_all":            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.5, random_state=42,
                                            n_jobs=-1, verbose=-1),
        "CAT_all":            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                random_seed=42, verbose=0),
    }
    return registry[model_name]


def feat_name_for(model_name: str) -> str:
    if "3dqsar" in model_name:
        return "ecfp4_rdkit_3dqsar"
    if "mordred3d" in model_name:
        return "ecfp4_mordred3d"
    if "mordred" in model_name:
        return "ecfp4_mordred"
    if "_all" in model_name:
        return "all_combined"
    if "ecfp4_rdkit" in model_name or model_name.startswith(("CAT_ecfp", "RF", "TabPFN")):
        return "ecfp4_rdkit"
    if "ecfp4" in model_name:
        return "ecfp4"
    return "ecfp4_rdkit"


# -----------------------------------------------------------------------
# Derive final weights from OOF analysis
# -----------------------------------------------------------------------

def derive_nnls_weights(model_names: list[str]) -> dict[str, float]:
    """Re-fit NNLS on OOF predictions for the requested model subset."""
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
# Main
# -----------------------------------------------------------------------

def main():
    print("=== Final Ensemble + Submission ===\n")

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

    # -- Model set (from ensemble analysis: greedy Caruana top selection) -
    # Full NNLS set for maximum accuracy; skip ChemProp gracefully if unavailable
    NNLS_MODELS = [
        "ChemProp_Chemeleon",   # 41.6% — most important, add graph-level features
        "XGB_ecfp4_mordred",    # 15.8%
        "XGB_all",              # 15.0%
        "XGB_ecfp4_rdkit",      # 11.3%
        "LGB_all",              #  5.4%
        "EN_ecfp4",             #  3.8%
        "LGB_3dqsar",           #  2.4%
        "CAT_ecfp4_mordred3d",  #  1.9%
        "XGB_3dqsar",           #  1.7%
        "LGB_ecfp4_rdkit",      #  1.0%
    ]

    # -- Train and predict each base model --------------------------------
    test_preds: dict[str, np.ndarray] = {}

    for model_name in NNLS_MODELS:
        print(f"\nRetraining {model_name} on full train set ...")

        if model_name == "ChemProp_Chemeleon":
            try:
                from cheminf_utils.chemprop_helpers import ChemPropChemeleonWrapper
                work_df_tr = pd.DataFrame({"SMILES": smiles_train, "pEC50": y})
                work_df_te = pd.DataFrame({"SMILES": smiles_test, "pEC50": [0.0] * len(smiles_test)})
                model = ChemPropChemeleonWrapper(y_name="pEC50")
                model.fit(work_df_tr, num_epochs=50, accelerator="auto")
                preds_test = model.predict(work_df_te)
                test_preds[model_name] = preds_test
                print(f"  Test preds: mean={preds_test.mean():.3f}  std={preds_test.std():.3f}")
            except Exception as e:
                print(f"  ChemProp_Chemeleon FAILED: {e} — will skip from ensemble")
            continue

        try:
            feat_name = feat_name_for(model_name)
            feat_path    = FEATURE_DIR / f"{feat_name}_train.npy"
            feat_path_te = FEATURE_DIR / f"{feat_name}_test.npy"

            if not feat_path.exists() or not feat_path_te.exists():
                print(f"  SKIP: {feat_name} features not found")
                continue

            X_tr = np.load(feat_path)
            X_te = np.load(feat_path_te)
            model = instantiate_model(model_name)
            model.fit(X_tr, y)
            preds_test = model.predict(X_te)
            test_preds[model_name] = preds_test
            print(f"  Test preds: mean={preds_test.mean():.3f}  std={preds_test.std():.3f}")

            with open(MODELS_DIR / f"{model_name}.pkl", "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"  ERROR in {model_name}: {e} — skipping")

    if not test_preds:
        print("ERROR: no predictions generated")
        sys.exit(1)

    available = list(test_preds.keys())
    print(f"\nModels available for ensemble: {available}")

    # -- Derive NNLS weights for available models -------------------------
    weights = derive_nnls_weights(available)
    print("\nNNLS ensemble weights (from OOF):")
    for m, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {w:.4f}  {m}")

    # -- Final predictions ------------------------------------------------
    if weights and len(available) >= 2:
        pred_matrix = np.column_stack([test_preds[m] for m in available])
        w_vec = np.array([weights.get(m, 0.0) for m in available])
        if w_vec.sum() > 0:
            w_vec /= w_vec.sum()
        final_preds = pred_matrix @ w_vec
        method = "NNLS ensemble"
    elif len(available) >= 2:
        pred_matrix = np.column_stack([test_preds[m] for m in available])
        final_preds = pred_matrix.mean(axis=1)
        method = "simple mean (NNLS weights unavailable)"
    else:
        final_preds = list(test_preds.values())[0]
        method = f"single model ({available[0]})"

    final_preds = np.clip(final_preds, 1.0, 9.0)
    print(f"\nFinal predictions via {method}")
    print(f"  range=[{final_preds.min():.3f}, {final_preds.max():.3f}]  "
          f"mean={final_preds.mean():.3f}  std={final_preds.std():.3f}")

    # -- Save submission --------------------------------------------------
    submission = test_df[["Molecule Name"]].copy()
    submission["pEC50_pred"] = final_preds
    for m, preds in test_preds.items():
        submission[f"pEC50_{m}"] = preds

    out_path = RESULTS_DIR / "submission_pEC50.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(submission)} rows)")

    assert len(submission) == len(test_df)
    assert submission["pEC50_pred"].isna().sum() == 0
    print("All checks passed.")


if __name__ == "__main__":
    main()
