"""
Hyperparameter optimisation for top GBM models using Optuna TPE.

Uses the same 5-fold Butina splits (seed=0) as the CV benchmark for fair comparison.
Runs 60 trials per model; saves best params to results/best_hparams.json.

Models tuned:
  XGB_all, LGB_all, XGB_ecfp4_rdkit, XGB_ecfp4_mordred, LGB_ecfp4_mordred

Run AFTER 03_feature_generation.py.  Takes ~2–4 h on CPU.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import FEATURE_DIR, DATA_DIR, butina_kfold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_FOLDS  = 5
CV_SEED  = 0        # single seed for HPO (speed); full 3-seed CV runs in 04
N_TRIALS = 60       # per model


# -----------------------------------------------------------------------
# Pre-compute splits once (expensive Butina distance matrix)
# -----------------------------------------------------------------------

def get_splits(smiles: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    print("Computing 5-fold Butina splits for HPO ...")
    splits = butina_kfold(smiles, k=N_FOLDS, seed=CV_SEED, cutoff=0.4)
    print(f"  {len(splits)} folds ready")
    return splits


def cv_rmse(model_factory, X: np.ndarray, y: np.ndarray,
            splits: list) -> float:
    """Mean RMSE across Butina folds."""
    rmses = []
    for tr_idx, va_idx in splits:
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[va_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y[va_idx], preds))))
    return float(np.mean(rmses))


# -----------------------------------------------------------------------
# Search spaces
# -----------------------------------------------------------------------

def xgb_objective(trial, X, y, splits):
    from xgboost import XGBRegressor
    params = dict(
        n_estimators       = trial.suggest_int("n_estimators", 200, 1500),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth          = trial.suggest_int("max_depth", 3, 10),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_child_weight   = trial.suggest_int("min_child_weight", 1, 10),
        gamma              = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha          = trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda         = trial.suggest_float("reg_lambda", 0.0, 5.0),
        random_state=42, n_jobs=-1, verbosity=0,
    )
    return cv_rmse(lambda: XGBRegressor(**params), X, y, splits)


def lgb_objective(trial, X, y, splits):
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators       = trial.suggest_int("n_estimators", 200, 1500),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth          = trial.suggest_int("max_depth", 3, 10),
        num_leaves         = trial.suggest_int("num_leaves", 20, 300),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha          = trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda         = trial.suggest_float("reg_lambda", 0.0, 5.0),
        min_child_samples  = trial.suggest_int("min_child_samples", 5, 50),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    return cv_rmse(lambda: LGBMRegressor(**params), X, y, splits)


def cat_objective(trial, X, y, splits):
    from catboost import CatBoostRegressor
    params = dict(
        iterations         = trial.suggest_int("iterations", 200, 1500),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth              = trial.suggest_int("depth", 3, 10),
        l2_leaf_reg        = trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bylevel  = trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        min_data_in_leaf   = trial.suggest_int("min_data_in_leaf", 1, 30),
        random_seed=42, verbose=0,
    )
    return cv_rmse(lambda: CatBoostRegressor(**params), X, y, splits)


# -----------------------------------------------------------------------
# HPO runner
# -----------------------------------------------------------------------

MODEL_CONFIGS = [
    # (model_name, feat_name, objective_fn)
    ("XGB_all_hpo",          "all_combined",   xgb_objective),
    ("LGB_all_hpo",          "all_combined",   lgb_objective),
    ("CAT_all_hpo",          "all_combined",   cat_objective),
    ("XGB_ecfp4_rdkit_hpo",  "ecfp4_rdkit",   xgb_objective),
    ("LGB_ecfp4_rdkit_hpo",  "ecfp4_rdkit",   lgb_objective),
    ("XGB_ecfp4_mordred_hpo","ecfp4_mordred",  xgb_objective),
    ("LGB_ecfp4_mordred_hpo","ecfp4_mordred",  lgb_objective),
]


def run_hpo_for(model_name: str, feat_name: str, objective_fn,
                X: np.ndarray, y: np.ndarray, splits: list,
                existing: dict) -> dict:
    if model_name in existing:
        print(f"  SKIP {model_name} (already in best_hparams.json)")
        return existing[model_name]

    print(f"\n{'='*60}")
    print(f"  HPO: {model_name}  feat={feat_name}  ({N_TRIALS} trials)")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X, y, splits),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_params
    best_val = study.best_value
    print(f"  Best RMSE: {best_val:.5f}")
    print(f"  Best params: {best}")
    return {"params": best, "cv_rmse": best_val, "feat_name": feat_name}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== Optuna HPO for GBM models ===\n")

    # Load data
    train_path = (DATA_DIR / "train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                  else DATA_DIR / "train_curated.csv")
    train_df   = pd.read_csv(train_path)
    y          = np.load(FEATURE_DIR / "y_train.npy")
    smiles     = train_df["SMILES"].tolist()

    assert len(y) == len(smiles)
    print(f"Training set: {len(y)} molecules\n")

    # Load existing results (for resume)
    hparams_path = RESULTS_DIR / "best_hparams.json"
    existing = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing HPO results from {hparams_path}")

    # Pre-compute splits once
    splits = get_splits(smiles)

    # Feature cache
    feat_cache: dict[str, np.ndarray] = {}

    results = dict(existing)

    for model_name, feat_name, obj_fn in MODEL_CONFIGS:
        if feat_name not in feat_cache:
            p = FEATURE_DIR / f"{feat_name}_train.npy"
            if not p.exists():
                print(f"  SKIP {model_name}: {feat_name} features not found")
                continue
            feat_cache[feat_name] = np.load(p)
            print(f"  Loaded {feat_name}: {feat_cache[feat_name].shape}")

        X = feat_cache[feat_name]
        result = run_hpo_for(model_name, feat_name, obj_fn, X, y, splits, existing)
        results[model_name] = result

        # Save after each model (resume-safe)
        with open(hparams_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {hparams_path}")

    # Summary
    print("\n=== HPO Summary ===")
    for name, info in sorted(results.items(), key=lambda x: x[1].get("cv_rmse", 9)):
        print(f"  {name:35s}  CV RMSE = {info['cv_rmse']:.5f}")

    print(f"\nAll done. Results saved to {hparams_path}")


if __name__ == "__main__":
    main()
