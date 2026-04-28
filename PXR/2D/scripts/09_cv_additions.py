"""
Add new model types to existing CV results:
  1. SVM with Tanimoto kernel (on ECFP4)
  2. HPO-tuned GBM variants (from 07_optuna_hpo.py)

Appends to results/cv_results.csv and results/oof_predictions/oof_predictions.csv.
Already-completed models are skipped (resume-safe).

Run AFTER 07_optuna_hpo.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import (
    FEATURE_DIR, DATA_DIR,
    compute_metrics, repeated_butina_cv,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "results"
OOF_DIR     = RESULTS_DIR / "oof_predictions"
RESULTS_DIR.mkdir(exist_ok=True)
OOF_DIR.mkdir(exist_ok=True)

N_FOLDS   = 5
N_REPEATS = 3
SEEDS     = [0, 1, 2]

CV_PATH  = RESULTS_DIR / "cv_results.csv"
OOF_PATH = OOF_DIR / "oof_predictions.csv"


# -----------------------------------------------------------------------
# Tanimoto kernel (vectorised, bit-vector ECFP4)
# -----------------------------------------------------------------------

def tanimoto_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """K(i,j) = |X_i ∩ Y_j| / |X_i ∪ Y_j|  (bit vectors)."""
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    XY   = X @ Y.T
    Xsum = X.sum(axis=1, keepdims=True)
    Ysum = Y.sum(axis=1, keepdims=True)
    denom = Xsum + Ysum.T - XY
    return np.where(denom > 0, XY / denom, 0.0)


# -----------------------------------------------------------------------
# Incremental save helpers (mirrors 04_cv_benchmark.py)
# -----------------------------------------------------------------------

def load_completed() -> set[str]:
    if not CV_PATH.exists():
        return set()
    df = pd.read_csv(CV_PATH)
    expected = N_FOLDS * N_REPEATS
    counts = df.groupby("model").size()
    return set(counts[counts >= expected].index.tolist())


def save_records(records: list[dict]):
    if not records:
        return
    df_new = pd.DataFrame(records)
    if CV_PATH.exists():
        df_all = pd.concat([pd.read_csv(CV_PATH), df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CV_PATH, index=False)


def save_oof(name: str, arr: np.ndarray, y: np.ndarray):
    if OOF_PATH.exists():
        df = pd.read_csv(OOF_PATH)
    else:
        df = pd.DataFrame({"y_true": y})
    df[name] = arr
    df.to_csv(OOF_PATH, index=False)


# -----------------------------------------------------------------------
# Generic CV runner (pre-computed splits)
# -----------------------------------------------------------------------

def run_cv(model_name: str, model_factory, X: np.ndarray,
           y: np.ndarray, splits: list,
           kernel_X: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    """
    model_factory: callable() → fresh model instance.
    kernel_X: if not None, treat X as identity indices; kernel_X is the
              full (n, n) precomputed kernel matrix, and X is ignored.
    """
    records   = []
    oof_preds = np.full(len(y), np.nan)
    use_kernel = kernel_X is not None

    for repeat_idx, fold_idx, tr_idx, va_idx in splits:
        m = model_factory()
        if use_kernel:
            K_tr = kernel_X[np.ix_(tr_idx, tr_idx)]
            K_va = kernel_X[np.ix_(va_idx, tr_idx)]
            m.fit(K_tr, y[tr_idx])
            preds = m.predict(K_va)
        else:
            m.fit(X[tr_idx], y[tr_idx])
            preds = m.predict(X[va_idx])

        if repeat_idx == 0:
            oof_preds[va_idx] = preds

        met = compute_metrics(y[va_idx], preds)
        met.update({"model": model_name, "repeat": repeat_idx, "fold": fold_idx,
                    "n_train": len(tr_idx), "n_val": len(va_idx)})
        records.append(met)
        print(f"    r={repeat_idx} f={fold_idx}  RMSE={met['RMSE']:.4f}  "
              f"R2={met['R2']:.3f}  Spearman={met['Spearman']:.3f}")

    return records, oof_preds


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== CV Additions: SVM Tanimoto + HPO-tuned GBMs ===\n")

    train_path = (DATA_DIR / "train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                  else DATA_DIR / "train_curated.csv")
    train_df   = pd.read_csv(train_path)
    y          = np.load(FEATURE_DIR / "y_train.npy")
    smiles     = train_df["SMILES"].tolist()

    completed = load_completed()
    if completed:
        print(f"Already completed: {sorted(completed)}\n")

    # OOF dict for accumulating
    oof_dict: dict[str, np.ndarray] = {}
    if OOF_PATH.exists():
        oof_df = pd.read_csv(OOF_PATH)
        for col in oof_df.columns:
            if col != "y_true":
                oof_dict[col] = oof_df[col].values

    # Pre-compute Butina splits once
    print("Pre-computing Butina splits ...")
    splits = repeated_butina_cv(smiles, k=N_FOLDS, n_repeats=N_REPEATS, seeds=SEEDS)
    print(f"  {len(splits)} (repeat, fold) pairs ready\n")

    # ------------------------------------------------------------------ #
    # 1. SVM with Tanimoto kernel
    # ------------------------------------------------------------------ #
    svm_configs = [
        # (model_name, C, epsilon)  — moderate hyperparams as starting point;
        # these are reasonable defaults for pEC50 regression on ECFP4
        ("SVM_tanimoto_C1",   1.0,  0.05),
        ("SVM_tanimoto_C10",  10.0, 0.05),
        ("SVM_tanimoto_C100", 100.0, 0.05),
    ]

    ecfp4 = np.load(FEATURE_DIR / "ecfp4_train.npy")
    print(f"Computing Tanimoto kernel ({ecfp4.shape[0]}×{ecfp4.shape[0]}) ...")
    K_full = tanimoto_kernel(ecfp4, ecfp4)
    print(f"  Kernel matrix: {K_full.shape}  dtype={K_full.dtype}\n")

    for model_name, C, eps in svm_configs:
        if model_name in completed:
            print(f"  SKIP {model_name} (already done)")
            continue
        print(f"\n--- {model_name}  C={C}  epsilon={eps} ---")
        try:
            records, oof = run_cv(
                model_name,
                lambda C=C, eps=eps: SVR(kernel="precomputed", C=C, epsilon=eps),
                X=ecfp4, y=y, splits=splits, kernel_X=K_full,
            )
            save_records(records)
            oof_dict[model_name] = oof
            save_oof(model_name, oof, y)
            completed.add(model_name)
            mean_r = np.mean([r["RMSE"] for r in records])
            print(f"  Mean RMSE: {mean_r:.5f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # 2. HPO-tuned GBM models
    # ------------------------------------------------------------------ #
    hparams_path = RESULTS_DIR / "best_hparams.json"
    if not hparams_path.exists():
        print("\nNo best_hparams.json found — skipping HPO models. "
              "Run 07_optuna_hpo.py first.")
    else:
        with open(hparams_path) as f:
            hparams = json.load(f)

        print(f"\nLoaded HPO params for {len(hparams)} models")

        for model_name, info in hparams.items():
            if model_name in completed:
                print(f"  SKIP {model_name} (already done)")
                continue

            feat_name = info["feat_name"]
            params    = info["params"]
            feat_path = FEATURE_DIR / f"{feat_name}_train.npy"
            if not feat_path.exists():
                print(f"  SKIP {model_name}: {feat_name} not found")
                continue

            X = np.load(feat_path)
            print(f"\n--- {model_name}  feat={feat_name}  X={X.shape} ---")

            try:
                # Determine model type from name prefix
                if model_name.startswith("XGB"):
                    from xgboost import XGBRegressor
                    factory = lambda p=params: XGBRegressor(
                        **p, random_state=42, n_jobs=-1, verbosity=0)
                elif model_name.startswith("LGB"):
                    from lightgbm import LGBMRegressor
                    factory = lambda p=params: LGBMRegressor(
                        **p, random_state=42, n_jobs=-1, verbose=-1)
                elif model_name.startswith("CAT"):
                    from catboost import CatBoostRegressor
                    factory = lambda p=params: CatBoostRegressor(
                        **p, random_seed=42, verbose=0)
                else:
                    print(f"  Unknown model type for {model_name}")
                    continue

                records, oof = run_cv(model_name, factory, X, y, splits)
                save_records(records)
                oof_dict[model_name] = oof
                save_oof(model_name, oof, y)
                completed.add(model_name)
                mean_r = np.mean([r["RMSE"] for r in records])
                print(f"  Mean RMSE: {mean_r:.5f}  (HPO CV was {info['cv_rmse']:.5f})")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # 3. Updated summary
    # ------------------------------------------------------------------ #
    if CV_PATH.exists():
        cv_df = pd.read_csv(CV_PATH)
        summary = (
            cv_df.groupby("model")[["RMSE", "Spearman"]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.sort_values("RMSE_mean")
        print("\n=== Updated CV Summary ===")
        print(summary.to_string())
        summary.to_csv(RESULTS_DIR / "cv_summary.csv")
        print(f"\nSaved updated cv_summary.csv")


if __name__ == "__main__":
    main()
