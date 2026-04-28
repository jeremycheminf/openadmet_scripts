"""
3 × 5-fold Butina CV benchmark across multiple models and descriptor sets.

Run AFTER 03_feature_generation.py.

Produces:
  results/cv_results.csv          - per-fold metrics for every model
  results/cv_summary.csv          - summary table
  results/cv_summary.png          - boxplot
  results/oof_predictions/        - OOF preds for stacking

Supports resume: already-completed models are skipped if cv_results.csv exists.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils import (
    FEATURE_DIR, DATA_DIR,
    compute_metrics, load_features, repeated_butina_cv,
    COL_PECSO,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof_predictions"
RESULTS_DIR.mkdir(exist_ok=True)
OOF_DIR.mkdir(exist_ok=True)

N_FOLDS   = 5
N_REPEATS = 3
SEEDS     = [0, 1, 2]


# -----------------------------------------------------------------------
# Extra fingerprints (Avalon, atom-pair, topo-torsion) — computed on-the-fly
# -----------------------------------------------------------------------

def compute_extra_fps(smiles_list: list[str]) -> np.ndarray:
    """Avalon (512) + Atom-pair (512) + RDKit path (1024) concatenated."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    try:
        from rdkit.Avalon.pyAvalonTools import GetAvalonFP
        use_avalon = True
    except ImportError:
        use_avalon = False

    n = len(smiles_list)
    ap_bits, av_bits, rdk_bits = 512, 512, 1024
    total = (av_bits if use_avalon else 0) + ap_bits + rdk_bits
    arr = np.zeros((n, total), dtype=np.uint8)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        col = 0
        if use_avalon:
            fp = GetAvalonFP(mol, nBits=av_bits)
            DataStructs.ConvertToNumpyArray(fp, arr[i, col:col+av_bits])
            col += av_bits
        fp_ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ap_bits)
        DataStructs.ConvertToNumpyArray(fp_ap, arr[i, col:col+ap_bits])
        col += ap_bits
        fp_rdk = Chem.RDKFingerprint(mol, fpSize=rdk_bits)
        DataStructs.ConvertToNumpyArray(fp_rdk, arr[i, col:col+rdk_bits])
    return arr


# -----------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------

def make_models(feat_cache: dict[str, np.ndarray]) -> dict:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV
    from xgboost import XGBRegressor

    models = {}

    # ---- Gradient boosting on ecfp4_rdkit ----
    if "ecfp4_rdkit" in feat_cache:
        models["XGB_ecfp4_rdkit"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=-1, verbosity=0),
            "ecfp4_rdkit"
        )
        models["LGB_ecfp4_rdkit"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, verbose=-1),
            "ecfp4_rdkit"
        )
        models["CAT_ecfp4_rdkit"] = (
            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=42, verbose=0),
            "ecfp4_rdkit"
        )
        models["RF_ecfp4_rdkit"] = (
            RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1),
            "ecfp4_rdkit"
        )
        # EN on ecfp4_rdkit REMOVED — mixed-scale features cause divergence

    # ---- ElasticNet on ECFP4 only (binary, safe for linear models) ----
    if "ecfp4" in feat_cache:
        models["EN_ecfp4"] = (
            ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                         cv=5, max_iter=5000, random_state=42, n_jobs=-1),
            "ecfp4"
        )

    # ---- Mordred 2D variants ----
    if "ecfp4_mordred" in feat_cache:
        models["XGB_ecfp4_mordred"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=-1, verbosity=0),
            "ecfp4_mordred"
        )
        models["LGB_ecfp4_mordred"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, verbose=-1),
            "ecfp4_mordred"
        )
        models["CAT_ecfp4_mordred"] = (
            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=42, verbose=0),
            "ecfp4_mordred"
        )

    # ---- 3D QSAR: ECFP4 + RDKit2D + WHIM/GETAWAY/RDF/MORSE/AUTOCORR3D ----
    if "ecfp4_rdkit_3dqsar" in feat_cache:
        models["XGB_3dqsar"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.5, random_state=42,
                         n_jobs=-1, verbosity=0),
            "ecfp4_rdkit_3dqsar"
        )
        models["LGB_3dqsar"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.5, random_state=42,
                          n_jobs=-1, verbose=-1),
            "ecfp4_rdkit_3dqsar"
        )
        models["CAT_3dqsar"] = (
            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=42, verbose=0),
            "ecfp4_rdkit_3dqsar"
        )

    # ---- Mordred 3D (ecfp4 + mordred2d + mordred3d) ----
    if "ecfp4_mordred3d" in feat_cache:
        models["XGB_ecfp4_mordred3d"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.5, random_state=42,
                         n_jobs=-1, verbosity=0),
            "ecfp4_mordred3d"
        )
        models["LGB_ecfp4_mordred3d"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.5, random_state=42,
                          n_jobs=-1, verbose=-1),
            "ecfp4_mordred3d"
        )
        models["CAT_ecfp4_mordred3d"] = (
            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=42, verbose=0),
            "ecfp4_mordred3d"
        )

    # ---- Extra fingerprints (Avalon + AP + RDK path) ----
    if "extra_fps" in feat_cache:
        models["XGB_extra_fps"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=-1, verbosity=0),
            "extra_fps"
        )
        models["LGB_extra_fps"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, verbose=-1),
            "extra_fps"
        )

    # ---- All combined (ecfp4 + rdkit2d + extra_fps) ----
    if "all_combined" in feat_cache:
        models["XGB_all"] = (
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.5, random_state=42,
                         n_jobs=-1, verbosity=0),
            "all_combined"
        )
        models["LGB_all"] = (
            LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.5, random_state=42,
                          n_jobs=-1, verbose=-1),
            "all_combined"
        )
        models["CAT_all"] = (
            CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                              random_seed=42, verbose=0),
            "all_combined"
        )

    return models


def tabpfn_model():
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(n_estimators=16, device="auto")
    except Exception as e:
        print(f"  TabPFN not available: {e}")
        return None


# -----------------------------------------------------------------------
# Incremental save helpers
# -----------------------------------------------------------------------

CV_RESULTS_PATH = RESULTS_DIR / "cv_results.csv"
OOF_PATH = OOF_DIR / "oof_predictions.csv"


def save_records_incremental(records: list[dict]):
    """Append new fold records to cv_results.csv."""
    if not records:
        return
    df_new = pd.DataFrame(records)
    if CV_RESULTS_PATH.exists():
        df_existing = pd.read_csv(CV_RESULTS_PATH)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CV_RESULTS_PATH, index=False)


def load_completed_models() -> set[str]:
    """Return set of model names already fully recorded in cv_results.csv."""
    if not CV_RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(CV_RESULTS_PATH)
    expected = N_FOLDS * N_REPEATS
    counts = df.groupby("model").size()
    return set(counts[counts >= expected].index.tolist())


def save_oof(oof_dict: dict[str, np.ndarray], y: np.ndarray):
    """Merge new OOF arrays into oof_predictions.csv."""
    if OOF_PATH.exists():
        existing = pd.read_csv(OOF_PATH)
    else:
        existing = pd.DataFrame({"y_true": y})
    for name, arr in oof_dict.items():
        existing[name] = arr
    existing.to_csv(OOF_PATH, index=False)


# -----------------------------------------------------------------------
# ChemPropChemeleonWrapper runner  (pre-computed splits)
# -----------------------------------------------------------------------

def run_chemeleon_cv(
    model_name: str,
    y: np.ndarray,
    smiles_list: list[str],
    splits: list,
    num_epochs: int = 30,
) -> tuple[list[dict], np.ndarray]:
    from cheminf_utils.chemprop_helpers import ChemPropChemeleonWrapper

    records = []
    oof_preds = np.full(len(y), np.nan)
    work_df = pd.DataFrame({"SMILES": smiles_list, "pEC50": y})

    for repeat_idx, fold_idx, train_idx, val_idx in splits:
        tr_df = work_df.iloc[train_idx].reset_index(drop=True)
        va_df = work_df.iloc[val_idx].reset_index(drop=True)

        model = ChemPropChemeleonWrapper(y_name="pEC50")
        try:
            model.fit(tr_df, num_epochs=num_epochs, accelerator="auto")
            preds = model.predict(va_df)
        except Exception as e:
            print(f"    ERROR r={repeat_idx} f={fold_idx}: {e}")
            continue

        if repeat_idx == 0:
            oof_preds[val_idx] = preds

        m = compute_metrics(y[val_idx], preds)
        m.update({"model": model_name, "repeat": repeat_idx, "fold": fold_idx,
                  "n_train": len(train_idx), "n_val": len(val_idx)})
        records.append(m)
        print(f"    r={repeat_idx} f={fold_idx}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.3f}")

    return records, oof_preds


# -----------------------------------------------------------------------
# Generic CV runner for sklearn-compatible models  (pre-computed splits)
# -----------------------------------------------------------------------

def run_cv_for_model(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
) -> tuple[list[dict], np.ndarray]:
    records = []
    oof_preds = np.full(len(y), np.nan)

    for repeat_idx, fold_idx, train_idx, val_idx in splits:
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])

        if repeat_idx == 0:
            oof_preds[val_idx] = preds

        m = compute_metrics(y[val_idx], preds)
        m.update({"model": model_name, "repeat": repeat_idx, "fold": fold_idx,
                  "n_train": len(train_idx), "n_val": len(val_idx)})
        records.append(m)
        print(f"    r={repeat_idx} f={fold_idx}  RMSE={m['RMSE']:.4f}  "
              f"R2={m['R2']:.3f}  Spearman={m['Spearman']:.3f}")

    return records, oof_preds


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== 3×5-fold Butina CV Benchmark ===\n")

    # -- Load targets & SMILES -------------------------------------------
    train_path = (DATA_DIR / "train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                  else DATA_DIR / "train_curated.csv")
    train_df   = pd.read_csv(train_path)
    y          = np.load(FEATURE_DIR / "y_train.npy")
    smiles_list = train_df["SMILES"].tolist()

    assert len(y) == len(smiles_list), \
        f"y_train ({len(y)}) vs SMILES ({len(smiles_list)}) mismatch — re-run 03_feature_generation.py"
    print(f"Training set: {len(y)} molecules,  pEC50 [{y.min():.2f}, {y.max():.2f}]\n")

    # -- Resume: find already-completed models ---------------------------
    completed = load_completed_models()
    if completed:
        print(f"Resuming — already completed: {sorted(completed)}\n")

    # -- Load OOF dict from disk (for resume) ----------------------------
    oof_dict: dict[str, np.ndarray] = {}
    if OOF_PATH.exists():
        oof_df = pd.read_csv(OOF_PATH)
        for col in oof_df.columns:
            if col != "y_true":
                oof_dict[col] = oof_df[col].values

    # -- Load pre-computed features --------------------------------------
    feat_names_2d = ["ecfp4", "fcfp4", "rdkit2d", "mordred2d",
                     "ecfp4_rdkit", "ecfp4_mordred"]
    feat_names_3d = ["ecfp4_rdkit_3dqsar", "ecfp4_mordred3d", "mordred3d", "rdkit3d_pharm"]

    feat_cache: dict[str, np.ndarray] = {}
    for name in feat_names_2d + feat_names_3d:
        p = FEATURE_DIR / f"{name}_train.npy"
        if p.exists():
            feat_cache[name] = np.load(p)
            print(f"  Loaded {name}: {feat_cache[name].shape}")

    # -- Extra fingerprints: load from disk or compute -------------------
    extra_fps_path = FEATURE_DIR / "extra_fps_train.npy"
    if extra_fps_path.exists():
        feat_cache["extra_fps"] = np.load(extra_fps_path)
        print(f"  Loaded extra_fps: {feat_cache['extra_fps'].shape}")
    else:
        print("\nComputing Avalon + AtomPair + RDKit-path fingerprints ...")
        extra_tr = compute_extra_fps(smiles_list)
        test_smiles_df = pd.read_csv(
            DATA_DIR / "test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
            else DATA_DIR / "test_raw.csv"
        )
        extra_te = compute_extra_fps(test_smiles_df["SMILES"].tolist())
        np.save(FEATURE_DIR / "extra_fps_train.npy", extra_tr)
        np.save(FEATURE_DIR / "extra_fps_test.npy",  extra_te)
        feat_cache["extra_fps"] = extra_tr
        print(f"  extra_fps: {extra_tr.shape}")

    # -- All-combined feature set: load from disk or build --------------
    all_combined_path = FEATURE_DIR / "all_combined_train.npy"
    if all_combined_path.exists():
        feat_cache["all_combined"] = np.load(all_combined_path)
        print(f"  Loaded all_combined: {feat_cache['all_combined'].shape}")
    elif "ecfp4" in feat_cache and "rdkit2d" in feat_cache and "extra_fps" in feat_cache:
        all_tr = np.hstack([feat_cache["ecfp4"], feat_cache["rdkit2d"], feat_cache["extra_fps"]])
        all_te = np.hstack([
            np.load(FEATURE_DIR / "ecfp4_test.npy"),
            np.load(FEATURE_DIR / "rdkit2d_test.npy"),
            np.load(FEATURE_DIR / "extra_fps_test.npy"),
        ])
        np.save(FEATURE_DIR / "all_combined_train.npy", all_tr)
        np.save(FEATURE_DIR / "all_combined_test.npy",  all_te)
        feat_cache["all_combined"] = all_tr
        print(f"  all_combined: {all_tr.shape}")

    # -- Pre-compute Butina splits ONCE (expensive: O(n²) distance matrix) ---
    print("Pre-computing Butina CV splits (runs once, reused by all models) ...")
    splits = repeated_butina_cv(smiles_list, k=N_FOLDS, n_repeats=N_REPEATS, seeds=SEEDS)
    print(f"  {len(splits)} (repeat, fold) pairs ready\n")

    # -- Run CV ----------------------------------------------------------
    # Classical / gradient boosting models
    models = make_models(feat_cache)
    for model_name, (model, feat_name) in models.items():
        if model_name in completed:
            print(f"  SKIP {model_name} (already done)")
            continue
        if feat_name not in feat_cache:
            print(f"  SKIP {model_name}: {feat_name} not available")
            continue
        X = feat_cache[feat_name]
        print(f"\n--- {model_name}  (X={X.shape}) ---")
        try:
            records, oof = run_cv_for_model(model_name, model, X, y, splits)
            save_records_incremental(records)
            oof_dict[model_name] = oof
            save_oof({model_name: oof}, y)
            completed.add(model_name)
        except Exception as e:
            print(f"  ERROR in {model_name}: {e}")
            import traceback; traceback.print_exc()

    # TabPFN
    if "TabPFN_ecfp4_rdkit" not in completed:
        tabpfn = tabpfn_model()
        if tabpfn is not None and "ecfp4_rdkit" in feat_cache:
            X_tab = feat_cache["ecfp4_rdkit"]
            print(f"\n--- TabPFN_ecfp4_rdkit  (X={X_tab.shape}) ---")
            try:
                records, oof = run_cv_for_model(
                    "TabPFN_ecfp4_rdkit", tabpfn, X_tab, y, splits
                )
                save_records_incremental(records)
                oof_dict["TabPFN_ecfp4_rdkit"] = oof
                save_oof({"TabPFN_ecfp4_rdkit": oof}, y)
                completed.add("TabPFN_ecfp4_rdkit")
            except Exception as e:
                print(f"  TabPFN failed: {e}")
                import traceback; traceback.print_exc()
    else:
        print("  SKIP TabPFN_ecfp4_rdkit (already done)")

    # ChemProp + Chemeleon
    if "ChemProp_Chemeleon" not in completed:
        try:
            from cheminf_utils.chemprop_helpers import ChemPropChemeleonWrapper  # noqa: F401
            print("\n--- ChemProp_Chemeleon (SMILES + pretrained MP encoder) ---")
            records, oof = run_chemeleon_cv(
                "ChemProp_Chemeleon", y, smiles_list, splits, num_epochs=30
            )
            if records:
                save_records_incremental(records)
                oof_dict["ChemProp_Chemeleon"] = oof
                save_oof({"ChemProp_Chemeleon": oof}, y)
                completed.add("ChemProp_Chemeleon")
        except ImportError:
            print("  ChemProp_Chemeleon skipped: cheminf_utils not available")
        except Exception as e:
            print(f"  ChemProp_Chemeleon failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  SKIP ChemProp_Chemeleon (already done)")

    # -- Load full results for summary -----------------------------------
    if not CV_RESULTS_PATH.exists():
        print("ERROR: No results recorded.")
        sys.exit(1)

    cv_df = pd.read_csv(CV_RESULTS_PATH)
    print(f"\nTotal fold records in cv_results.csv: {len(cv_df)}")

    # -- Summary ---------------------------------------------------------
    summary = (
        cv_df.groupby("model")[["RMSE", "MAE", "R2", "Spearman"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("RMSE_mean")
    print("\n=== CV Summary (sorted by mean RMSE) ===")
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "cv_summary.csv")

    # -- Plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_models = len(cv_df["model"].unique())
        fig, ax = plt.subplots(figsize=(max(10, n_models * 0.9), 5))
        order = summary.index.tolist()
        palette = sns.color_palette("Blues_r", len(order))
        sns.boxplot(data=cv_df, x="model", y="RMSE", order=order, ax=ax, palette=palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        ax.set_title(f"3×{N_FOLDS}-fold Butina CV — RMSE by model (lower = better)")
        best_rmse = summary["RMSE_mean"].min()
        ax.axhline(best_rmse, color="red", linestyle="--", alpha=0.5,
                   label=f"best mean={best_rmse:.4f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "cv_summary.png", dpi=150)
        plt.close()
        print("Saved cv_summary.png")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    best = summary.index[0]
    print(f"\nBest model: {best}  (mean RMSE={summary.loc[best,'RMSE_mean']:.4f}  "
          f"Spearman={summary.loc[best,'Spearman_mean']:.3f})")


if __name__ == "__main__":
    main()
