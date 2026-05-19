"""
UniMol fine-tuning — 50 epochs (Track A improvement over sub8's 30 epochs).

Identical to 16_unimol_finetune.py but with:
  EPOCHS = 50  (was 30)
  early_stopping = 15  (was 10)
  New save paths: models/unimol_ft_s{seed}_e50/
  New cache files: results/unimol_ft_e50_s{seed}_oof.npy / _test.npy

Run with unimol_env (NOT cheminf_utils):
  wsl -e bash -c "/home/jeremy/mambaforge/bin/conda run -n unimol_env python \
    /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/19_unimol_e50.py"

Saves:
  models/unimol_ft_s{seed}_e50/    — kfold=5 checkpoints
  results/unimol_ft_e50_oof_preds.npy  — averaged OOF (4083,)
  results/submission_pEC50_11.csv       — 513 test predictions
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEEDS      = [0, 1, 2]
KFOLD      = 5
EPOCHS     = 50        # was 30 in sub8
LR         = 1e-4
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    train_path = (DATA_DIR / "train_final.csv"
                  if (DATA_DIR / "train_final.csv").exists()
                  else DATA_DIR / "train_curated.csv")
    test_path  = (DATA_DIR / "test_curated.csv"
                  if (DATA_DIR / "test_curated.csv").exists()
                  else DATA_DIR / "test_raw.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_path = ROOT / "features" / "y_train.npy"
    if y_path.exists():
        y = np.load(y_path)
    else:
        y = train_df["pEC50"].values

    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    print(f"pEC50 range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.3f}")
    return train_df, test_df, y


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

def run_finetune_seed(seed: int, smiles_train: list[str], y: np.ndarray,
                      smiles_test: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import random, torch
    from unimol_tools import MolTrain, MolPredict

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir   = str(MODELS_DIR / f"unimol_ft_s{seed}_e{EPOCHS}")
    oof_cache  = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_s{seed}_oof.npy"
    test_cache = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_s{seed}_test.npy"

    print(f"\n  === Seed {seed}  epochs={EPOCHS}  save_path={save_dir} ===")

    if oof_cache.exists() and test_cache.exists():
        print(f"  SKIP seed {seed} (cached)")
        return np.load(oof_cache), np.load(test_cache)

    train_data = pd.DataFrame({"SMILES": smiles_train, "pEC50": y.tolist()})

    clf = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=EPOCHS,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        early_stopping=15,
        kfold=KFOLD,
        smiles_col="SMILES",
        target_cols=["pEC50"],
        target_normalize="standard",
        remove_hs=False,
        save_path=save_dir,
        use_gpu=True,
        conf_cache_level=2,
        params={"seed": seed},
    )

    clf.fit(train_data)

    oof = np.array(clf.cv_pred).flatten()
    print(f"  OOF shape: {oof.shape}  mean={oof.mean():.3f}  std={oof.std():.3f}")

    test_data = pd.DataFrame({
        "SMILES": smiles_test,
        "pEC50":  [-1.0] * len(smiles_test),
    })
    predictor = MolPredict(load_model=save_dir)
    pred_out  = predictor.predict(test_data)

    if isinstance(pred_out, pd.DataFrame):
        pred_col   = [c for c in pred_out.columns if c.startswith("predict_")][0]
        test_preds = pred_out[pred_col].values.flatten()
    else:
        test_preds = np.array(pred_out).flatten()

    print(f"  Test preds: mean={test_preds.mean():.3f}  std={test_preds.std():.3f}")

    np.save(str(oof_cache), oof)
    np.save(str(test_cache), test_preds)
    return oof, test_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== UniMol fine-tuning — {EPOCHS} epochs (19_unimol_e50.py) ===\n")

    train_df, test_df, y = load_data()
    smiles_train = train_df["SMILES"].tolist()
    smiles_test  = test_df["SMILES"].tolist()

    all_oof_preds  = []
    all_test_preds = []

    for seed in SEEDS:
        oof, test_preds = run_finetune_seed(seed, smiles_train, y, smiles_test)
        all_oof_preds.append(oof)
        all_test_preds.append(test_preds)

        oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
        oof_rho  = float(spearmanr(y, oof).statistic)
        print(f"  Seed {seed}  OOF RMSE={oof_rmse:.5f}  Spearman={oof_rho:.4f}")

    mean_oof        = np.mean(all_oof_preds, axis=0)
    mean_test_preds = np.mean(all_test_preds, axis=0)

    overall_oof_rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    overall_oof_rho  = float(spearmanr(y, mean_oof).statistic)
    print(f"\nEnsembled ({len(SEEDS)} seeds × {KFOLD} folds):")
    print(f"  OOF RMSE  = {overall_oof_rmse:.5f}  (sub8 was 0.57791)")
    print(f"  Spearman  = {overall_oof_rho:.4f}")

    oof_path = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_oof_preds.npy"
    np.save(str(oof_path), mean_oof)
    print(f"\nSaved OOF → {oof_path}")

    final_preds = np.clip(mean_test_preds, 1.0, 9.0)
    print(f"\nTest predictions: range=[{final_preds.min():.3f}, {final_preds.max():.3f}]"
          f"  mean={final_preds.mean():.3f}  std={final_preds.std():.3f}")

    submission = test_df[["SMILES", "Molecule Name"]].copy()
    submission["pEC50"] = final_preds
    out_path = RESULTS_DIR / "submission_pEC50_11.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(submission)} rows)")

    assert submission["pEC50"].isna().sum() == 0
    assert submission.columns.tolist() == ["SMILES", "Molecule Name", "pEC50"]
    print("All checks passed.")

    # Guidance for Ridge integration
    if overall_oof_rmse < 0.57791:
        print(f"\n>>> Improved vs sub8 (0.57791) — replace UniMol_FT in Ridge with e50 version.")
    else:
        print(f"\n>>> No improvement vs sub8 (0.57791) at OOF level — 30 epochs was sufficient.")
    print("    NOTE: OOF is random k-fold (optimistic); compare blind test for true assessment.")


if __name__ == "__main__":
    main()
