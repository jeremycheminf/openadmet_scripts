"""Aggregate cached UniMol e50 seed OOF/test predictions and save sub11."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EPOCHS      = 50
SEEDS       = [0, 1, 2]

train_path = DATA_DIR / "train_final.csv" if (DATA_DIR / "train_final.csv").exists() else DATA_DIR / "train_curated.csv"
test_path  = DATA_DIR / "test_curated.csv" if (DATA_DIR / "test_curated.csv").exists() else DATA_DIR / "test_raw.csv"
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
y_path   = ROOT / "features" / "y_train.npy"
y = np.load(y_path) if y_path.exists() else train_df["pEC50"].values

all_oof, all_test = [], []
for seed in SEEDS:
    oof_f  = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_s{seed}_oof.npy"
    test_f = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_s{seed}_test.npy"
    oof  = np.load(oof_f)
    test = np.load(test_f)
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    rho  = float(spearmanr(y, oof).statistic)
    print(f"  Seed {seed}  OOF RMSE={rmse:.5f}  Spearman={rho:.4f}")
    all_oof.append(oof)
    all_test.append(test)

mean_oof  = np.mean(all_oof, axis=0)
mean_test = np.mean(all_test, axis=0)

rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
rho  = float(spearmanr(y, mean_oof).statistic)
print(f"\nEnsembled ({len(SEEDS)} seeds × 5 folds):")
print(f"  OOF RMSE  = {rmse:.5f}  (sub8 was 0.57791)")
print(f"  Spearman  = {rho:.4f}")

oof_path = RESULTS_DIR / f"unimol_ft_e{EPOCHS}_oof_preds.npy"
np.save(str(oof_path), mean_oof)
print(f"Saved OOF → {oof_path}")

final_preds = np.clip(mean_test, 1.0, 9.0)
smiles_col  = "SMILES" if "SMILES" in test_df.columns else test_df.columns[0]
name_col    = next((c for c in ["Molecule Name", "OCNT_ID", "ID"] if c in test_df.columns), None)
if name_col:
    submission = test_df[[smiles_col, name_col]].copy()
else:
    submission = test_df[[smiles_col]].copy()
submission["pEC50"] = final_preds

out_path = RESULTS_DIR / "submission_pEC50_11.csv"
submission.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(submission)} rows)")
assert submission["pEC50"].isna().sum() == 0
print("All checks passed.")
