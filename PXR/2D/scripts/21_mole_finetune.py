"""
MolE fine-tuning for PXR pEC50 regression.

MolE: Recursion Pharma foundation model (Nature Comms 2024).
Architecture: DeBERTa-style disentangled attention on molecular graphs.
Pretrained checkpoint: MolE_GuacaMol_27113.ckpt from CodeOcean capsule 2105466.

Workflow:
  1. Download pretrained checkpoint if not cached (~995 MB)
  2. Butina 3×5-fold OOF (15 models) for Ridge ensemble integration
  3. Full training (3 seeds × 80/15 split) + test predictions

Requires mole_env (separate from cheminf_utils / unimol_env):
  Create env (one-time):
    wsl -e bash -c "
      /home/jeremy/mambaforge/bin/mamba create -n mole_env python=3.10 -y
      git clone https://github.com/recursionpharma/mole_public.git ~/mole_public
      /home/jeremy/mambaforge/bin/conda run -n mole_env pip install \
        -r ~/mole_public/requirements/main_3.10_gpu.txt
      /home/jeremy/mambaforge/bin/conda run -n mole_env pip install -e ~/mole_public
    "

  Run script:
    wsl -e bash -c "/home/jeremy/mambaforge/bin/conda run -n mole_env python \
      /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/21_mole_finetune.py"

Saves:
  models/mole_pretrained/MolE_GuacaMol_27113.ckpt  — downloaded pretrained weights
  models/mole_s{seed}_f{fold}/                      — OOF fold checkpoints
  models/mole_full_s{seed}/                          — full-train checkpoints
  results/mole_oof_s{seed}_f{fold}.npy              — per-fold OOF cache
  results/mole_oof_preds.npy                         — averaged OOF (4083,)
  results/submission_pEC50_13.csv                    — 513 test predictions
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from utils import butina_kfold

SEEDS      = [0, 1, 2]
KFOLD      = 5
EPOCHS     = 50
LR         = 1e-5   # transfer learning rate from pretrained
DROPOUT    = 0.1
BATCH_SIZE = 8      # 4GB GPU limit
TAG        = f"mole_e{EPOCHS}"  # output prefix; preserves prior runs at different EPOCHS

PRETRAINED_URL  = (
    "https://files.codeocean.com/files/verified/"
    "d5a961a4-d159-4218-8f70-53f515052de0_v1.0/data/"
    "MolE_GuacaMol_27113.ckpt?download"
)
PRETRAINED_CKPT = MODELS_DIR / "mole_pretrained" / "MolE_GuacaMol_27113.ckpt"
MOLE_TRAIN_BIN  = "/home/jeremy/mambaforge/envs/mole_env/bin/mole_train"


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
    y = np.load(y_path) if y_path.exists() else train_df["pEC50"].values

    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    return train_df, test_df, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_pretrained():
    """Download GuacaMol pretrained MolE checkpoint (~995 MB) if not cached."""
    PRETRAINED_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if PRETRAINED_CKPT.exists():
        print(f"  Pretrained checkpoint found: {PRETRAINED_CKPT}")
        return
    print(f"  Downloading MolE pretrained checkpoint (~995 MB) ...")
    urlretrieve(PRETRAINED_URL, str(PRETRAINED_CKPT))
    print(f"  Saved → {PRETRAINED_CKPT}")


def write_mole_csv(df: pd.DataFrame, path: Path):
    """Write CSV with lowercase 'smiles' column as required by MolE."""
    out = df[["SMILES", "pEC50"]].copy()
    out.columns = ["smiles", "pEC50"]
    out.to_csv(path, index=False)


def train_mole(train_df: pd.DataFrame, val_df: pd.DataFrame,
               out_dir: Path) -> Path:
    """Run mole_train CLI for one fold. Returns path to saved checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    val_csv   = out_dir / "val.csv"
    write_mole_csv(train_df, train_csv)
    write_mole_csv(val_df,   val_csv)

    cmd = [
        MOLE_TRAIN_BIN, "model=finetune",
        f"data_file={train_csv}",
        f"checkpoint_path={PRETRAINED_CKPT}",
        f"dropout={DROPOUT}",
        f"lr={LR}",
        "task=regression",
        "num_tasks=1",
        f"model.hyperparameters.datamodule.validation_data={val_csv}",
        f"model.hyperparameters.datamodule.batch_size={BATCH_SIZE}",
        "model.hyperparameters.datamodule.num_workers=0",
        f"model.data.trainer.max_epochs={EPOCHS}",
        f"hydra.run.dir={out_dir}",
        "logger.log_model=false",  # don't upload/consume checkpoint via wandb
    ]
    print(f"    Running mole_train → {out_dir.name} ...")
    # cwd=out_dir so that dirpath='./' in ModelCheckpoint resolves to out_dir
    subprocess.run(cmd, check=True, cwd=str(out_dir))

    ckpts = sorted(out_dir.glob("lightning_checkpoint-*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoint found in {out_dir}")
    return ckpts[0]   # save_top_k=1 → only one checkpoint


def predict_mole(smiles_list: list[str], ckpt_path: Path) -> np.ndarray:
    """Predict pEC50 from a fine-tuned MolE checkpoint."""
    from mole import mole_predict
    preds = mole_predict.predict_ckpt(
        smiles=smiles_list,
        task="regression",
        num_tasks=1,
        pretrained_model=str(ckpt_path),
        batch_size=BATCH_SIZE,
        num_workers=0,
        accelerator="gpu",
    )
    return np.array(preds).flatten()


# ---------------------------------------------------------------------------
# OOF loop (Butina 3×5-fold)
# ---------------------------------------------------------------------------

def compute_oof(train_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    all_seed_oof = []

    for seed in SEEDS:
        smiles = train_df["SMILES"].tolist()
        folds  = butina_kfold(smiles, k=KFOLD, seed=seed)
        oof    = np.full(len(train_df), np.nan)

        seed_cached = True
        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            cache = RESULTS_DIR / f"{TAG}_oof_s{seed}_f{fold_i}.npy"
            if not cache.exists():
                seed_cached = False
                break

        if seed_cached:
            print(f"  SKIP seed {seed} (all folds cached)")
            fold_preds = []
            for fold_i, (tr_idx, va_idx) in enumerate(folds):
                cache = RESULTS_DIR / f"{TAG}_oof_s{seed}_f{fold_i}.npy"
                fold_preds_arr = np.load(cache)
                oof[va_idx] = fold_preds_arr
            all_seed_oof.append(oof)
            oof_rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])))
            oof_rho  = float(spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)]).statistic)
            print(f"  Seed {seed}  OOF RMSE={oof_rmse:.5f}  Spearman={oof_rho:.4f}")
            continue

        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            cache = RESULTS_DIR / f"{TAG}_oof_s{seed}_f{fold_i}.npy"
            if cache.exists():
                oof[va_idx] = np.load(cache)
                print(f"  Seed {seed}  fold {fold_i+1}/{KFOLD} SKIP (cached)")
                continue

            print(f"  Seed {seed}  fold {fold_i+1}/{KFOLD} ...")
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            va_df = train_df.iloc[va_idx].reset_index(drop=True)

            out_dir = MODELS_DIR / f"{TAG}_s{seed}_f{fold_i}"
            ckpt    = train_mole(tr_df, va_df, out_dir)
            preds   = predict_mole(va_df["SMILES"].tolist(), ckpt)
            oof[va_idx] = preds

            fold_rmse = float(np.sqrt(mean_squared_error(y[va_idx], preds)))
            print(f"    fold RMSE={fold_rmse:.5f}")
            np.save(cache, preds)

        oof_rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])))
        oof_rho  = float(spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)]).statistic)
        print(f"  Seed {seed}  OOF RMSE={oof_rmse:.5f}  Spearman={oof_rho:.4f}")
        all_seed_oof.append(oof)

    mean_oof = np.nanmean(all_seed_oof, axis=0)
    oof_rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    oof_rho  = float(spearmanr(y, mean_oof).statistic)
    print(f"\nMean OOF ({len(SEEDS)} seeds × {KFOLD} folds): "
          f"RMSE={oof_rmse:.5f}  Spearman={oof_rho:.4f}")
    return mean_oof


# ---------------------------------------------------------------------------
# Full training → test predictions
# ---------------------------------------------------------------------------

def compute_test_preds(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    test_cache = RESULTS_DIR / f"{TAG}_test_preds.npy"
    if test_cache.exists():
        print(f"  SKIP full training (cached)")
        return np.load(test_cache)

    seed_preds = []
    for seed in SEEDS:
        out_dir = MODELS_DIR / f"{TAG}_full_s{seed}"
        ckpt_cache = out_dir / "lightning_checkpoint-*.ckpt"
        existing = sorted(out_dir.glob("lightning_checkpoint-*.ckpt")) if out_dir.exists() else []

        if existing:
            print(f"  SKIP full seed {seed} (model exists)")
            ckpt = existing[0]
        else:
            print(f"  Full train seed {seed} ...")
            tr_df, va_df = train_test_split(train_df, test_size=0.15, random_state=seed)
            ckpt = train_mole(tr_df.reset_index(drop=True),
                              va_df.reset_index(drop=True), out_dir)

        preds = predict_mole(test_df["SMILES"].tolist(), ckpt)
        seed_preds.append(preds)
        print(f"    mean={preds.mean():.3f}  std={preds.std():.3f}")

    mean_test = np.mean(seed_preds, axis=0)
    np.save(test_cache, mean_test)
    return mean_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== MolE fine-tuning (21_mole_finetune.py) ===\n")

    download_pretrained()
    train_df, test_df, y = load_data()

    print("\n[1/2] Computing Butina OOF ...")
    mean_oof = compute_oof(train_df, y)
    oof_path = RESULTS_DIR / f"{TAG}_oof_preds.npy"
    np.save(oof_path, mean_oof)
    print(f"Saved OOF → {oof_path}")

    print("\n[2/2] Full training → test predictions ...")
    mean_test = compute_test_preds(train_df, test_df)

    final_preds = np.clip(mean_test, 1.0, 9.0)
    print(f"\nTest: range=[{final_preds.min():.3f}, {final_preds.max():.3f}]  "
          f"mean={final_preds.mean():.3f}  std={final_preds.std():.3f}")

    submission = test_df[["SMILES", "Molecule Name"]].copy()
    submission["pEC50"] = final_preds
    out_path = RESULTS_DIR / "submission_pEC50_13.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(submission)} rows)")

    assert submission["pEC50"].isna().sum() == 0
    assert submission.columns.tolist() == ["SMILES", "Molecule Name", "pEC50"]
    print("All checks passed.")

    oof_rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    print(f"\nFinal OOF RMSE: {oof_rmse:.5f}")
    print(f"Baseline ChemProp (single-task): 0.5977")
    if oof_rmse < 0.5977:
        print(">>> MolE improves over single-task ChemProp — add to Ridge ensemble.")
    else:
        print(">>> MolE does not improve at OOF level — check blind test anyway.")


if __name__ == "__main__":
    main()
