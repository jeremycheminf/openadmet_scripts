"""
CheMeleon HTS pretraining + frozen embedding extraction.

Stage 1: Fine-tune the pretrained Chemeleon D-MPNN backbone on the competition's
21K single-concentration HTS data (log2_fc_estimate), with log10(concentration_M)
as an extra molecular descriptor. 30 epochs, LR=5e-5.

Stage 2: Freeze the HTS-tuned encoder and extract 2048-dim embeddings for all
train+test compounds. These embeddings feed into TabICL/TabPFN/LightGBM as a
base learner (Δ −0.023 CV in top competitor pipeline).

CRITICAL: Use FROZEN embeddings, NOT end-to-end fine-tune on pEC50.
End-to-end fine-tune causes negative transfer for ~4K samples.

Run in cheminf_utils env:
  wsl -e bash -c "/home/jeremy/mambaforge/bin/conda run -n cheminf_utils python \
    /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/23_chemeleon_hts_pretrain.py"

Saves:
  models/chemeleon_hts.ckpt            — HTS fine-tuned backbone checkpoint
  features/chemeleon_hts_train.npy     — (n_train, 2048) float32 embeddings
  features/chemeleon_hts_test.npy      — (n_test,  2048) float32 embeddings
  results/chemeleon_hts_oof_preds.npy  — OOF pEC50 from LightGBM on HTS embeddings
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
FEATURE_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
FEATURE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from utils import butina_kfold

HTS_EPOCHS  = 30
HTS_LR      = 5e-5
BATCH_SIZE  = 64
SEEDS       = [0, 1, 2]
KFOLD       = 5

_CHEMELEON_SEARCH = [
    Path("chemeleon_mp.pt"),
    Path.home() / "chemeleon_mp.pt",
    Path.home() / ".chemprop" / "chemeleon_mp.pt",
    ROOT.parent.parent.parent / "chemeleon_mp.pt",
]
_CHEMELEON_CACHE = Path.home() / ".chemprop" / "chemeleon_mp.pt"


# ---------------------------------------------------------------------------
# Data
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
    y_path   = ROOT / "features" / "y_train.npy"
    y = np.load(y_path) if y_path.exists() else train_df["pEC50"].values
    return train_df, test_df, y


def load_hts_data() -> pd.DataFrame:
    hts_path = DATA_DIR / "single_concentration_train.csv"
    hts = pd.read_csv(hts_path)
    hts = hts.dropna(subset=["SMILES", "log2_fc_estimate", "concentration_M"])
    hts["log_conc"] = np.log10(hts["concentration_M"].values)
    # Z-score both inputs for stable training
    hts["log_conc_z"]   = (hts["log_conc"]         - hts["log_conc"].mean())         / hts["log_conc"].std()
    hts["log2_fc_z"]    = (hts["log2_fc_estimate"]  - hts["log2_fc_estimate"].mean()) / hts["log2_fc_estimate"].std()
    print(f"  HTS data: {len(hts)} rows  "
          f"log2_fc range=[{hts['log2_fc_estimate'].min():.2f}, {hts['log2_fc_estimate'].max():.2f}]")
    return hts


def get_chemeleon_path() -> str:
    for p in _CHEMELEON_SEARCH:
        if p.exists():
            return str(p)
    _CHEMELEON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading chemeleon_mp.pt → {_CHEMELEON_CACHE}")
    urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                str(_CHEMELEON_CACHE))
    return str(_CHEMELEON_CACHE)


# ---------------------------------------------------------------------------
# HTS pretraining (Stage 1)
# ---------------------------------------------------------------------------

def pretrain_hts(hts_df: pd.DataFrame, chemeleon_path: str) -> str:
    """Fine-tune Chemeleon on HTS log2_fc data. Returns checkpoint path."""
    ckpt_path = str(MODELS_DIR / "chemeleon_hts.ckpt")
    if Path(ckpt_path).exists():
        print(f"  SKIP HTS pretraining (checkpoint exists: {ckpt_path})")
        return ckpt_path

    import torch
    import lightning.pytorch as pl
    from chemprop import data, featurizers, nn, models
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    # Extra feature: z-scored log10(concentration)
    hts_df = hts_df.reset_index(drop=True)
    log_conc_z = hts_df["log_conc_z"].values.astype(np.float32)
    targets    = hts_df["log2_fc_z"].values.astype(np.float32)

    pts = []
    for idx in range(len(hts_df)):
        dp = data.MoleculeDatapoint.from_smi(
            hts_df.loc[idx, "SMILES"],
            [float(targets[idx])],
        )
        dp.x_d = np.array([log_conc_z[idx]], dtype=np.float32)
        pts.append(dp)

    from sklearn.model_selection import train_test_split as sk_split
    tr_pts, va_pts = sk_split(pts, test_size=0.1, random_state=42)

    featurizer  = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset  = data.MoleculeDataset(tr_pts, featurizer)
    scalers     = train_dset.normalize_targets()
    val_dset    = data.MoleculeDataset(va_pts, featurizer)
    val_dset.normalize_targets(scalers)

    train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = data.build_dataloader(val_dset,   batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    # Load Chemeleon backbone
    chemeleon_ckpt = torch.load(chemeleon_path, weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_ckpt["hyper_parameters"])
    mp.load_state_dict(chemeleon_ckpt["state_dict"])

    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + 1,   # +1 for log_conc extra feature
        n_tasks=1,
        output_transform=nn.UnscaleTransform.from_standard_scaler(scalers),
    )
    model = models.MPNN(mp, agg, ffn, batch_norm=False,
                        metrics=[nn.metrics.RMSE()])

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="chemeleon_hts",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=HTS_EPOCHS,
        callbacks=[checkpoint_cb, early_stop],
    )
    trainer.fit(model, train_loader, val_loader)

    saved = checkpoint_cb.best_model_path
    print(f"  HTS checkpoint saved → {saved}")
    return saved


# ---------------------------------------------------------------------------
# Embedding extraction (Stage 2)
# ---------------------------------------------------------------------------

def extract_embeddings(smiles_list: list[str], ckpt_path: str) -> np.ndarray:
    """Extract frozen 2048-dim D-MPNN embeddings (before FFN head)."""
    import torch
    import lightning.pytorch as pl
    from chemprop import data, featurizers, nn, models

    # Load model on CPU for embedding extraction (avoid BatchMolGraph device issues)
    model = models.MPNN.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()

    # x_d is only used in FFN — MP+agg embeddings don't need concentration
    pts = [data.MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles_list]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset       = data.MoleculeDataset(pts, featurizer)
    loader     = data.build_dataloader(dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    embeddings = []
    with torch.inference_mode():
        for batch in loader:
            bmg = batch[0]  # TrainingBatch is 7-tuple: (bmg, V_d, X_d, targets, weights, lt_mask, gt_mask)
            # MP + aggregation only — 2048-dim, before FFN/concentration concatenation
            h = model.message_passing(bmg)
            h = model.agg(h, bmg.batch)
            embeddings.append(h.numpy())

    return np.vstack(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# OOF LightGBM base learner on HTS embeddings (Butina 3×5-fold)
# ---------------------------------------------------------------------------

def compute_oof(X_train: np.ndarray, y: np.ndarray,
                smiles: list[str]) -> np.ndarray:
    import lightgbm as lgb

    LGB_PARAMS = dict(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        random_state=42,
        n_jobs=4,
    )

    all_seed_oof = []
    for seed in SEEDS:
        folds = butina_kfold(smiles, k=KFOLD, seed=seed)
        oof   = np.full(len(y), np.nan)
        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(X_train[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict(X_train[va_idx])
        rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])))
        rho  = float(spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)]).statistic)
        print(f"  Seed {seed}  OOF RMSE={rmse:.5f}  Spearman={rho:.4f}")
        all_seed_oof.append(oof)

    mean_oof = np.nanmean(all_seed_oof, axis=0)
    rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    rho  = float(spearmanr(y, mean_oof).statistic)
    print(f"\nMean OOF RMSE={rmse:.5f}  Spearman={rho:.4f}")
    return mean_oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== CheMeleon HTS pretraining + frozen embedding extraction ===\n")

    train_df, test_df, y = load_data()
    hts_df = load_hts_data()
    chemeleon_path = get_chemeleon_path()
    print(f"  Chemeleon: {chemeleon_path}\n")

    # Stage 1: HTS pretraining
    print("[1/3] HTS pretraining (30 epochs, LR=5e-5) ...")
    ckpt_path = pretrain_hts(hts_df, chemeleon_path)

    # Stage 2: Extract frozen embeddings
    emb_train_path = FEATURE_DIR / "chemeleon_hts_train.npy"
    emb_test_path  = FEATURE_DIR / "chemeleon_hts_test.npy"

    if emb_train_path.exists() and emb_test_path.exists():
        print("\n[2/3] SKIP embedding extraction (cached)")
        X_train_emb = np.load(emb_train_path)
        X_test_emb  = np.load(emb_test_path)
    else:
        print("\n[2/3] Extracting frozen embeddings ...")
        all_smiles  = train_df["SMILES"].tolist() + test_df["SMILES"].tolist()
        all_emb     = extract_embeddings(all_smiles, ckpt_path)
        n_train     = len(train_df)
        X_train_emb = all_emb[:n_train]
        X_test_emb  = all_emb[n_train:]
        np.save(emb_train_path, X_train_emb)
        np.save(emb_test_path,  X_test_emb)
        print(f"  Train embeddings: {X_train_emb.shape}")
        print(f"  Test  embeddings: {X_test_emb.shape}")

    # Stage 3: OOF LightGBM on frozen embeddings
    print("\n[3/3] Butina 3×5-fold OOF (LightGBM on HTS embeddings) ...")
    smiles = train_df["SMILES"].tolist()
    mean_oof = compute_oof(X_train_emb, y, smiles)

    oof_path = RESULTS_DIR / "chemeleon_hts_oof_preds.npy"
    np.save(oof_path, mean_oof)
    print(f"\nSaved OOF → {oof_path}")

    test_preds_path = RESULTS_DIR / "chemeleon_hts_test_preds.npy"
    import lightgbm as lgb
    final_model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=600, learning_rate=0.03,
        num_leaves=63, verbose=-1, random_state=42, n_jobs=4,
    )
    final_model.fit(X_train_emb, y)
    test_preds = final_model.predict(X_test_emb)
    np.save(test_preds_path, test_preds)
    print(f"Saved test preds → {test_preds_path}")

    print("\nDone. To integrate into Ridge ensemble:")
    print("  OOF:  results/chemeleon_hts_oof_preds.npy")
    print("  Test: results/chemeleon_hts_test_preds.npy")


if __name__ == "__main__":
    main()
