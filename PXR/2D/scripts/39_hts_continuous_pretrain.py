"""HTS continuous pretraining — log2_fc_estimate regression with concentration as descriptor.

Goal: replace our binary HTS encoder (chemeleon_hts.ckpt, trained on active>=4 binary
labels) with a continuous-regression encoder. Reilly's repo achieves CV RAE 0.4824
solo with this style of encoder + TabICL — vs our 0.500 with the binary encoder.
End-to-end fine-tuning destroys foundation models at ~4K samples (Reilly, sub24);
the operating pattern is frozen encoder + foundation regressor, so the encoder
must be pretrained as richly as possible.

Data: data/single_concentration_train.csv (21,004 rows; ~3K unique compounds at
multiple concentrations). Each row is one (compound, concentration) dose-response
point. We use concentration_M as a scalar input descriptor (log10) so the encoder
learns concentration-aware dose-response, not just binary hit/non-hit.

Strategy:
  - Filter: drop rows whose compound (by inchikey) appears in test_curated.csv
  - Warm-start: load BondMessagePassing from chemeleon_hts.ckpt (already
    CheMeleon → binary HTS); then re-train the full MPNN with continuous target.
  - Tikhonov sample weights: 1/(log2_fc_stderr² + 0.3²), clipped at 5/95 quantile.
  - 10% held-out validation for early stopping (patience 5 on val_loss).
  - Single regression task: log2_fc_estimate.
  - Save full MPNN to models/chemeleon_hts_cont.ckpt — downstream scripts
    (40_tabicl_hts_cont.py) load it and extract message_passing.

Run in cheminf_utils env (GPU needed, ~4-6h on RTX-class):
  wsl -e bash -c "bash /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/logs/_run_hts_cont.sh"

Outputs:
  models/chemeleon_hts_cont.ckpt   — full MPNN checkpoint
  features/chemeleon_hts_cont_train.npy  (4083, 2048) embeddings
  features/chemeleon_hts_cont_test.npy   (513, 2048)  embeddings
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi as _inchi
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import DATA_DIR, FEATURE_DIR  # type: ignore

warnings.filterwarnings("ignore")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
FEATURE_DIR.mkdir(exist_ok=True)

WARM_START_CKPT = MODELS_DIR / "chemeleon_hts.ckpt"
OUT_CKPT        = MODELS_DIR / "chemeleon_hts_cont.ckpt"

EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-4
PATIENCE   = 5
SIGMA      = 0.3   # Tikhonov regularizer for sample weights
SEED       = 0


def smiles_to_inchikey(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return _inchi.MolToInchiKey(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_hts_data():
    print("Loading HTS single-concentration data ...")
    df = pd.read_csv(DATA_DIR / "single_concentration_train.csv")
    print(f"  Raw HTS rows: {len(df)}")

    # Drop NaN target / SMILES
    df = df.dropna(subset=["SMILES", "log2_fc_estimate", "concentration_M"])
    print(f"  After NaN drop: {len(df)}")

    # Filter out compounds present in test set (avoid leakage)
    test_df = pd.read_csv(DATA_DIR / ("test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
                                       else "test_raw.csv"))
    test_keys = {smiles_to_inchikey(s) for s in test_df["SMILES"]}
    test_keys.discard(None)
    df["inchikey"] = df["SMILES"].map(smiles_to_inchikey)
    pre = len(df)
    df = df[~df["inchikey"].isin(test_keys)].dropna(subset=["inchikey"])
    print(f"  After test-leakage filter: {len(df)} (-{pre - len(df)})")

    # Tikhonov sample weights
    se = df["log2_fc_stderr"].fillna(df["log2_fc_stderr"].median())
    w  = 1.0 / (se**2 + SIGMA**2)
    q05, q95 = np.quantile(w, [0.05, 0.95])
    df["sample_weight"] = np.clip(w.values, q05, q95)

    # log10(concentration) descriptor — handle non-positive defensively
    conc = df["concentration_M"].clip(lower=1e-12)
    df["log_conc"] = np.log10(conc)
    print(f"  log_conc range: [{df['log_conc'].min():.2f}, {df['log_conc'].max():.2f}]")
    print(f"  log2_fc range:  [{df['log2_fc_estimate'].min():.2f}, {df['log2_fc_estimate'].max():.2f}]")
    print(f"  weight range:   [{df['sample_weight'].min():.2f}, {df['sample_weight'].max():.2f}]")

    return df.reset_index(drop=True)


def build_datapoints(df: pd.DataFrame):
    """One MoleculeDatapoint per row, with log_conc as x_d extra descriptor."""
    from chemprop import data
    pts = []
    for _, row in df.iterrows():
        x_d = np.array([row["log_conc"]], dtype=np.float32)
        pts.append(data.MoleculeDatapoint.from_smi(
            row["SMILES"],
            [float(row["log2_fc_estimate"])],
            x_d=x_d,
            weight=float(row["sample_weight"]),
        ))
    return pts


# ---------------------------------------------------------------------------
# Encoder warm-start
# ---------------------------------------------------------------------------

def load_warm_encoder():
    """Load message_passing from chemeleon_hts.ckpt for warm-start."""
    from chemprop import models
    if not WARM_START_CKPT.exists():
        raise FileNotFoundError(f"{WARM_START_CKPT} missing")
    full = models.MPNN.load_from_checkpoint(str(WARM_START_CKPT), map_location="cpu")
    return full.message_passing


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    import torch
    import lightning.pytorch as pl
    from chemprop import data, featurizers, nn, models
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    torch.manual_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    df = load_hts_data()
    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=SEED,
                                    stratify=pd.qcut(df["log2_fc_estimate"], q=10, duplicates="drop"))
    print(f"\nTrain rows: {len(tr_df)}  Val rows: {len(va_df)}")

    tr_pts = build_datapoints(tr_df)
    va_pts = build_datapoints(va_df)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    tr_dset = data.MoleculeDataset(tr_pts, featurizer)
    scalers = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(va_pts, featurizer)
    va_dset.normalize_targets(scalers)

    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH_SIZE, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    mp  = load_warm_encoder()
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + 1,   # +1 for log_conc x_d
        n_tasks=1,
        output_transform=nn.UnscaleTransform.from_standard_scaler(scalers),
        criterion=nn.metrics.MSE(),
        dropout=0.1,
    )
    model = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.RMSE()])
    model.hparams["lr"] = LR

    ckpt_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR / "hts_cont_tmp"),
        filename="best",
        monitor="val_loss", mode="min", save_top_k=1, save_last=False,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", min_delta=1e-3)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto", devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[ckpt_cb, es_cb],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, tr_loader, va_loader)

    # Copy best checkpoint to OUT_CKPT
    best = Path(ckpt_cb.best_model_path)
    if best.exists():
        import shutil
        shutil.copy(best, OUT_CKPT)
        print(f"\nSaved best ckpt → {OUT_CKPT}  (val_loss={ckpt_cb.best_model_score:.4f})")
    else:
        trainer.save_checkpoint(str(OUT_CKPT))
        print(f"\nSaved final ckpt → {OUT_CKPT}")

    return OUT_CKPT


# ---------------------------------------------------------------------------
# Extract embeddings for the challenge train + test sets
# ---------------------------------------------------------------------------

def extract_embeddings(ckpt_path: Path):
    import torch
    from chemprop import data, featurizers, models

    print("\nExtracting embeddings for challenge train + test ...")
    train_df = pd.read_csv(DATA_DIR / ("train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                                       else "train_curated.csv"))
    test_df  = pd.read_csv(DATA_DIR / ("test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
                                       else "test_raw.csv"))
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

    full = models.MPNN.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    mp = full.message_passing
    agg = full.agg
    mp.eval(); agg.eval()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def embed(smiles_list):
        # x_d=0 for downstream embedding extraction (we want pure structure embedding)
        pts = [data.MoleculeDatapoint.from_smi(s, [0.0],
                                               x_d=np.array([0.0], dtype=np.float32))
               for s in smiles_list]
        dset = data.MoleculeDataset(pts, featurizer)
        loader = data.build_dataloader(dset, batch_size=BATCH_SIZE,
                                       num_workers=0, shuffle=False)
        out = []
        with torch.inference_mode():
            for batch in loader:
                bmg = batch.bmg
                H = mp(bmg)
                emb = agg(H, bmg.batch)
                out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0)

    X_tr = embed(train_df["SMILES"].tolist())
    X_te = embed(test_df["SMILES"].tolist())
    np.save(FEATURE_DIR / "chemeleon_hts_cont_train.npy", X_tr)
    np.save(FEATURE_DIR / "chemeleon_hts_cont_test.npy",  X_te)
    print(f"  Saved: features/chemeleon_hts_cont_{{train,test}}.npy  "
          f"shapes {X_tr.shape} {X_te.shape}")


def main():
    print("=== 39_hts_continuous_pretrain.py ===\n")
    if OUT_CKPT.exists():
        print(f"  {OUT_CKPT} exists — skipping training, extracting embeddings only")
    else:
        train()
    extract_embeddings(OUT_CKPT)


if __name__ == "__main__":
    main()
