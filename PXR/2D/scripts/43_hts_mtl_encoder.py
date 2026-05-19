"""HTS multi-task pretraining: continuous log2_fc + soft-binary active signal.

Builds on 39_hts_continuous_pretrain.py — same data, warm start, x_d, weights —
but with 2 regression heads so the encoder learns both granular dose-response
AND threshold (hit / non-hit) structure:

  task 0: log2_fc_estimate (continuous, weight 1.0)
  task 1: p_active = sigmoid(2 * (log2_fc_estimate - 1.0))
          smooth soft-binary near pEC50≥4 threshold, weight 0.5

Both heads are regression with MSE; sigmoid-smoothed target avoids the
mixed reg+cls plumbing while still concentrating capacity around the
active/inactive decision boundary.

Outputs:
  models/chemeleon_hts_mtl.ckpt
  features/chemeleon_hts_mtl_train.npy (4083, 2048)
  features/chemeleon_hts_mtl_test.npy  (513,  2048)
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi as _inchi
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import DATA_DIR, FEATURE_DIR

warnings.filterwarnings("ignore")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

WARM_START_CKPT = MODELS_DIR / "chemeleon_hts.ckpt"
OUT_CKPT        = MODELS_DIR / "chemeleon_hts_mtl.ckpt"

EPOCHS         = 30
BATCH_SIZE     = 256
TEST_BATCH     = 171              # divides 513 cleanly (sub28 dropped a row at 256)
LR             = 1e-4
PATIENCE       = 5
SIGMA          = 0.3
SEED           = 0
TASK_WEIGHTS   = [1.0, 0.5]       # log2_fc primary, soft-binary aux
ACTIVE_THRESHOLD = 1.0            # log2_fc above which "active"
SIG_STEEPNESS    = 2.0


def smiles_to_inchikey(smi):
    if not isinstance(smi, str): return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return _inchi.MolToInchiKey(mol)
    except Exception: return None


def load_hts_data():
    print("Loading HTS single-concentration data ...")
    df = pd.read_csv(DATA_DIR / "single_concentration_train.csv")
    df = df.dropna(subset=["SMILES", "log2_fc_estimate", "concentration_M"])
    print(f"  rows after NaN drop: {len(df)}")

    test_df = pd.read_csv(DATA_DIR / ("test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
                                       else "test_raw.csv"))
    test_keys = {smiles_to_inchikey(s) for s in test_df["SMILES"]} - {None}
    df["inchikey"] = df["SMILES"].map(smiles_to_inchikey)
    pre = len(df)
    df = df[~df["inchikey"].isin(test_keys)].dropna(subset=["inchikey"])
    print(f"  rows after test-leakage filter: {len(df)} (-{pre - len(df)})")

    se = df["log2_fc_stderr"].fillna(df["log2_fc_stderr"].median())
    w  = 1.0 / (se**2 + SIGMA**2)
    q05, q95 = np.quantile(w, [0.05, 0.95])
    df["sample_weight"] = np.clip(w.values, q05, q95)
    df["log_conc"] = np.log10(df["concentration_M"].clip(lower=1e-12))

    # Soft-binary aux target: sigmoid around ACTIVE_THRESHOLD
    z = SIG_STEEPNESS * (df["log2_fc_estimate"] - ACTIVE_THRESHOLD)
    df["p_active"] = 1.0 / (1.0 + np.exp(-z))
    print(f"  log2_fc range [{df['log2_fc_estimate'].min():.2f}, {df['log2_fc_estimate'].max():.2f}]")
    print(f"  p_active mean {df['p_active'].mean():.3f}  (fraction 'active'-like)")

    return df.reset_index(drop=True)


def build_datapoints(df):
    from chemprop import data
    pts = []
    for _, row in df.iterrows():
        targets = [float(row["log2_fc_estimate"]), float(row["p_active"])]
        x_d = np.array([row["log_conc"]], dtype=np.float32)
        pts.append(data.MoleculeDatapoint.from_smi(
            row["SMILES"], targets, x_d=x_d, weight=float(row["sample_weight"])))
    return pts


def load_warm_encoder():
    from chemprop import models
    if not WARM_START_CKPT.exists():
        raise FileNotFoundError(WARM_START_CKPT)
    full = models.MPNN.load_from_checkpoint(str(WARM_START_CKPT), map_location="cpu")
    return full.message_passing


def train():
    import torch
    import lightning.pytorch as pl
    from chemprop import data, featurizers, nn, models
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    torch.manual_seed(SEED); pl.seed_everything(SEED, workers=True)

    df = load_hts_data()
    tr_df, va_df = train_test_split(
        df, test_size=0.1, random_state=SEED,
        stratify=pd.qcut(df["log2_fc_estimate"], q=10, duplicates="drop"),
    )
    print(f"\nTrain: {len(tr_df)}  Val: {len(va_df)}")

    tr_pts = build_datapoints(tr_df); va_pts = build_datapoints(va_df)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(tr_pts, featurizer)
    scalers = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(va_pts, featurizer)
    va_dset.normalize_targets(scalers)

    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH_SIZE, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    mp  = load_warm_encoder()
    agg = nn.MeanAggregation()
    criterion = nn.metrics.MSE(task_weights=TASK_WEIGHTS)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + 1,
        n_tasks=2,
        output_transform=nn.UnscaleTransform.from_standard_scaler(scalers),
        criterion=criterion,
        dropout=0.1,
    )
    model = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.RMSE()])
    model.hparams["lr"] = LR

    ckpt_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR / "hts_mtl_tmp"),
        filename="best", monitor="val_loss", mode="min",
        save_top_k=1, save_last=False,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", min_delta=1e-3)

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="auto", devices=1,
        logger=False, enable_progress_bar=False,
        callbacks=[ckpt_cb, es_cb], gradient_clip_val=1.0,
    )
    trainer.fit(model, tr_loader, va_loader)

    best = Path(ckpt_cb.best_model_path)
    if best.exists():
        import shutil
        shutil.copy(best, OUT_CKPT)
        print(f"\nSaved best ckpt → {OUT_CKPT}  (val_loss={ckpt_cb.best_model_score:.4f})")
    else:
        trainer.save_checkpoint(str(OUT_CKPT))
        print(f"\nSaved final ckpt → {OUT_CKPT}")
    return OUT_CKPT


def extract_embeddings(ckpt_path: Path):
    import torch
    from chemprop import data, featurizers, models

    print("\nExtracting embeddings for challenge train + test ...")
    train_df = pd.read_csv(DATA_DIR / ("train_final.csv" if (DATA_DIR / "train_final.csv").exists()
                                       else "train_curated.csv"))
    test_df  = pd.read_csv(DATA_DIR / ("test_curated.csv" if (DATA_DIR / "test_curated.csv").exists()
                                       else "test_raw.csv"))
    print(f"  train: {len(train_df)}  test: {len(test_df)}")

    full = models.MPNN.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    mp = full.message_passing.eval()
    agg = full.agg.eval()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    def embed(smiles, batch):
        pts = [data.MoleculeDatapoint.from_smi(s, [0.0, 0.0],
                                               x_d=np.array([0.0], dtype=np.float32))
               for s in smiles]
        dset = data.MoleculeDataset(pts, featurizer)
        loader = data.build_dataloader(dset, batch_size=batch, num_workers=0, shuffle=False)
        out = []
        with torch.inference_mode():
            for b in loader:
                H = mp(b.bmg)
                e = agg(H, b.bmg.batch)
                out.append(e.cpu().numpy())
        return np.concatenate(out, axis=0)

    X_tr = embed(train_df["SMILES"].tolist(), BATCH_SIZE)
    X_te = embed(test_df["SMILES"].tolist(), TEST_BATCH)   # 513 = 3 × 171
    assert X_tr.shape[0] == len(train_df)
    assert X_te.shape[0] == len(test_df), f"missing rows: {X_te.shape[0]} vs {len(test_df)}"
    np.save(FEATURE_DIR / "chemeleon_hts_mtl_train.npy", X_tr)
    np.save(FEATURE_DIR / "chemeleon_hts_mtl_test.npy",  X_te)
    print(f"  saved chemeleon_hts_mtl_{{train,test}}.npy  shapes {X_tr.shape} {X_te.shape}")


def main():
    print("=== 43_hts_mtl_encoder.py ===\n")
    if OUT_CKPT.exists():
        print(f"  {OUT_CKPT} exists — skipping training")
    else:
        train()
    extract_embeddings(OUT_CKPT)


if __name__ == "__main__":
    main()
