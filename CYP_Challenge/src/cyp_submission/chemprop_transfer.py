"""ChemProp transfer-learning utilities: frozen-embedding extraction and multitask
fine-tuning, using chemprop's Python API directly (not its CLI — see main project's
notes on a WSL/CLI interop issue if you hit something similar)."""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.models.model import MPNN
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping


def compute_inverse_count_task_weights(df: pd.DataFrame, target_cols: list[str]) -> list[float]:
    """Per-task weights inversely proportional to each task's non-NaN row count,
    mean-normalized to 1.0 — scarcer tasks get more gradient share. Use a flat
    manual weight for auxiliary heads instead (would starve the primaries otherwise)."""
    counts = df[target_cols].notna().sum().to_numpy(dtype=float)
    inv = 1.0 / counts
    return (inv / inv.mean()).tolist()


def _resolve_accelerator(acc: str = "auto") -> str:
    if acc in {"gpu", "cuda", "auto"}:
        return "gpu" if torch.cuda.is_available() else "cpu"
    return acc


def _to_datapoints(df: pd.DataFrame, target_cols: list[str]) -> list:
    pts = []
    for _, row in df.iterrows():
        targets = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in target_cols]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], targets))
    return pts


def build_finetune_model(pretrained_path: str | Path, n_tasks: int, *, hidden_dim: int = 300,
                          n_layers: int = 2, dropout: float = 0.1, output_transform=None,
                          freeze_encoder: bool = False, task_weights: list[float] | None = None) -> MPNN:
    """Load a pretrained MPNN, keep its encoder, attach a fresh n_tasks-way head."""
    pretrained = MPNN.load_from_file(str(pretrained_path))
    mp = pretrained.message_passing
    if freeze_encoder:
        for p in mp.parameters():
            p.requires_grad = False
    agg = pretrained.agg
    criterion = nn.metrics.MSE(task_weights=task_weights) if task_weights is not None else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            n_tasks=n_tasks, dropout=dropout, output_transform=output_transform,
            **({"criterion": criterion} if criterion is not None else {}),
        )
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    return models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)


def fit_finetune(train_df: pd.DataFrame, val_df: pd.DataFrame, target_cols: list[str],
                  pretrained_path: str | Path, *, num_epochs: int = 50, hidden_dim: int = 300,
                  n_layers: int = 2, dropout: float = 0.1, freeze_encoder: bool = False,
                  accelerator: str = "gpu", batch_size: int = 256, patience: int = 10,
                  task_weights: list[float] | None = None):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_pts = _to_datapoints(train_df, target_cols)
    val_pts = _to_datapoints(val_df, target_cols)

    train_dset = data.MoleculeDataset(train_pts, featurizer)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_pts, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=0, batch_size=batch_size)
    val_loader = data.build_dataloader(val_dset, num_workers=0, batch_size=batch_size, shuffle=False)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_finetune_model(
        pretrained_path, n_tasks=len(target_cols), hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=dropout, output_transform=output_transform, freeze_encoder=freeze_encoder,
        task_weights=task_weights,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            accelerator=_resolve_accelerator(accelerator), devices=1, max_epochs=num_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        )
        trainer.fit(model, train_loader, val_loader)
    return model, scaler


def embed_with_model(model: MPNN, smiles_list: list[str], batch_size: int = 256) -> np.ndarray:
    """Frozen graph-level embeddings from an already-loaded MPNN (message_passing +
    agg only, via chemprop's own MPNN.fingerprint() — no FFN head involved)."""
    model.eval()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(smi, [0.0]) for smi in smiles_list]
    dset = data.MoleculeDataset(pts, featurizer)
    # drop_last=False is required -- chemprop's default silently drops a size-1
    # remainder batch, desyncing every downstream index-aligned array otherwise.
    loader = data.build_dataloader(dset, num_workers=0, batch_size=batch_size, shuffle=False,
                                    drop_last=False)
    vectors = []
    with torch.inference_mode():
        for batch in loader:
            bmg, V_d, X_d, *_ = batch
            h = model.fingerprint(bmg, V_d, X_d)
            vectors.append(h.numpy())
    result = np.concatenate(vectors, axis=0)
    if len(result) != len(smiles_list):
        raise RuntimeError(f"embedded {len(result)} molecules but expected {len(smiles_list)}")
    return result


def extract_embeddings(pretrained_path: str | Path, smiles_list: list[str],
                        batch_size: int = 256) -> np.ndarray:
    model = MPNN.load_from_file(str(pretrained_path))
    return embed_with_model(model, smiles_list, batch_size=batch_size)


def predict_finetune(model: MPNN, df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_pts = [data.MoleculeDatapoint.from_smi(smi, [0.0] * len(target_cols)) for smi in df["SMILES"]]
    test_dset = data.MoleculeDataset(test_pts, featurizer)
    test_loader = data.build_dataloader(test_dset, num_workers=0, shuffle=False)
    with torch.inference_mode():
        trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                              accelerator=_resolve_accelerator("auto"), devices=1)
        preds = trainer.predict(model, test_loader)
    arr = np.array(list(itertools.chain(*preds)))
    return pd.DataFrame(arr, columns=target_cols, index=df.index)
