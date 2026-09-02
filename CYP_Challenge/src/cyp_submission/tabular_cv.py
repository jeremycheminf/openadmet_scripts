"""Shared CV loop for tabular-foundation-model runs (TabPFN/TabICL on PCA-200 of
frozen embeddings) -- fit one model per isoform per fold (Butina CV), save OOF +
full-fit test predictions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from cyp_submission.paths import RESULTS_DIR
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def pca_pipeline(X_train_raw: np.ndarray, X_test_raw: np.ndarray, n_components: int = 200):
    n = min(n_components, X_train_raw.shape[0], X_train_raw.shape[1])
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n, random_state=42)),
    ])
    X_train = pipe.fit_transform(X_train_raw).astype(np.float32)
    X_test = pipe.transform(X_test_raw).astype(np.float32)
    return X_train, X_test


def run_regression_cv(model_factory, tag: str, df: pd.DataFrame, X_train: np.ndarray,
                       X_test: np.ndarray, folds: np.ndarray, isoforms: tuple[str, ...],
                       test_ids: pd.DataFrame) -> None:
    oof_df = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"]})
    test_df = test_ids[["Molecule_Name", "SMILES"]].copy()
    n_folds = int(folds.max()) + 1

    for iso in isoforms:
        col = f"{iso}_pIC50_direct_inhibition"
        y = df[col].to_numpy()
        mask = ~np.isnan(y)
        oof = np.full(len(df), np.nan, dtype=np.float32)
        for f in range(n_folds):
            tr, va = mask & (folds != f), mask & (folds == f)
            if va.sum() == 0 or tr.sum() == 0:
                continue
            model = model_factory()
            model.fit(X_train[tr], y[tr])
            oof[va] = np.asarray(model.predict(X_train[va])).ravel()
        oof_df[f"{iso}_pred"] = oof
        final = model_factory()
        final.fit(X_train[mask], y[mask])
        test_df[f"{iso}_pred"] = np.asarray(final.predict(X_test)).ravel()
        print(f"[{tag}] {iso}: n_oof={int((mask & ~np.isnan(oof)).sum())}")

    oof_df.to_csv(RESULTS_DIR / f"{tag}_oof.csv", index=False)
    test_df.to_csv(RESULTS_DIR / f"{tag}_test.csv", index=False)
    print(f"[{tag}] wrote {tag}_oof.csv / {tag}_test.csv")


def run_tdi_cv(model_factory, tag: str, df: pd.DataFrame, X_train: np.ndarray,
               X_test: np.ndarray, folds: np.ndarray, tdi_isoforms: tuple[str, ...],
               test_ids: pd.DataFrame) -> None:
    oof_df = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"]})
    test_df = test_ids[["Molecule_Name", "SMILES"]].copy()
    n_folds = int(folds.max()) + 1

    for iso in tdi_isoforms:
        col = f"{iso}_is_TDI"
        y = df[col].to_numpy()
        mask = ~pd.isna(y)
        oof = np.full(len(df), np.nan)
        for f in range(n_folds):
            tr, va = mask & (folds != f), mask & (folds == f)
            if va.sum() == 0 or tr.sum() == 0:
                continue
            y_tr = df.loc[tr, col].astype(bool).to_numpy()
            if len(np.unique(y_tr)) < 2:
                continue
            model = model_factory()
            model.fit(X_train[tr], y_tr)
            oof[va] = np.asarray(model.predict(X_train[va])).ravel()
        oof_df[f"{iso}_pred"] = oof
        final = model_factory()
        final.fit(X_train[mask], df.loc[mask, col].astype(bool).to_numpy())
        test_df[f"{iso}_pred"] = np.asarray(final.predict(X_test)).ravel().astype(bool)
        print(f"[{tag}] {iso}: n_oof={int((mask & ~np.isnan(oof)).sum())}")

    oof_df.to_csv(RESULTS_DIR / f"{tag}_oof.csv", index=False)
    test_df.to_csv(RESULTS_DIR / f"{tag}_test.csv", index=False)
    print(f"[{tag}] wrote {tag}_oof.csv / {tag}_test.csv")
