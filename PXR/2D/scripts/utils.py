"""Shared utilities: descriptors, Butina CV splitter, metrics."""
from __future__ import annotations

# ---- Column name constants (actual names from the dataset) ----------------
COL_SMILES = "SMILES"
COL_NAME   = "Molecule Name"
COL_PECSO  = "pEC50"
COL_SE     = "pEC50_std.error (-log10(molarity))"
COL_ID     = "OCNT_ID"   # only present in train

import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# SMILES → mol helpers
# ---------------------------------------------------------------------------

def smiles_to_mol(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # re-parse canonical form to catch implicit valence issues
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None


def remove_salts(mol):
    from rdkit.Chem.SaltRemover import SaltRemover
    remover = SaltRemover()
    stripped = remover.StripMol(mol)
    return stripped if stripped.GetNumAtoms() > 0 else mol


def mol_to_inchikey(mol) -> str | None:
    try:
        inchi = Chem.MolToInchi(mol)
        if inchi is None:
            return None
        return Chem.InchiToInchiKey(inchi)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def ecfp4_fp(mol, n_bits: int = 2048):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def ecfp4_array(mols, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is not None:
            fp = ecfp4_fp(mol, n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def fcfp4_array(mols, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((len(mols), n_bits), dtype=np.uint8)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=False)
    ffgen = AllChem.AdditionalOutput()
    for i, mol in enumerate(mols):
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=n_bits, useFeatures=True
            )
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def tanimoto_matrix(fps) -> np.ndarray:
    n = len(fps)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        mat[i, :i] = sims
        mat[:i, i] = sims
    np.fill_diagonal(mat, 1.0)
    return mat


# ---------------------------------------------------------------------------
# 2D Descriptors
# ---------------------------------------------------------------------------

def rdkit2d_array(mols) -> tuple[np.ndarray, list[str]]:
    rows = []
    for mol in mols:
        if mol is not None:
            d = Descriptors.CalcMolDescriptors(mol)
            rows.append(list(d.values()))
        else:
            rows.append([np.nan] * len(Descriptors._descList))
    names = [n for n, _ in Descriptors._descList]
    arr = np.array(rows, dtype=np.float64)
    return arr, names


def mordred2d_array(mols) -> tuple[np.ndarray, list[str]]:
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=True)
    results = calc.pandas(mols)
    names = list(results.columns)
    arr = results.to_numpy(dtype=np.float64, na_value=np.nan)
    return arr, names


def clean_descriptor_matrix(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    names: list[str],
    nan_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[str], RobustScaler]:
    """Drop columns with too many NaN/inf in train, then scale."""
    arr = train_arr.copy().astype(np.float64)
    arr[~np.isfinite(arr)] = np.nan
    nan_frac = np.isnan(arr).mean(axis=0)
    keep = nan_frac < nan_threshold
    arr = arr[:, keep]
    test = test_arr.copy().astype(np.float64)[:, keep]
    test[~np.isfinite(test)] = np.nan
    # impute remaining NaN with column median
    col_medians = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_medians[j]
        mask2 = np.isnan(test[:, j])
        test[mask2, j] = col_medians[j]
    scaler = RobustScaler()
    arr = scaler.fit_transform(arr)
    test = scaler.transform(test)
    kept_names = [n for n, k in zip(names, keep) if k]
    return arr, test, kept_names, scaler


# ---------------------------------------------------------------------------
# Butina k-fold CV splitter
# ---------------------------------------------------------------------------

def butina_kfold(
    smiles: list[str],
    k: int = 5,
    seed: int = 0,
    cutoff: float = 0.4,
    n_bits: int = 2048,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Scaffold-diverse k-fold splits via Butina clustering.
    Returns list of (train_indices, val_indices).
    """
    mols = [smiles_to_mol(s) for s in smiles]
    fps = [ecfp4_fp(m, n_bits) if m is not None else None for m in mols]

    # distance matrix (upper triangle, row-major)
    n = len(fps)
    dists = []
    for i in range(1, n):
        fp_i = fps[i]
        if fp_i is None:
            dists.extend([1.0] * i)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(
                fp_i, [fp if fp is not None else fp_i for fp in fps[:i]]
            )
            dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # clusters: list of tuples of indices (first element = centroid)

    rng = np.random.default_rng(seed)
    cluster_list = [list(c) for c in clusters]
    rng.shuffle(cluster_list)

    # Distribute clusters into k folds (greedy round-robin by cluster size, largest first)
    cluster_list.sort(key=len, reverse=True)
    folds: list[list[int]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for clust in cluster_list:
        min_fold = int(np.argmin(fold_sizes))
        folds[min_fold].extend(clust)
        fold_sizes[min_fold] += len(clust)

    splits = []
    all_idx = np.arange(n)
    for fi in range(k):
        val_idx = np.array(folds[fi])
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def repeated_butina_cv(
    smiles: list[str],
    k: int = 5,
    n_repeats: int = 3,
    seeds: list[int] | None = None,
    cutoff: float = 0.4,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Returns list of (repeat_idx, fold_idx, train_indices, val_indices).
    """
    if seeds is None:
        seeds = list(range(n_repeats))
    result = []
    for ri, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(butina_kfold(smiles, k=k, seed=seed, cutoff=cutoff)):
            result.append((ri, fi, tr, va))
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


# ---------------------------------------------------------------------------
# Feature I/O
# ---------------------------------------------------------------------------

FEATURE_DIR = Path(__file__).parent / "features"
DATA_DIR = Path(__file__).parent / "data"


def save_features(name: str, train: np.ndarray, test: np.ndarray, scaler=None):
    FEATURE_DIR.mkdir(exist_ok=True)
    np.save(FEATURE_DIR / f"{name}_train.npy", train)
    np.save(FEATURE_DIR / f"{name}_test.npy", test)
    if scaler is not None:
        with open(FEATURE_DIR / f"{name}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    print(f"  {name}: train={train.shape}, test={test.shape}")


def load_features(name: str) -> tuple[np.ndarray, np.ndarray]:
    train = np.load(FEATURE_DIR / f"{name}_train.npy")
    test = np.load(FEATURE_DIR / f"{name}_test.npy")
    return train, test
