"""
TabICL base learner on multiple feature sets for PXR ensemble.

TabICL is a pre-trained tabular foundation model (soda-inria/tabicl, sklearn API).
Competitor found TabICL + embedding stack = 0.5415 RAE solo, competitive with TabPFN.
Used here as a base learner on: ECFP4+RDKit2D, Mordred, ADMET-AI features.

Run in tabicl_env:
  wsl -e bash -c "/home/jeremy/mambaforge/bin/conda run -n tabicl_env python \
    /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/24_tabicl_base.py"

Prerequisites (features must exist):
  features/ecfp4_train.npy, features/rdkit2d_train.npy  (from 03_feature_generation.py)
  features/mordred2d_train.npy                            (from 03_feature_generation.py)
  features/admet_ai_train.npy                             (from 22_admet_features.py)
  features/chemeleon_hts_train.npy                        (from 23_chemeleon_hts_pretrain.py)

Saves:
  results/tabicl_{feature_set}_oof_preds.npy   — one per feature set
  results/tabicl_{feature_set}_test_preds.npy
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
FEATURE_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
from utils import butina_kfold

SEEDS  = [0, 1, 2]
KFOLD  = 5
PCA_DIM = 256   # compress high-dim features before TabICL (as competitor did)

TABICL_KWARGS = dict(
    n_estimators=4,
    random_state=42,
    device="cuda",   # falls back to cpu if unavailable
    use_amp=False,
    use_fa3=False,
    verbose=False,
)


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


def load_feature_set(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load train+test feature arrays. Returns None if not yet generated."""
    train_path = FEATURE_DIR / f"{name}_train.npy"
    test_path  = FEATURE_DIR / f"{name}_test.npy"
    if not train_path.exists() or not test_path.exists():
        print(f"  SKIP {name}: features not found (run prerequisite script first)")
        return None
    X_tr = np.load(train_path).astype(np.float32)
    X_te = np.load(test_path).astype(np.float32)
    if np.isnan(X_tr).all() or np.isnan(X_te).all():
        print(f"  SKIP {name}: all-NaN embeddings")
        return None
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X_train: np.ndarray, X_test: np.ndarray,
               pca_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN, StandardScale, optionally PCA-compress."""
    # Impute NaN with column median
    col_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        m = np.isnan(X_train[:, j]); X_train[m, j] = col_medians[j]
        m = np.isnan(X_test[:, j]);  X_test[m, j]  = col_medians[j]

    # Drop near-zero variance
    keep = np.var(X_train, axis=0) > 1e-8
    X_train, X_test = X_train[:, keep], X_test[:, keep]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train).astype(np.float32)
    X_test  = sc.transform(X_test).astype(np.float32)

    if pca_dim is not None and X_train.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train).astype(np.float32)
        X_test  = pca.transform(X_test).astype(np.float32)
        print(f"    PCA {keep.sum()} → {pca_dim}d  "
              f"(explained var: {pca.explained_variance_ratio_.sum():.1%})")

    return X_train, X_test


# ---------------------------------------------------------------------------
# OOF loop
# ---------------------------------------------------------------------------

def compute_oof(X_train: np.ndarray, X_test: np.ndarray,
                y: np.ndarray, smiles: list[str],
                label: str) -> tuple[np.ndarray, np.ndarray]:
    from tabicl import TabICLRegressor

    oof_cache  = RESULTS_DIR / f"tabicl_{label}_oof_preds.npy"
    test_cache = RESULTS_DIR / f"tabicl_{label}_test_preds.npy"

    if oof_cache.exists() and test_cache.exists():
        print(f"  SKIP {label} (cached)")
        return np.load(oof_cache), np.load(test_cache)

    all_seed_oof   = []
    all_seed_test  = []

    for seed in SEEDS:
        folds = butina_kfold(smiles, k=KFOLD, seed=seed)
        oof   = np.full(len(y), np.nan)
        seed_test_preds = []

        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            kwargs = {**TABICL_KWARGS, "random_state": seed * 100 + fold_i}
            r = TabICLRegressor(**kwargs)
            r.fit(X_train[tr_idx], y[tr_idx])
            oof[va_idx] = r.predict(X_train[va_idx])
            seed_test_preds.append(r.predict(X_test))

        rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])))
        rho  = float(spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)]).statistic)
        print(f"  Seed {seed}  OOF RMSE={rmse:.5f}  Spearman={rho:.4f}")
        all_seed_oof.append(oof)
        all_seed_test.append(np.mean(seed_test_preds, axis=0))

    mean_oof  = np.nanmean(all_seed_oof, axis=0)
    mean_test = np.mean(all_seed_test, axis=0)

    rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    rho  = float(spearmanr(y, mean_oof).statistic)
    print(f"\n  {label}  Mean OOF RMSE={rmse:.5f}  Spearman={rho:.4f}")

    np.save(oof_cache,  mean_oof)
    np.save(test_cache, mean_test)
    return mean_oof, mean_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FEATURE_SETS = [
    # (feature_name_in_files, label_for_output, apply_pca)
    ("ecfp4_rdkit2d",   "ecfp4_rdkit2d",    PCA_DIM),  # concat ECFP4+RDKit2D
    ("mordred2d",       "mordred2d",         PCA_DIM),
    ("admet_ai",        "admet_ai",          None),     # only 98 features, no PCA needed
    ("chemeleon_hts",   "chemeleon_hts",     PCA_DIM),  # 2048-dim HTS embeddings (binary pretrain)
    ("chemeleon_hts_cont", "chemeleon_hts_cont", PCA_DIM),  # 2048d continuous-HTS pretrain (39)
    ("chemeleon_hts_mtl",  "chemeleon_hts_mtl",  PCA_DIM),  # 2048d multi-task HTS pretrain (43)
    ("quadmetformer",      "quadmetformer",       None),     # 128d QM-pretrained equivariant GNN (zero-shot)
    ("quadmetformer_ft",   "quadmetformer_ft",    None),     # 128d QMF fine-tuned on PXR pEC50
    ("molformer",          "molformer",           PCA_DIM),  # 768d MoLFormer-XL — NaN, skipped
    ("mist_sider",         "mist_sider",          PCA_DIM),  # 512d MIST-28M SIDER frozen (76)
    ("mist_qm9",           "mist_qm9",            PCA_DIM),  # 512d MIST-26.9M QM9 frozen (76)
    ("mist_sider_ft",      "mist_sider_ft",       PCA_DIM),  # 512d MIST-SIDER fine-tuned encoder (77)
    ("mist_qm9_ft",        "mist_qm9_ft",         PCA_DIM),  # 512d MIST-QM9 fine-tuned encoder (77)
    # DrugCLIP embeddings from docked poses (78_drugclip_embeddings.py)
    ("drugclip_emb_2o9i",  "drugclip_emb_2o9i",   PCA_DIM),  # 128d DrugCLIP ligand emb, 2o9i
    ("drugclip_emb_8r81",  "drugclip_emb_8r81",   PCA_DIM),  # 128d DrugCLIP ligand emb, 8r81
    ("drugclip_emb_8eqz",  "drugclip_emb_8eqz",   PCA_DIM),  # 128d DrugCLIP ligand emb, 8eqz
    ("drugclip_emb",       "drugclip_emb",        PCA_DIM),  # 128d DrugCLIP mean across receptors
    ("drugclip_concat",    "drugclip_concat",     PCA_DIM),  # 384d DrugCLIP concat 3 receptors
    ("mqn",                "mqn",                 None),     # 42-d MQN (de la Vega: surprisingly competitive)
    ("ecfp4_count",        "ecfp4_count",         PCA_DIM),  # 2048-d count ECFP4 (>binary by ~0.02 MAE)
    ("chemeleon_ecfp4",   "chemeleon_ecfp4",     None),     # 300d vanilla→ECFP4 (55, small encoder)
    ("chemeleon_ecfp4_warm", "chemeleon_ecfp4_warm", PCA_DIM),  # 2048d CheMeleon→ECFP4 warm-start (56)
    ("vanilla_cp2048",    "vanilla_cp2048",      PCA_DIM),  # 2048d vanilla→pEC50 encoder emb (57)
    ("vanilla_ecfp4_2048","vanilla_ecfp4_2048",  PCA_DIM),  # 2048d vanilla→ECFP4 (58, fair vs 56)
    ("vanilla_hts",       "vanilla_hts",         PCA_DIM),  # 2048d vanilla→HTS log2_fc (59, vs chemeleon_hts_cont)
    ("all_combined",      "all_combined",        1024),     # E3-style 4313-d: PCA-1024 (79% var; 256 only 45%)
    ("ecfp4_16384",       "ecfp4_16384",         PCA_DIM),  # 16384-bit ECFP4 → PCA-256 (1024 OOM'd)
    # New: 3D descriptors + UniMol embeddings
    ("rdkit3d_pharm",   "rdkit3d_pharm",     None),     # 901d shape/pharmacophore, no PCA needed
    ("mordred3d",       "mordred3d",         None),     # 213d pure 3D Mordred, no PCA needed
    ("unimol",          "unimol",            None),     # 512d UniMol embeddings, no PCA needed
    # ChemBERTa-PubChem10M CLS embeddings
    ("chemberta_pubchem", "chemberta_pubchem", PCA_DIM), # 768d, PCA→256d like chemeleon_hts
]


def main():
    print("=== TabICL base learners (24_tabicl_base.py) ===\n")

    train_df, test_df, y = load_data()
    smiles = train_df["SMILES"].tolist()

    for feat_name, label, pca_dim in FEATURE_SETS:
        print(f"\n--- Feature set: {label} ---")

        # Handle concatenated feature sets
        if feat_name == "ecfp4_rdkit2d":
            r1 = load_feature_set("ecfp4")
            r2 = load_feature_set("rdkit2d")
            if r1 is None or r2 is None:
                continue
            X_tr = np.hstack([r1[0], r2[0]])
            X_te = np.hstack([r1[1], r2[1]])
        else:
            result = load_feature_set(feat_name)
            if result is None:
                continue
            X_tr, X_te = result

        print(f"  Raw shape: train={X_tr.shape}  test={X_te.shape}")
        X_tr, X_te = preprocess(X_tr.copy(), X_te.copy(), pca_dim=pca_dim)
        print(f"  Processed shape: {X_tr.shape}")

        compute_oof(X_tr, X_te, y, smiles, label)

    print("\n=== All TabICL base learners complete ===")
    print("OOF predictions saved to results/tabicl_*_oof_preds.npy")
    print("Add these as columns in the Ridge meta-learner ensemble.")


if __name__ == "__main__":
    main()
