"""
TabPFN v3 Butina OOF — proper base learner for NNLS ensemble.

Requires: tabpfn>=8.0.2 + TABPFN_TOKEN env var (register at https://ux.priorlabs.ai)

Previous TabPFN usage was test-only (no OOF) so it couldn't participate in
the NNLS stack as a weighted base learner. With GPU and the new v3 model,
we can now run proper Butina 3×5-fold OOF on multiple feature sets and
compare directly against TabICL.

Feature sets tested:
  ecfp4_rdkit2d  (2265-d) — matches our best combo-study finding
  chemeleon_hts  (2048-d) — our dominant TabICL feature

TabPFN v3 uses: model_path="v3", device="cuda", n_estimators=8

Saves:
  results/tabpfn_v3_{feature}_oof_preds.npy
  results/tabpfn_v3_{feature}_test_preds.npy

Run: wsl -e bash -c "bash logs/_run_tabpfn_v3.sh"
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import DATA_DIR, FEATURE_DIR, butina_kfold  # type: ignore

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = ROOT / "results"
SEEDS       = [0, 1, 2]
KFOLD       = 5
MODEL_VER   = "auto"         # picks best available (v3 if token set, else v2.5)
PCA_DIM     = 256            # compress high-d features, same as TabICL
DEVICE      = "cuda"
N_EST       = 8

FEATURE_SETS = [
    ("chemeleon_hts_cont", 2048, PCA_DIM),  # cached — 23% NNLS weight
    # New: TabPFN_v3 on additional embeddings
    ("unimol",            512,  PCA_DIM),   # 512-d UniMol embedding
    ("drugclip_emb_2o9i", 128,  None),      # 128-d DrugCLIP, best receptor coverage
    ("drugclip_emb",      128,  None),      # 128-d DrugCLIP mean across receptors
    ("drugclip_concat",   384,  None),      # 384-d DrugCLIP concat
    # MolE embeddings — runs only if 81_mole_embeddings.py was run first
    # (otherwise load_features() will fail and skip this entry)
    ("mole",              512,  PCA_DIM),   # MolE embeddings (DeBERTa-style, 512-d)
]


def load_data():
    path = DATA_DIR / "train_final.csv" if (DATA_DIR/"train_final.csv").exists() \
           else DATA_DIR / "train_curated.csv"
    test = DATA_DIR / "test_curated.csv" if (DATA_DIR/"test_curated.csv").exists() \
           else DATA_DIR / "test_raw.csv"
    return pd.read_csv(path), pd.read_csv(test), np.load(FEATURE_DIR/"y_train.npy")


def load_features(name, pca_dim):
    X_tr = np.load(FEATURE_DIR / f"{name}_train.npy").astype(np.float32)
    X_te = np.load(FEATURE_DIR / f"{name}_test.npy").astype(np.float32)
    if pca_dim and X_tr.shape[1] > pca_dim:
        sc  = StandardScaler().fit(X_tr)
        X_tr = sc.transform(X_tr)
        X_te = sc.transform(X_te)
        pca = PCA(n_components=pca_dim, random_state=0).fit(X_tr)
        ev  = pca.explained_variance_ratio_.sum()
        print(f"  PCA {X_tr.shape[1]} → {pca_dim}d  (var={ev:.3f})")
        X_tr = pca.transform(X_tr).astype(np.float32)
        X_te = pca.transform(X_te).astype(np.float32)
    return X_tr, X_te


def run_tabpfn_oof(tag, X_tr, X_te, y, smiles):
    from tabpfn import TabPFNRegressor

    oof_path  = RESULTS_DIR / f"tabpfn_v3_{tag}_oof_preds.npy"
    test_path = RESULTS_DIR / f"tabpfn_v3_{tag}_test_preds.npy"
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path)
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        rho  = float(spearmanr(y, oof).statistic)
        print(f"  SKIP {tag} (cached)  OOF RMSE={rmse:.5f}  ρ={rho:.4f}")
        return

    all_oof, all_test = [], []
    for seed in SEEDS:
        folds = butina_kfold(smiles, k=KFOLD, seed=seed)
        oof_seed = np.full(len(y), np.nan, dtype=np.float32)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            print(f"  Seed {seed} fold {fi+1}/{KFOLD} ...", flush=True)
            m = TabPFNRegressor(model_path=MODEL_VER, device=DEVICE,
                                n_estimators=N_EST, random_state=seed)
            m.fit(X_tr[tr_idx], y[tr_idx])
            oof_seed[va_idx] = m.predict(X_tr[va_idx])

        rmse_s = float(np.sqrt(mean_squared_error(y, oof_seed)))
        rho_s  = float(spearmanr(y, oof_seed).statistic)
        print(f"  Seed {seed}: RMSE={rmse_s:.5f}  ρ={rho_s:.4f}")
        all_oof.append(oof_seed)

        m_full = TabPFNRegressor(model_path=MODEL_VER, device=DEVICE,
                                 n_estimators=N_EST, random_state=seed)
        m_full.fit(X_tr, y)
        all_test.append(m_full.predict(X_te))

    mean_oof  = np.nanmean(all_oof, axis=0)
    mean_test = np.mean(all_test, axis=0)
    rmse = float(np.sqrt(mean_squared_error(y, mean_oof)))
    rho  = float(spearmanr(y, mean_oof).statistic)
    print(f"  {tag} FINAL: RMSE={rmse:.5f}  ρ={rho:.4f}  "
          f"(TabICL baseline: chemeleon_hts=0.500, mtl=0.495)")

    np.save(oof_path,  mean_oof)
    np.save(test_path, mean_test)
    print(f"  Saved tabpfn_v3_{tag}")


def main():
    print(f"=== 69_tabpfn_v3_oof.py  model={MODEL_VER}  device={DEVICE} ===\n")

    if not os.environ.get("TABPFN_TOKEN"):
        print("WARNING: TABPFN_TOKEN not set — will fail for v3.")
        print("  Set: export TABPFN_TOKEN=<key from https://ux.priorlabs.ai>")

    train_df, test_df, y = load_data()
    smiles = train_df["SMILES"].tolist()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}\n")

    for feat_name, n_dim, pca_dim in FEATURE_SETS:
        print(f"\n--- TabPFN v3 on {feat_name} ---")
        try:
            X_tr, X_te = load_features(feat_name, pca_dim)
        except FileNotFoundError as e:
            print(f"  SKIP {feat_name}: feature file missing ({e})")
            continue
        run_tabpfn_oof(feat_name, X_tr, X_te, y, smiles)

    print("\nDone. Add to 26_submission13.py MODELS dict to include in NNLS.")
    print("Results comparison vs TabICL:")
    for feat_name, _, _ in FEATURE_SETS:
        p = RESULTS_DIR / f"tabpfn_v3_{feat_name}_oof_preds.npy"
        if p.exists():
            oof = np.load(p)
            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            rho  = float(spearmanr(y, oof).statistic)
            print(f"  tabpfn_v3_{feat_name:25s}  RMSE={rmse:.5f}  ρ={rho:.4f}")


if __name__ == "__main__":
    main()
