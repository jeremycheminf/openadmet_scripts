"""TabPFN on frozen embeddings (regression track, both checkpoints). Runs in
whichever env has `tabpfn` installed (needs a CUDA GPU for reasonable runtime;
falls back to CPU otherwise).

Run from the repo root:  python scripts/04_run_tabpfn.py
"""

import numpy as np
from cyp_submission.data import ISOFORMS, load_test_blinded, load_train_inhibition
from cyp_submission.paths import DATA_INTERIM, ensure_dirs
from cyp_submission.tabular_cv import pca_pipeline, run_regression_cv
from tabpfn import TabPFNRegressor


def tabpfn_factory():
    return TabPFNRegressor(n_estimators=8, device="cuda", random_state=42)


def main() -> None:
    ensure_dirs()
    reg_train = load_train_inhibition()
    test = load_test_blinded()
    reg_folds = np.load(DATA_INTERIM / "train_inhibition_folds.npy")

    for name in ("adme_pretrain", "chemeleon"):
        X_train_raw = np.load(DATA_INTERIM / f"train_activity_{name}_emb.npy")
        X_test_raw = np.load(DATA_INTERIM / f"test_{name}_emb.npy")
        X_train, X_test = pca_pipeline(X_train_raw, X_test_raw)
        run_regression_cv(tabpfn_factory, f"tabpfn_{name}", reg_train, X_train, X_test,
                           reg_folds, ISOFORMS, test)


if __name__ == "__main__":
    main()
