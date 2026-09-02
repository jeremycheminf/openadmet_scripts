"""TabICL on frozen adme_pretrain embeddings (TDI track only -- TabICL specifically
beat TabPFN here, see README). Runs in whichever env has `tabicl` installed (usually
a separate env from tabpfn -- the two packages commonly conflict on shared deps).

Run from the repo root:  python scripts/05_run_tabicl_tdi.py
"""

import numpy as np
from cyp_submission.data import TDI_ISOFORMS, load_test_blinded, load_train_tdi
from cyp_submission.paths import DATA_INTERIM, ensure_dirs
from cyp_submission.tabular_cv import pca_pipeline, run_tdi_cv
from tabicl import TabICLClassifier


def tabicl_factory():
    return TabICLClassifier(n_estimators=8, device="cuda", random_state=42)


def main() -> None:
    ensure_dirs()
    tdi_train = load_train_tdi()
    test = load_test_blinded()
    tdi_folds = np.load(DATA_INTERIM / "train_tdi_folds.npy")

    X_train_raw = np.load(DATA_INTERIM / "train_tdi_adme_pretrain_emb.npy")
    X_test_raw = np.load(DATA_INTERIM / "test_adme_pretrain_emb.npy")
    X_train, X_test = pca_pipeline(X_train_raw, X_test_raw)
    run_tdi_cv(tabicl_factory, "tabicl_tdi_adme_pretrain", tdi_train, X_train, X_test,
               tdi_folds, TDI_ISOFORMS, test)


if __name__ == "__main__":
    main()
