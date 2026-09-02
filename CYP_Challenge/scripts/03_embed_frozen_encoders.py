"""Frozen-embedding extraction: message_passing+agg forward pass only, no
fine-tuning -- our repeated finding (see README) is that freezing these checkpoints
and feeding a tabular in-context model (TabPFN/TabICL) beats fine-tuning them on
this challenge's own labels. Two checkpoints, both from ``adme_pretrain`` (a
multitask ChemProp D-MPNN pretrained on public CYP/ADME data + a public-Novartis-
surrogate CYP panel referenced in an OpenADMET community talk): ``chemprop_medium``
(the strongest single embedding source) and ``chemprop_chemeleon`` (CheMeleon-
foundation-initialized, then further trained on the same corpus -- a genuinely
decorrelated second cluster, see README).

Embeddings don't depend on which track uses them, only on the encoder + SMILES, so
the same test-set embedding is reused for both the regression and TDI tracks.

Run from the repo root:  python scripts/03_embed_frozen_encoders.py
"""

import numpy as np
from cyp_submission.chemprop_transfer import extract_embeddings
from cyp_submission.data import load_test_blinded, load_train_inhibition, load_train_tdi
from cyp_submission.paths import CHECKPOINTS_DIR, DATA_INTERIM, ensure_dirs

CHECKPOINTS = {
    "adme_pretrain": CHECKPOINTS_DIR / "chemprop_medium.pt",
    "chemeleon": CHECKPOINTS_DIR / "chemprop_chemeleon.pt",
}


def main() -> None:
    ensure_dirs()
    reg_train = load_train_inhibition()
    tdi_train = load_train_tdi()
    test = load_test_blinded()

    for name, ckpt in CHECKPOINTS.items():
        print(f"=== {name} ===")
        reg_emb = extract_embeddings(ckpt, reg_train["SMILES"].tolist())
        tdi_emb = extract_embeddings(ckpt, tdi_train["SMILES"].tolist())
        test_emb = extract_embeddings(ckpt, test["SMILES"].tolist())
        np.save(DATA_INTERIM / f"train_activity_{name}_emb.npy", reg_emb)
        np.save(DATA_INTERIM / f"train_tdi_{name}_emb.npy", tdi_emb)
        np.save(DATA_INTERIM / f"test_{name}_emb.npy", test_emb)
        print(f"  train_activity={reg_emb.shape}  train_tdi={tdi_emb.shape}  test={test_emb.shape}")


if __name__ == "__main__":
    main()
