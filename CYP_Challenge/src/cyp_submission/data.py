"""Loaders for the CYP challenge CSVs (HF dataset ``openadmet/cyp-challenge-train-test``,
downloaded by ``scripts/01_download_data.py``)."""

from __future__ import annotations

import pandas as pd

from cyp_submission.paths import DATA_RAW

HF_DIR = DATA_RAW / "hf_cyp_challenge"

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TDI_ISOFORMS = ("CYP2D6", "CYP3A4")  # only these two are scored on the TDI track


def load_train_inhibition() -> pd.DataFrame:
    return pd.read_csv(HF_DIR / "cyp-challenge-TRAIN_inhibition.csv")


def load_train_tdi() -> pd.DataFrame:
    return pd.read_csv(HF_DIR / "cyp-challenge-TRAIN_TDI.csv")


def load_single_concentration() -> pd.DataFrame:
    return pd.read_csv(HF_DIR / "cyp-challenge-single-concentration-TRAIN.csv")


def load_test_blinded() -> pd.DataFrame:
    return pd.read_csv(HF_DIR / "cyp-challenge-TEST-BLINDED.csv")
