"""Official challenge metrics, loaded directly from the vendored tutorial code so
local scores use the *exact* same formula as the leaderboard
(``reference/VENDORED_FROM.txt``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from cyp_submission.data import ISOFORMS, TDI_ISOFORMS
from cyp_submission.paths import REFERENCE_DIR


def _load_vendored_scoring_module():
    path = REFERENCE_DIR / "evaluation" / "custom_scoring_functions.py"
    spec = importlib.util.spec_from_file_location("_cyp_challenge_official_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_official = _load_vendored_scoring_module()
st_rae = _official.rae_soft_threshold_absolute_error


def isoform_st_rae(y_true: np.ndarray, y_pred: np.ndarray,
                    conf_low: np.ndarray, conf_high: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    return float(st_rae(y_true[mask], y_pred[mask], y_true_upper=conf_high[mask], y_true_lower=conf_low[mask]))


def macro_st_rae(df: pd.DataFrame, pred_cols: dict[str, str]) -> dict[str, float]:
    scores = {}
    for iso in ISOFORMS:
        base = f"{iso}_pIC50_direct_inhibition"
        scores[iso] = isoform_st_rae(
            df[base].to_numpy(), df[pred_cols[iso]].to_numpy(),
            df[f"{base}_conf_low"].to_numpy(), df[f"{base}_conf_high"].to_numpy(),
        )
    scores["MA"] = float(np.nanmean([scores[iso] for iso in ISOFORMS]))
    return scores


def tdi_mcc(df: pd.DataFrame, pred_cols: dict[str, str]) -> dict[str, float]:
    scores = {}
    for iso in TDI_ISOFORMS:
        label_col = f"{iso}_is_TDI"
        mask = df[label_col].notna()
        if mask.sum() == 0:
            scores[iso] = float("nan")
            continue
        scores[iso] = float(matthews_corrcoef(
            df.loc[mask, label_col].astype(bool), df.loc[mask, pred_cols[iso]].astype(bool)))
    scores["MA"] = float(np.nanmean([scores[iso] for iso in TDI_ISOFORMS]))
    return scores
