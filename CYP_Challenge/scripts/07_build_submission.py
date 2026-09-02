"""Combine every model into the final submissions via Caruana bagged ensemble
selection (Caruana et al. 2004) -- forward stepwise selection with replacement,
bagged over random candidate subsets. Beats a plain mean or NNLS on true blind data
(see README references) because it doesn't destructively reallocate weight among
correlated candidates the way a continuous optimizer can.

Run from the repo root, after scripts 02-05:  python scripts/06_build_submission.py
"""

import json
import sys

import numpy as np
import pandas as pd
from cyp_submission.calibration import apply_placement_correction, fit_placement_correction
from cyp_submission.caruana import caruana_bagged_ensemble
from cyp_submission.data import ISOFORMS, TDI_ISOFORMS, load_test_blinded, load_train_inhibition, load_train_tdi
from cyp_submission.metrics import isoform_st_rae
from cyp_submission.paths import REFERENCE_DIR, RESULTS_DIR, ensure_dirs
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, str(REFERENCE_DIR))
from validation.activity_validation import validate_activity_submission  # noqa: E402
from validation.tdi_validation import validate_tdi_submission  # noqa: E402

REGRESSION_SOURCES = {
    "lgb": (RESULTS_DIR / "lgb_activity_oof.csv", RESULTS_DIR / "lgb_activity_test.csv"),
    "tabpfn_adme_pretrain": (RESULTS_DIR / "tabpfn_adme_pretrain_oof.csv", RESULTS_DIR / "tabpfn_adme_pretrain_test.csv"),
    "tabpfn_chemeleon": (RESULTS_DIR / "tabpfn_chemeleon_oof.csv", RESULTS_DIR / "tabpfn_chemeleon_test.csv"),
    "chemprop_multitask": (RESULTS_DIR / "chemprop_multitask_oof.csv", RESULTS_DIR / "chemprop_multitask_test.csv"),
}
TDI_SOURCES = {
    "lgb": (RESULTS_DIR / "lgb_tdi_oof.csv", RESULTS_DIR / "lgb_tdi_test.csv"),
    "tabicl_adme_pretrain": (RESULTS_DIR / "tabicl_tdi_adme_pretrain_oof.csv",
                              RESULTS_DIR / "tabicl_tdi_adme_pretrain_test.csv"),
}


def build_regression_submission() -> None:
    df = load_train_inhibition()
    test = load_test_blinded()
    submission = test[["SMILES", "Molecule_Name"]].copy()
    scores = {}

    for iso in ISOFORMS:
        col = f"{iso}_pIC50_direct_inhibition"
        y = df[col].to_numpy()
        mask = ~np.isnan(y)
        conf_lo, conf_hi = df[f"{col}_conf_low"].to_numpy(), df[f"{col}_conf_high"].to_numpy()

        names, oof_cols, test_cols = [], [], []
        for name, (oof_path, test_path) in REGRESSION_SOURCES.items():
            if not (oof_path.exists() and test_path.exists()):
                print(f"skipping {name}: results not found, run its script first")
                continue
            oof_pred = pd.read_csv(oof_path)[f"{iso}_pred"].to_numpy()
            test_pred = pd.read_csv(test_path)[f"{iso}_pred"].to_numpy()
            m = mask & ~np.isnan(oof_pred)
            slope, intercept = fit_placement_correction(y[m], oof_pred[m])
            names += [f"{name}_raw", f"{name}_cal"]
            oof_cols += [oof_pred, apply_placement_correction(oof_pred, slope, intercept)]
            test_cols += [test_pred, apply_placement_correction(test_pred, slope, intercept)]

        oof_matrix = np.vstack(oof_cols).T
        test_matrix = np.vstack(test_cols).T
        m = mask & ~np.isnan(oof_matrix).any(axis=1)

        def score_fn(y_true, pred, _lo=conf_lo[m], _hi=conf_hi[m]):
            return isoform_st_rae(y_true, pred, _lo, _hi)

        weights = caruana_bagged_ensemble(oof_matrix[m], y[m], score_fn, minimize=True)
        st_rae = isoform_st_rae(y[m], oof_matrix[m] @ weights, conf_lo[m], conf_hi[m])
        top = sorted(zip(names, weights), key=lambda kv: -kv[1])[:5]
        scores[iso] = st_rae
        print(f"{iso}: Caruana ST-RAE={st_rae:.4f}  top picks={[(n, round(float(w), 3)) for n, w in top if w > 0]}")
        submission[col] = test_matrix @ weights

    scores["MA"] = float(np.mean([scores[iso] for iso in ISOFORMS]))
    print(f"\nMA-ST-RAE={scores['MA']:.4f}")

    out_path = RESULTS_DIR / "submission_activity.csv"
    submission.to_csv(out_path, index=False)
    ok, errs = validate_activity_submission(out_path, expected_ids=set(test["Molecule_Name"]))
    print(f"wrote {out_path}  valid={ok}")
    for e in errs:
        print(f"  ! {e}")
    (RESULTS_DIR / "final_scores_activity.json").write_text(json.dumps(scores, indent=2))


def build_tdi_submission() -> None:
    df = load_train_tdi()
    test = load_test_blinded()
    submission = test[["SMILES", "Molecule_Name"]].copy()
    scores = {}

    for iso in TDI_ISOFORMS:
        col = f"{iso}_is_TDI"
        y = df[col].to_numpy()
        mask = ~pd.isna(y)
        y_bool = y[mask].astype(bool)

        names, oof_cols, test_cols = [], [], []
        for name, (oof_path, test_path) in TDI_SOURCES.items():
            if not (oof_path.exists() and test_path.exists()):
                print(f"skipping {name}: results not found, run its script first")
                continue
            names.append(name)
            oof_cols.append(pd.read_csv(oof_path)[f"{iso}_pred"].to_numpy(dtype=float)[mask])
            test_cols.append(pd.read_csv(test_path)[f"{iso}_pred"].to_numpy(dtype=float))

        oof_matrix = np.vstack(oof_cols).T
        test_matrix = np.vstack(test_cols).T
        m = ~np.isnan(oof_matrix).any(axis=1)

        def score_fn(y_true, pred):
            return matthews_corrcoef(y_true, pred >= 0.5)

        weights = caruana_bagged_ensemble(oof_matrix[m], y_bool[m], score_fn, minimize=False)
        vote_share = oof_matrix[m] @ weights
        best_t, best_mcc = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 37):
            mcc = matthews_corrcoef(y_bool[m], vote_share >= t)
            if mcc > best_mcc:
                best_mcc, best_t = mcc, t
        scores[iso] = best_mcc
        print(f"{iso}: Caruana MCC={best_mcc:.4f}  threshold={best_t:.2f}  weights={dict(zip(names, np.round(weights, 3)))}")
        submission[col] = (test_matrix @ weights) >= best_t

    scores["MA"] = float(np.mean([scores[iso] for iso in TDI_ISOFORMS]))
    print(f"\nMA-MCC={scores['MA']:.4f}")

    out_path = RESULTS_DIR / "submission_tdi.csv"
    submission.to_csv(out_path, index=False)
    ok, errs = validate_tdi_submission(out_path, expected_ids=set(test["Molecule_Name"]))
    print(f"wrote {out_path}  valid={ok}")
    for e in errs:
        print(f"  ! {e}")
    (RESULTS_DIR / "final_scores_tdi.json").write_text(json.dumps(scores, indent=2))


def main() -> None:
    ensure_dirs()
    build_regression_submission()
    build_tdi_submission()


if __name__ == "__main__":
    main()
