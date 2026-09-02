"""LightGBM baseline on ECFP4+RDKit2D, Butina 5-fold CV, both tracks. Fast, no GPU
-- also serves as a real (if modest) ensemble contributor later, since classical
descriptors are the least-correlated feature source against everything else here.

Run from the repo root:  python scripts/02_baseline_lgb.py
"""

import json

import numpy as np
import pandas as pd
from cyp_submission.data import ISOFORMS, TDI_ISOFORMS, load_test_blinded, load_train_inhibition, load_train_tdi
from cyp_submission.features import featurize
from cyp_submission.metrics import isoform_st_rae
from cyp_submission.paths import DATA_INTERIM, RESULTS_DIR, ensure_dirs
from cyp_submission.splits import butina_kfold
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import matthews_corrcoef

N_FOLDS = 5
LGB_REG_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
LGB_CLF_PARAMS = dict(**LGB_REG_PARAMS, class_weight="balanced")


def run_regression() -> None:
    df = load_train_inhibition()
    test = load_test_blinded()
    X = featurize(df["SMILES"].tolist())
    X_test = featurize(test["SMILES"].tolist())

    folds_path = DATA_INTERIM / "train_inhibition_folds.npy"
    folds = np.load(folds_path) if folds_path.exists() else butina_kfold(df["SMILES"].tolist(), N_FOLDS)
    np.save(folds_path, folds)

    oof = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"]})
    test_out = test[["Molecule_Name", "SMILES"]].copy()
    scores = {}
    for iso in ISOFORMS:
        col = f"{iso}_pIC50_direct_inhibition"
        y = df[col].to_numpy()
        mask = ~np.isnan(y)
        pred = np.full(len(df), np.nan)
        for f in range(N_FOLDS):
            tr, va = mask & (folds != f), mask & (folds == f)
            model = LGBMRegressor(**LGB_REG_PARAMS)
            model.fit(X[tr], y[tr])
            pred[va] = model.predict(X[va])
        oof[f"{iso}_pred"] = pred
        final = LGBMRegressor(**LGB_REG_PARAMS)
        final.fit(X[mask], y[mask])
        test_out[f"{iso}_pred"] = final.predict(X_test)
        scores[iso] = isoform_st_rae(y, pred, df[f"{col}_conf_low"].to_numpy(), df[f"{col}_conf_high"].to_numpy())
        print(f"lgb/{iso}: ST-RAE={scores[iso]:.4f}")
    scores["MA"] = float(np.mean([scores[iso] for iso in ISOFORMS]))
    print(f"lgb MA-ST-RAE={scores['MA']:.4f}")

    oof.to_csv(RESULTS_DIR / "lgb_activity_oof.csv", index=False)
    test_out.to_csv(RESULTS_DIR / "lgb_activity_test.csv", index=False)
    (RESULTS_DIR / "lgb_activity_scores.json").write_text(json.dumps(scores, indent=2))


def run_tdi() -> None:
    df = load_train_tdi()
    test = load_test_blinded()
    X = featurize(df["SMILES"].tolist())
    X_test = featurize(test["SMILES"].tolist())

    folds_path = DATA_INTERIM / "train_tdi_folds.npy"
    folds = np.load(folds_path) if folds_path.exists() else butina_kfold(df["SMILES"].tolist(), N_FOLDS)
    np.save(folds_path, folds)

    oof = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"]})
    test_out = test[["Molecule_Name", "SMILES"]].copy()
    scores = {}
    for iso in TDI_ISOFORMS:
        col = f"{iso}_is_TDI"
        y = df[col].to_numpy()
        mask = ~pd.isna(y)
        pred = np.full(len(df), np.nan)
        for f in range(N_FOLDS):
            tr, va = mask & (folds != f), mask & (folds == f)
            y_tr = df.loc[tr, col].astype(bool).to_numpy()
            if len(np.unique(y_tr)) < 2:
                continue
            model = LGBMClassifier(**LGB_CLF_PARAMS)
            model.fit(X[tr], y_tr)
            pred[va] = model.predict(X[va])
        oof[f"{iso}_pred"] = pred
        final = LGBMClassifier(**LGB_CLF_PARAMS)
        final.fit(X[mask], df.loc[mask, col].astype(bool).to_numpy())
        test_out[f"{iso}_pred"] = final.predict(X_test).astype(bool)
        m = mask & ~np.isnan(pred)
        scores[iso] = float(matthews_corrcoef(df.loc[m, col].astype(bool), pred[m].astype(bool)))
        print(f"lgb_tdi/{iso}: MCC={scores[iso]:.4f}")
    scores["MA"] = float(np.mean([scores[iso] for iso in TDI_ISOFORMS]))
    print(f"lgb_tdi MA-MCC={scores['MA']:.4f}")

    oof.to_csv(RESULTS_DIR / "lgb_tdi_oof.csv", index=False)
    test_out.to_csv(RESULTS_DIR / "lgb_tdi_test.csv", index=False)
    (RESULTS_DIR / "lgb_tdi_scores.json").write_text(json.dumps(scores, indent=2))


def main() -> None:
    ensure_dirs()
    run_regression()
    run_tdi()


if __name__ == "__main__":
    main()
