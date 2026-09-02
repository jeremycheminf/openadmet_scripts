"""Multitask ChemProp fine-tune, warm-started from chemprop_medium: 4 primary
(challenge pIC50) + 4 log2fc (real single-concentration screening data, same
compounds) + 4 ChEMBL pIC50 + 5 PubChem-qHTS pIC50-like heads = 17 heads. Primary
heads get inverse-count weights (mean-normalized to 1.0); every auxiliary head gets
a flat 0.3x that scale. This is the single largest ensemble contributor in our own
runs (see README) -- even though its raw calibrated score is unremarkable, it's
consistently the best chemprop-family *ranker*.

Run from the repo root:  python scripts/05_finetune_chemprop_multitask.py
"""

import json

import numpy as np
import pandas as pd
import torch
from cyp_submission.chemprop_transfer import compute_inverse_count_task_weights, fit_finetune, predict_finetune
from cyp_submission.data import HF_DIR, ISOFORMS, load_test_blinded, load_train_inhibition
from cyp_submission.metrics import isoform_st_rae
from cyp_submission.paths import CHECKPOINTS_DIR, DATA_EXTERNAL, DATA_INTERIM, RESULTS_DIR, ensure_dirs
from cyp_submission.splits import butina_kfold
from sklearn.metrics import mean_absolute_error, r2_score

N_FOLDS = 5
NUM_EPOCHS = 60
BATCH_SIZE = 256
AUX_WEIGHT = 0.3
PRETRAINED = CHECKPOINTS_DIR / "chemprop_medium.pt"
CHEMBL_CYP_CSV = DATA_EXTERNAL / "cyp_chembl.csv"
PUBCHEM_QHTS_CSV = DATA_EXTERNAL / "pubchem_cyp_qhts_aid1851.csv"

PRIMARY_COLS = [f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_COLS = [f"{iso}_log2fc" for iso in ISOFORMS]
CHEMBL_COLS = [f"{iso}_ChEMBL_PIC50" for iso in ISOFORMS]
PUBCHEM_ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4", "CYP2C19"]
PUBCHEM_COLS = [f"{iso}_PubChem_qHTS" for iso in PUBCHEM_ISOS]
TARGET_COLS = PRIMARY_COLS + LOG2FC_COLS + CHEMBL_COLS + PUBCHEM_COLS
N_AUX = len(LOG2FC_COLS) + len(CHEMBL_COLS) + len(PUBCHEM_COLS)


def build_challenge_df() -> pd.DataFrame:
    df = load_train_inhibition()
    sc = pd.read_csv(HF_DIR / "cyp-challenge-single-concentration-TRAIN.csv")
    wide = sc.pivot_table(index="Molecule_Name", columns="enzyme", values="log2fc_estimate")
    wide.columns = [f"{c}_log2fc" for c in wide.columns]
    df = df.merge(wide, on="Molecule_Name", how="left")
    for c in CHEMBL_COLS + PUBCHEM_COLS:
        df[c] = np.nan
    return df


def build_chembl_df(exclude_smiles: set[str]) -> pd.DataFrame:
    chembl = pd.read_csv(CHEMBL_CYP_CSV)
    chembl = chembl[~chembl["smiles"].isin(exclude_smiles)].reset_index(drop=True)
    df = pd.DataFrame({"SMILES": chembl["smiles"], "Molecule_Name": [f"CHEMBL_{i}" for i in range(len(chembl))]})
    for iso, col in zip(ISOFORMS, CHEMBL_COLS):
        df[col] = chembl[f"{iso}_IC50_PIC50"]
    for c in PRIMARY_COLS + LOG2FC_COLS + PUBCHEM_COLS:
        df[c] = np.nan
    return df[df[CHEMBL_COLS].notna().any(axis=1)].reset_index(drop=True)


def build_pubchem_df(exclude_smiles: set[str]) -> pd.DataFrame:
    pc = pd.read_csv(PUBCHEM_QHTS_CSV)
    pc = pc[~pc["smiles"].isin(exclude_smiles)].reset_index(drop=True)
    df = pd.DataFrame({"SMILES": pc["smiles"], "Molecule_Name": [f"PUBCHEM_{i}" for i in range(len(pc))]})
    for iso, col in zip(PUBCHEM_ISOS, PUBCHEM_COLS):
        pcol = f"{iso}_pIC50"
        df[col] = pc[pcol] if pcol in pc.columns else np.nan
    for c in PRIMARY_COLS + LOG2FC_COLS + CHEMBL_COLS:
        df[c] = np.nan
    return df[df[PUBCHEM_COLS].notna().any(axis=1)].reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    challenge_df = build_challenge_df()
    excl = set(challenge_df["SMILES"])
    chembl_df = build_chembl_df(exclude_smiles=excl)
    pubchem_df = build_pubchem_df(exclude_smiles=excl | set(chembl_df["SMILES"]))
    aux_df = pd.concat([chembl_df, pubchem_df], ignore_index=True)

    primary_weights = compute_inverse_count_task_weights(challenge_df, PRIMARY_COLS)
    task_weights = primary_weights + [AUX_WEIGHT] * N_AUX
    print(f"challenge rows: {len(challenge_df)}  aux rows: {len(aux_df)} "
          f"(chembl={len(chembl_df)}, pubchem={len(pubchem_df)})")

    folds_path = DATA_INTERIM / "train_inhibition_folds.npy"
    folds = np.load(folds_path) if folds_path.exists() else butina_kfold(
        challenge_df["SMILES"].tolist(), n_folds=N_FOLDS)

    oof = pd.DataFrame({"Molecule_Name": challenge_df["Molecule_Name"], "SMILES": challenge_df["SMILES"]})
    oof_preds = {iso: np.full(len(challenge_df), np.nan) for iso in ISOFORMS}

    for f in range(N_FOLDS):
        val_fold = (f + 1) % N_FOLDS
        train_mask = ~np.isin(folds, [f, val_fold])
        val_mask = folds == val_fold
        test_mask = folds == f
        train_df = pd.concat([challenge_df.loc[train_mask], aux_df], ignore_index=True)
        val_df = challenge_df.loc[val_mask].reset_index(drop=True)
        test_df = challenge_df.loc[test_mask].reset_index(drop=True)
        print(f"fold {f}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        model, _ = fit_finetune(train_df, val_df, TARGET_COLS, PRETRAINED, task_weights=task_weights,
                                 num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, accelerator="gpu")
        preds = predict_finetune(model, test_df, TARGET_COLS)
        for iso, col in zip(ISOFORMS, PRIMARY_COLS):
            oof_preds[iso][np.flatnonzero(test_mask)] = preds[col].to_numpy()

    scores = {}
    for iso, col in zip(ISOFORMS, PRIMARY_COLS):
        oof[f"{iso}_pred"] = oof_preds[iso]
        y = challenge_df[col].to_numpy()
        pred = oof_preds[iso]
        mask = ~np.isnan(y) & ~np.isnan(pred)
        st_rae = isoform_st_rae(y, pred, challenge_df[f"{col}_conf_low"].to_numpy(),
                                 challenge_df[f"{col}_conf_high"].to_numpy())
        scores[iso] = {"ST_RAE": st_rae, "MAE": mean_absolute_error(y[mask], pred[mask]),
                        "R2": r2_score(y[mask], pred[mask])}
        print(f"{iso}: {scores[iso]}")
    scores["MA"] = {k: float(np.mean([scores[iso][k] for iso in ISOFORMS])) for k in ("ST_RAE", "MAE", "R2")}
    print(f"MA-ST-RAE={scores['MA']['ST_RAE']:.4f}")

    oof.to_csv(RESULTS_DIR / "chemprop_multitask_oof.csv", index=False)
    (RESULTS_DIR / "chemprop_multitask_scores.json").write_text(json.dumps(scores, indent=2))

    final_train = pd.concat([challenge_df.loc[folds != N_FOLDS - 1], aux_df], ignore_index=True)
    final_val = challenge_df.loc[folds == N_FOLDS - 1].reset_index(drop=True)
    final_model, _ = fit_finetune(final_train, final_val, TARGET_COLS, PRETRAINED, task_weights=task_weights,
                                   num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, accelerator="gpu")
    torch.save(final_model, RESULTS_DIR / "chemprop_multitask_final_model.pt")

    test_df = load_test_blinded()
    test_preds = predict_finetune(final_model, test_df, TARGET_COLS)
    test_out = pd.concat(
        [test_df[["Molecule_Name", "SMILES"]].reset_index(drop=True),
         test_preds[PRIMARY_COLS].rename(columns=dict(zip(PRIMARY_COLS, [f"{iso}_pred" for iso in ISOFORMS])))],
        axis=1)
    test_out.to_csv(RESULTS_DIR / "chemprop_multitask_test.csv", index=False)
    print("wrote chemprop_multitask_{oof,test}.csv")


if __name__ == "__main__":
    main()
