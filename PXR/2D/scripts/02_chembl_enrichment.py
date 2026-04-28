"""
Fetch PXR (CHEMBL3401) EC50 activation data from ChEMBL,
convert to pEC50, deduplicate against curated train set, and save.

Run AFTER 01_eda_curation.ipynb (needs train_curated.csv).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import DATA_DIR, COL_PECSO, remove_salts, smiles_to_mol

CHEMBL_TARGET = "CHEMBL3401"
OUTPUT_FILE = DATA_DIR / "chembl_enrichment.csv"
FINAL_FILE  = DATA_DIR / "train_final.csv"

ACTIVATION_KEYWORDS  = re.compile(r"agonist|activat|induc|stimulat", re.I)
ANTAGONIST_KEYWORDS  = re.compile(r"antagonist|inhibit|block|suppress|reduce", re.I)


def fetch_chembl_activities(target_id: str) -> pd.DataFrame:
    from chembl_webresource_client.new_client import new_client
    activity_client = new_client.activity

    print(f"Querying ChEMBL for target {target_id}, standard_type=EC50 ...")
    results = list(activity_client.filter(
        target_chembl_id=target_id,
        standard_type="EC50",
    ))
    print(f"  Raw results: {len(results)}")
    return pd.DataFrame(results)


def filter_activation(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only activation/agonism records; remove antagonism."""
    if "activity_comment" not in df.columns:
        return df

    def classify(comment):
        if pd.isna(comment) or comment == "":
            return "unspecified"
        if ANTAGONIST_KEYWORDS.search(str(comment)):
            return "antagonist"
        if ACTIVATION_KEYWORDS.search(str(comment)):
            return "agonist"
        return "unspecified"

    df["activity_class"] = df["activity_comment"].apply(classify)
    before = len(df)
    # Remove explicit antagonists; keep agonists and unspecified
    df = df[df["activity_class"] != "antagonist"].copy()
    print(f"  Removed {before - len(df)} antagonist/inhibitor records")
    return df


def to_pecso(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard_value (nM) → pEC50 = -log10(value_M)."""
    df = df[
        (df["standard_relation"] == "=") &
        (df["standard_units"].isin(["nM", "uM", "mM", "M"]))
    ].copy()

    unit_to_M = {"nM": 1e-9, "uM": 1e-6, "mM": 1e-3, "M": 1.0}
    df["value_M"] = df.apply(
        lambda r: float(r["standard_value"]) * unit_to_M.get(r["standard_units"], np.nan),
        axis=1
    )
    df = df[df["value_M"] > 0].copy()
    df[COL_PECSO] = -np.log10(df["value_M"])
    df = df[(df[COL_PECSO] >= 1.0) & (df[COL_PECSO] <= 9.5)].copy()
    print(f"  After unit/range filter: {len(df)} records")
    return df


def standardise_smiles(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """Standardise SMILES and compute InChIKey."""
    rows = []
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, "") or "")
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        mol = remove_salts(mol)
        if mol.GetNumAtoms() == 0:
            continue
        try:
            inchi = Chem.MolToInchi(mol)
            if inchi is None:
                continue
            ik = Chem.InchiToInchiKey(inchi)
        except Exception:
            continue
        rows.append({
            "SMILES": Chem.MolToSmiles(mol),
            "inchikey": ik,
            COL_PECSO: row[COL_PECSO],
            "chembl_id": row.get("molecule_chembl_id", ""),
            "assay_id": row.get("assay_chembl_id", ""),
            "activity_class": row.get("activity_class", "unspecified"),
        })
    return pd.DataFrame(rows)


def deduplicate(chembl_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Remove ChEMBL molecules already in train (by InChIKey)."""
    train_keys = set(train_df["inchikey"].dropna())
    before = len(chembl_df)
    chembl_df = chembl_df[~chembl_df["inchikey"].isin(train_keys)].copy()
    print(f"  Removed {before - len(chembl_df)} duplicates already in curated train")

    # Also deduplicate within ChEMBL itself: keep median pEC50 per InChIKey
    chembl_df = (
        chembl_df.groupby("inchikey")
        .agg(SMILES=("SMILES", "first"), pEC50=(COL_PECSO, "median"),
             chembl_id=("chembl_id", "first"), n_records=("pEC50", "count"))
        .reset_index()
    )
    chembl_df["source"] = "chembl"
    print(f"  Unique novel ChEMBL compounds: {len(chembl_df)}")
    return chembl_df


def main():
    if not (DATA_DIR / "train_curated.csv").exists():
        print("ERROR: train_curated.csv not found. Run 01_eda_curation.ipynb first.")
        sys.exit(1)

    train_curated = pd.read_csv(DATA_DIR / "train_curated.csv")
    print(f"Loaded train_curated.csv: {len(train_curated)} rows")

    # --- Fetch from ChEMBL --------------------------------------------------
    df_raw = fetch_chembl_activities(CHEMBL_TARGET)
    if df_raw.empty:
        print("No results from ChEMBL. Skipping enrichment.")
        train_curated["source"] = "train"
        train_curated.to_csv(FINAL_FILE, index=False)
        print(f"Saved {FINAL_FILE} (no enrichment)")
        return

    # --- Assay type filter (binding=B, functional=F) ------------------------
    if "assay_type" in df_raw.columns:
        df_raw = df_raw[df_raw["assay_type"].isin(["B", "F"])].copy()
        print(f"  After assay_type B/F filter: {len(df_raw)}")

    df_raw = filter_activation(df_raw)
    df_raw = to_pecso(df_raw)

    if df_raw.empty:
        print("No valid EC50 activation records found.")
        train_curated["source"] = "train"
        train_curated.to_csv(FINAL_FILE, index=False)
        return

    # --- Standardise SMILES -------------------------------------------------
    print("Standardising SMILES ...")
    df_std = standardise_smiles(df_raw)
    print(f"  After SMILES standardisation: {len(df_std)}")

    # --- Deduplicate --------------------------------------------------------
    df_novel = deduplicate(df_std, train_curated)
    df_novel.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE}: {len(df_novel)} novel compounds")

    # --- pEC50 statistics ---------------------------------------------------
    print(f"  ChEMBL pEC50 range: [{df_novel[COL_PECSO].min():.2f}, {df_novel[COL_PECSO].max():.2f}]")
    print(f"  median={df_novel[COL_PECSO].median():.2f}")

    # --- Merge with curated train -------------------------------------------
    train_out = train_curated[["Molecule Name", "SMILES", COL_PECSO, "inchikey"]].copy()
    train_out["source"] = "train"
    train_out["n_records"] = 1

    chembl_out = df_novel[["SMILES", COL_PECSO, "inchikey", "source", "n_records"]].copy()
    chembl_out["Molecule Name"] = chembl_out["inchikey"]

    combined = pd.concat([train_out, chembl_out], ignore_index=True)
    combined.to_csv(FINAL_FILE, index=False)
    print(f"\nSaved train_final.csv: {len(combined)} rows "
          f"({len(train_out)} train + {len(chembl_out)} ChEMBL)")


if __name__ == "__main__":
    main()
