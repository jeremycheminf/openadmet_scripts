"""
Compute all descriptor sets for train_final + test and save to features/.
Run AFTER 02_chembl_enrichment.py (needs train_final.csv and test_curated.csv).

Feature sets produced:
  ecfp4           - Morgan r=2, 2048-bit binary fingerprints
  fcfp4           - Feature-based Morgan r=2, 2048-bit binary fingerprints
  rdkit2d         - ~200 RDKit 2D descriptors (scaled)
  mordred2d       - ~1600 Mordred 2D descriptors (scaled)
  mordred3d       - ~200 Mordred 3D descriptors (from cached ETKDG conformers)
  ecfp4_rdkit     - concat(ecfp4, rdkit2d)          — primary gradient-boosting input
  ecfp4_mordred   - concat(ecfp4, mordred2d)         — richer 2D input
  ecfp4_mordred3d - concat(ecfp4, mordred2d+mordred3d) — full 2D+3D input

Conformers are cached to features/conformers_{split}.sdf so re-runs skip generation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils import (
    DATA_DIR, FEATURE_DIR, COL_PECSO,
    clean_descriptor_matrix, ecfp4_array, fcfp4_array,
    mordred2d_array, rdkit2d_array, save_features, smiles_to_mol,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------
# 3D conformer helpers — ETKDG + optional MMFF, cached to SDF
# -----------------------------------------------------------------------

def generate_conformers_cached(
    mols: list,
    split_name: str,
    optimize: bool = True,
    max_attempts: int = 3,
) -> list:
    """
    Generate or load ETKDG conformers.  Results are cached to
    features/conformers_{split_name}.sdf so this runs only once.
    Returns list of mols with conformers (None where generation failed).
    """
    cache_path = FEATURE_DIR / f"conformers_{split_name}.sdf"

    if cache_path.exists():
        print(f"   Loading cached conformers from {cache_path.name} ...")
        writer_mols = []
        suppl = Chem.SDMolSupplier(str(cache_path), removeHs=False)
        for m in suppl:
            writer_mols.append(m)  # None if parsing failed
        if len(writer_mols) == len(mols):
            return writer_mols
        print(f"   Cache size mismatch ({len(writer_mols)} vs {len(mols)}), regenerating ...")

    print(f"   Generating ETKDG conformers for {len(mols)} molecules "
          f"({'+ MMFF opt' if optimize else 'geometry only'}) ...")
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # use all CPUs

    conf_mols = []
    failed = 0
    for i, mol in enumerate(mols):
        if mol is None:
            conf_mols.append(None)
            failed += 1
            continue
        mol_h = Chem.AddHs(mol)
        ok = False
        for _ in range(max_attempts):
            res = AllChem.EmbedMolecule(mol_h, params)
            if res == 0:
                ok = True
                break
        if not ok:
            conf_mols.append(None)
            failed += 1
            continue
        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass  # keep unoptimised geometry
        conf_mols.append(mol_h)
        if (i + 1) % 500 == 0:
            print(f"   ... {i + 1}/{len(mols)} done, {failed} failed so far")

    print(f"   Conformer generation done. Failed: {failed}/{len(mols)}")

    # Write to SDF cache (write None slots as empty mol to preserve index)
    writer = Chem.SDWriter(str(cache_path))
    empty_mol = Chem.RWMol()
    for m in conf_mols:
        writer.write(m if m is not None else empty_mol)
    writer.close()
    print(f"   Cached to {cache_path.name}")
    return conf_mols


def rdkit3d_pharm_array(mols_3d: list) -> tuple[np.ndarray, list[str]]:
    """
    Alignment-free 3D pharmacophore descriptors from RDKit:
    WHIM (114), GETAWAY (273), RDF (210), MORSE (224), AUTOCORR3D (80).
    Requires molecules with conformers (from generate_conformers_cached).
    """
    from rdkit.Chem import rdMolDescriptors as rmd

    def desc_for_mol(mol):
        if mol is None or mol.GetNumConformers() == 0:
            return None
        try:
            whim     = list(rmd.CalcWHIM(mol))
            getaway  = list(rmd.CalcGETAWAY(mol))
            rdf      = list(rmd.CalcRDF(mol))
            morse    = list(rmd.CalcMORSE(mol))
            autocorr = list(rmd.CalcAUTOCORR3D(mol))
            return whim + getaway + rdf + morse + autocorr
        except Exception:
            return None

    # Build names once from a valid molecule
    names = None
    for mol in mols_3d:
        if mol is not None and mol.GetNumConformers() > 0:
            try:
                names = (
                    [f"WHIM_{i}"     for i in range(len(rmd.CalcWHIM(mol)))]
                  + [f"GETAWAY_{i}"  for i in range(len(rmd.CalcGETAWAY(mol)))]
                  + [f"RDF_{i}"      for i in range(len(rmd.CalcRDF(mol)))]
                  + [f"MORSE_{i}"    for i in range(len(rmd.CalcMORSE(mol)))]
                  + [f"AUTOCORR3D_{i}" for i in range(len(rmd.CalcAUTOCORR3D(mol)))]
                )
                break
            except Exception:
                continue

    if names is None:
        raise RuntimeError("No valid 3D molecules to derive descriptor names from")

    rows = [desc_for_mol(m) for m in mols_3d]
    arr = np.full((len(mols_3d), len(names)), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            arr[i] = row
    return arr, names


def mordred3d_array(mols_3d: list) -> tuple[np.ndarray, list[str]]:
    """Compute Mordred 3D descriptors from molecules with conformers."""
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=False)
    # Filter to only 3D-dependent descriptors for speed
    calc_3d = Calculator([d for d in calc.descriptors
                          if getattr(d, 'require_3D', False)], ignore_3D=False)
    if len(calc_3d.descriptors) == 0:
        # Fallback: compute all and note it
        calc_3d = Calculator(mdesc, ignore_3D=False)

    rows = []
    for mol in mols_3d:
        if mol is None or mol.GetNumConformers() == 0:
            # placeholder row — will be NaN-dropped later
            rows.append(None)
        else:
            try:
                result = calc_3d(mol)
                rows.append([v if isinstance(v, (int, float)) else np.nan
                              for v in result.values()])
            except Exception:
                rows.append(None)

    names = [str(d) for d in calc_3d.descriptors]
    n_desc = len(names)
    arr = np.full((len(mols_3d), n_desc), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            arr[i] = row
    return arr, names


def load_datasets():
    # Prefer train_final (with ChEMBL); fall back to train_curated
    train_path = DATA_DIR / "train_final.csv"
    if not train_path.exists():
        train_path = DATA_DIR / "train_curated.csv"
        print(f"train_final.csv not found, using {train_path.name}")
    test_path = DATA_DIR / "test_curated.csv"
    if not test_path.exists():
        # fall back to raw test
        test_path = DATA_DIR / "test_raw.csv"
        print(f"test_curated.csv not found, using test_raw.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Use canonical SMILES if available, else raw SMILES
    train_smiles = train_df.get("SMILES", train_df.get("canonical_smiles")).tolist()
    test_smiles  = test_df.get("SMILES", test_df.get("canonical_smiles")).tolist()

    print(f"Train: {len(train_smiles)} molecules")
    print(f"Test : {len(test_smiles)}  molecules")
    return train_smiles, test_smiles, train_df, test_df


def smiles_to_mols(smiles_list: list[str]):
    mols = []
    failed = 0
    for s in smiles_list:
        m = smiles_to_mol(str(s))
        mols.append(m)
        if m is None:
            failed += 1
    if failed:
        print(f"  WARNING: {failed} SMILES failed to parse (will be all-zero row)")
    return mols


def main():
    FEATURE_DIR.mkdir(exist_ok=True)
    print("=== Feature Generation ===\n")

    train_smiles, test_smiles, train_df, test_df = load_datasets()
    train_mols = smiles_to_mols(train_smiles)
    test_mols  = smiles_to_mols(test_smiles)

    # ------------------------------------------------------------------
    # 1. ECFP4 (Morgan r=2, 2048-bit binary)
    # ------------------------------------------------------------------
    print("1. ECFP4 fingerprints ...")
    ecfp4_tr = ecfp4_array(train_mols, n_bits=2048)
    ecfp4_te = ecfp4_array(test_mols,  n_bits=2048)
    save_features("ecfp4", ecfp4_tr, ecfp4_te)

    # ------------------------------------------------------------------
    # 2. FCFP4 (feature-based Morgan r=2, 2048-bit)
    # ------------------------------------------------------------------
    print("2. FCFP4 fingerprints ...")
    fcfp4_tr = fcfp4_array(train_mols, n_bits=2048)
    fcfp4_te = fcfp4_array(test_mols,  n_bits=2048)
    save_features("fcfp4", fcfp4_tr, fcfp4_te)

    # ------------------------------------------------------------------
    # 3. RDKit 2D descriptors (~200 features)
    # ------------------------------------------------------------------
    print("3. RDKit 2D descriptors ...")
    rdkit_tr_raw, rdkit_names = rdkit2d_array(train_mols)
    rdkit_te_raw, _           = rdkit2d_array(test_mols)
    rdkit_tr, rdkit_te, rdkit_names_clean, rdkit_scaler = clean_descriptor_matrix(
        rdkit_tr_raw, rdkit_te_raw, rdkit_names
    )
    save_features("rdkit2d", rdkit_tr, rdkit_te, rdkit_scaler)
    print(f"   RDKit 2D: {rdkit_tr.shape[1]} clean features (of {len(rdkit_names)})")

    # ------------------------------------------------------------------
    # 4. Mordred 2D descriptors (~1600 features)
    # ------------------------------------------------------------------
    print("4. Mordred 2D descriptors (this may take a few minutes) ...")
    try:
        mordred_tr_raw, mordred_names = mordred2d_array(train_mols)
        mordred_te_raw, _             = mordred2d_array(test_mols)
        mordred_tr, mordred_te, mordred_names_clean, mordred_scaler = clean_descriptor_matrix(
            mordred_tr_raw, mordred_te_raw, mordred_names
        )
        save_features("mordred2d", mordred_tr, mordred_te, mordred_scaler)
        print(f"   Mordred 2D: {mordred_tr.shape[1]} clean features (of {len(mordred_names)})")
        mordred_ok = True
    except Exception as e:
        print(f"   WARNING: Mordred failed ({e}). Skipping mordred2d.")
        mordred_ok = False

    # ------------------------------------------------------------------
    # 5. 3D conformers (ETKDG+MMFF, cached to SDF) — shared by 3D feature sets
    # ------------------------------------------------------------------
    print("5. Generating/loading 3D conformers (ETKDG+MMFF, cached to SDF) ...")
    train_mols_3d = generate_conformers_cached(train_mols, "train", optimize=True)
    test_mols_3d  = generate_conformers_cached(test_mols,  "test",  optimize=True)
    n_train_conf = sum(1 for m in train_mols_3d if m is not None and m.GetNumConformers() > 0)
    n_test_conf  = sum(1 for m in test_mols_3d  if m is not None and m.GetNumConformers() > 0)
    print(f"   Conformers OK: train={n_train_conf}/{len(train_mols_3d)}  "
          f"test={n_test_conf}/{len(test_mols_3d)}")

    # ------------------------------------------------------------------
    # 5a. RDKit 3D pharmacophore descriptors (WHIM + GETAWAY + RDF + MORSE + AUTOCORR3D)
    #     — these are alignment-free, work with a single conformer per molecule
    # ------------------------------------------------------------------
    print("5a. RDKit 3D pharmacophore descriptors (WHIM, GETAWAY, RDF, MORSE, AUTOCORR3D) ...")
    try:
        rdkit3d_tr_raw, rdkit3d_names = rdkit3d_pharm_array(train_mols_3d)
        rdkit3d_te_raw, _             = rdkit3d_pharm_array(test_mols_3d)
        rdkit3d_tr, rdkit3d_te, _, rdkit3d_scaler = clean_descriptor_matrix(
            rdkit3d_tr_raw, rdkit3d_te_raw, rdkit3d_names
        )
        save_features("rdkit3d_pharm", rdkit3d_tr, rdkit3d_te, rdkit3d_scaler)
        print(f"   RDKit 3D pharm: {rdkit3d_tr.shape[1]} features (of {len(rdkit3d_names)})")
        rdkit3d_ok = True
    except Exception as e:
        print(f"   WARNING: RDKit 3D pharm failed ({e}).")
        rdkit3d_ok = False

    # ------------------------------------------------------------------
    # 5b. Mordred 3D descriptors (from cached conformers)
    # ------------------------------------------------------------------
    print("5b. Mordred 3D descriptors (from cached conformers) ...")
    mordred3d_ok = False
    try:
        m3d_tr_raw, m3d_names = mordred3d_array(train_mols_3d)
        m3d_te_raw, _         = mordred3d_array(test_mols_3d)
        mordred3d_tr, mordred3d_te, _, m3d_scaler = clean_descriptor_matrix(
            m3d_tr_raw, m3d_te_raw, m3d_names
        )
        save_features("mordred3d", mordred3d_tr, mordred3d_te, m3d_scaler)
        print(f"   Mordred 3D: {mordred3d_tr.shape[1]} clean features (of {len(m3d_names)})")
        mordred3d_ok = True
    except Exception as e:
        print(f"   WARNING: Mordred 3D failed ({e}). Skipping.")

    # ------------------------------------------------------------------
    # 6. Combined feature sets
    # ------------------------------------------------------------------
    print("6. Combined feature sets ...")

    ecfp4_rdkit_tr = np.hstack([ecfp4_tr, rdkit_tr])
    ecfp4_rdkit_te = np.hstack([ecfp4_te, rdkit_te])
    save_features("ecfp4_rdkit", ecfp4_rdkit_tr, ecfp4_rdkit_te)

    if mordred_ok:
        ecfp4_mordred_tr = np.hstack([ecfp4_tr, mordred_tr])
        ecfp4_mordred_te = np.hstack([ecfp4_te, mordred_te])
        save_features("ecfp4_mordred", ecfp4_mordred_tr, ecfp4_mordred_te)

        if mordred3d_ok:
            ecfp4_m2d3d_tr = np.hstack([ecfp4_tr, mordred_tr, mordred3d_tr])
            ecfp4_m2d3d_te = np.hstack([ecfp4_te, mordred_te, mordred3d_te])
            save_features("ecfp4_mordred3d", ecfp4_m2d3d_tr, ecfp4_m2d3d_te)

    if rdkit3d_ok:
        # 3D QSAR: ECFP4 + RDKit 2D + 3D pharmacophore descriptors
        ecfp4_3dqsar_tr = np.hstack([ecfp4_tr, rdkit_tr, rdkit3d_tr])
        ecfp4_3dqsar_te = np.hstack([ecfp4_te, rdkit_te, rdkit3d_te])
        save_features("ecfp4_rdkit_3dqsar", ecfp4_3dqsar_tr, ecfp4_3dqsar_te)

    # ------------------------------------------------------------------
    # 7. Save molecule index files alongside features
    # ------------------------------------------------------------------
    train_df[["Molecule Name"]].to_csv(FEATURE_DIR / "train_index.csv", index=False)
    test_df[["Molecule Name"]].to_csv(FEATURE_DIR / "test_index.csv",  index=False)

    y_train = train_df[COL_PECSO].values.astype(np.float64)
    np.save(FEATURE_DIR / "y_train.npy", y_train)
    print(f"\ny_train shape: {y_train.shape},  range=[{y_train.min():.2f}, {y_train.max():.2f}]")

    print("\n=== Feature generation complete ===")
    print(f"All features saved to {FEATURE_DIR}/")


if __name__ == "__main__":
    main()
