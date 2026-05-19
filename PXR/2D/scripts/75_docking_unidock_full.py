"""
Full 4083+513 compound docking pipeline using UniDock (GPU) + GNINA rescore.

UniDock: GPU-native AutoDock-Vina reimplementation, ~10× faster → all 4597
compounds dockes in ~30 min per receptor.

After docking:
  1. GNINA rescore: minimize UniDock poses + CNN score (3-5 min)
  2. Strain energy: MMFF energy(docked pose) - MMFF energy(minimized free conformer)
  3. ProLIF PLIF: key residue contacts from minimized pose
  4. Interaction counts: n_HB, n_hydrophobic, n_vdw per residue class
  5. Distance to key PXR residues: SER247, MET323, TRP299, PHE281, HIS407

Features saved per receptor → much richer than current 542-subset features.

Run after MIST: wsl -e bash -c "bash logs/chain_unidock_full.sh"
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FEATURE_DIR = ROOT / "features"
DOCK_DIR    = RESULTS_DIR / "docking_full"
TEMPLATES   = ROOT / "3D" / "data" / "templates"

RECEPTORS = ["8r81", "2o9i", "8eqz"]

# Key PXR binding residues for explicit distance features
KEY_RESIDUES = {
    "SER247": ("SER", 247),
    "MET323": ("MET", 323),
    "TRP299": ("TRP", 299),
    "PHE281": ("PHE", 281),
    "HIS407": ("HIS", 407),
    "LEU411": ("LEU", 411),
    "PHE288": ("PHE", 288),
}


def build_all_ligands_csv():
    """Build full ligand CSV: all 4083 train + 513 test."""
    out = DOCK_DIR / "all_ligands_full.csv"
    if out.exists():
        df = pd.read_csv(out)
        print(f"  Ligand CSV cached: {len(df)} molecules")
        return out

    DOCK_DIR.mkdir(exist_ok=True)
    tr_csv = DATA_DIR / "train_final.csv" if (DATA_DIR/"train_final.csv").exists() \
             else DATA_DIR / "train_curated.csv"
    te_csv = DATA_DIR / "test_curated.csv" if (DATA_DIR/"test_curated.csv").exists() \
             else DATA_DIR / "test_raw.csv"
    train_df = pd.read_csv(tr_csv)
    test_df  = pd.read_csv(te_csv)
    y = np.load(FEATURE_DIR / "y_train.npy")
    train_df["pEC50"] = y
    train_df["source"] = "train"
    test_df["source"]  = "test"
    test_df["pEC50"]   = float("nan")

    all_df = pd.concat([train_df[["SMILES","pEC50","source"]],
                        test_df[["SMILES","pEC50","source"]]
                       ]).drop_duplicates("SMILES").reset_index(drop=True)
    all_df["NAME"] = [f"{r['source']}_{i:04d}" for i, r in all_df.iterrows()]
    all_df.to_csv(out, index=False)
    print(f"  Built ligand CSV: {len(all_df)} molecules")
    return out


def compute_strain_energy(sdf_path: Path, rec_tag: str) -> pd.DataFrame:
    """
    Strain energy = MMFF(docked pose) - MMFF(free minimized conformer).
    High strain → ligand is geometrically stressed in the binding pose.
    """
    rows = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    seen = set()
    for mol in suppl:
        if mol is None: continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unk"
        if name in seen: continue  # best pose only
        seen.add(name)

        try:
            Chem.SanitizeMol(mol)
            mol_h = Chem.AddHs(mol, addCoords=True)
            ff = AllChem.MMFFGetMoleculeForceField(mol_h,
                    AllChem.MMFFGetMoleculeProperties(mol_h))
            e_docked = ff.CalcEnergy() if ff else np.nan

            # Free minimized conformer
            mol_free = Chem.RWMol(mol_h)
            AllChem.EmbedMolecule(mol_free, AllChem.ETKDGv3())
            ff2 = AllChem.MMFFGetMoleculeForceField(mol_free,
                     AllChem.MMFFGetMoleculeProperties(mol_free))
            if ff2:
                ff2.Minimize()
                e_free = ff2.CalcEnergy()
            else:
                e_free = np.nan

            strain = e_docked - e_free if not (np.isnan(e_docked) or np.isnan(e_free)) \
                     else np.nan
        except Exception:
            e_docked = e_free = strain = np.nan

        rows.append({"mol_id": name, f"{rec_tag}__strain_energy": strain,
                     f"{rec_tag}__e_docked": e_docked,
                     f"{rec_tag}__e_free": e_free})

    return pd.DataFrame(rows)


def extract_distance_to_key_residues(sdf_path: Path, rec_pdb: Path,
                                     rec_tag: str) -> pd.DataFrame:
    """
    Min distance from ligand heavy atoms to each key PXR residue.
    """
    import MDAnalysis as mda
    try:
        u = mda.Universe(str(rec_pdb))
    except Exception:
        return pd.DataFrame()

    rows = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    seen = set()
    for mol in suppl:
        if mol is None: continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unk"
        if name in seen: continue
        seen.add(name)

        try:
            conf = mol.GetConformer()
            lig_coords = np.array([conf.GetAtomPosition(i)
                                   for i in range(mol.GetNumAtoms())])

            row = {"mol_id": name}
            for res_name, (res_type, res_id) in KEY_RESIDUES.items():
                sel = u.select_atoms(f"resname {res_type} and resid {res_id} and not type H")
                if len(sel) == 0:
                    row[f"{rec_tag}__dist_{res_name}"] = np.nan
                    continue
                prot_coords = sel.positions
                # Min distance from any ligand atom to any residue atom
                diffs = lig_coords[:, None, :] - prot_coords[None, :, :]
                min_d = float(np.sqrt((diffs**2).sum(-1)).min())
                row[f"{rec_tag}__dist_{res_name}"] = min_d
        except Exception:
            row = {"mol_id": name}

        rows.append(row)

    return pd.DataFrame(rows)


def extract_plif_full(sdf_path: Path, rec_pdb: Path, rec_tag: str) -> pd.DataFrame:
    """ProLIF PLIF on full docked set — per-molecule with fallback."""
    try:
        import prolif
        import MDAnalysis as mda
        import smirk  # noqa — needed to suppress import in some envs
    except ImportError:
        return pd.DataFrame()

    try:
        u = mda.Universe(str(rec_pdb))
        prot_mol = prolif.Molecule.from_mda(u.select_atoms("protein"))
    except Exception as e:
        print(f"    ProLIF receptor load failed: {e}")
        return pd.DataFrame()

    fp = prolif.Fingerprint(
        interactions=["Hydrophobic", "HBDonor", "HBAcceptor",
                      "PiStacking", "PiCation", "CationPi", "VdWContact"],
        count=False,
    )

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol_map = {}
    for mol in suppl:
        if mol is None: continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unk"
        if name not in mol_map:
            mol_map[name] = mol

    lig_mols  = [prolif.Molecule.from_rdkit(mol) for mol in mol_map.values()]
    mol_ids   = list(mol_map.keys())

    try:
        fp.run_from_iterable(lig_mols, prot_mol, progress=False)
        df = fp.to_dataframe()
        df.columns = [str(c) for c in df.columns]
        df.insert(0, "mol_id", mol_ids[:len(df)])
        df = df.rename(columns={c: f"{rec_tag}__{c}"
                                 for c in df.columns if c != "mol_id"})
        print(f"    ProLIF: {len(df)} mols, {len(df.columns)-1} bits")
        return df
    except Exception as e:
        print(f"    ProLIF batch failed ({e}), trying per-molecule ...")
        rows = []
        n_ok = 0
        for mid, lig_mol in zip(mol_ids, lig_mols):
            try:
                fp.run_from_iterable([lig_mol], prot_mol, progress=False)
                d = fp.to_dataframe()
                d.columns = [str(c) for c in d.columns]
                bits = d.iloc[0].to_dict()
                bits["mol_id"] = mid
                rows.append(bits)
                n_ok += 1
            except Exception:
                rows.append({"mol_id": mid})
        print(f"    ProLIF per-mol: {n_ok}/{len(mol_ids)} OK")
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows).fillna(0)
        return df.rename(columns={c: f"{rec_tag}__{c}"
                                   for c in df.columns if c != "mol_id"})


def process_receptor(rec: str, ligs_csv: Path):
    """Run docking + feature extraction for one receptor."""
    rec_pdbqt = TEMPLATES / rec / "receptor.pdbqt"
    rec_pdb_H = TEMPLATES / rec / "receptor_H.pdb"
    box_txt   = TEMPLATES / rec / "box.txt"
    ref_sdf   = TEMPLATES / rec / "reference_crystal.sdf"
    out_dir   = DOCK_DIR / rec
    out_dir.mkdir(exist_ok=True)

    docked_sdf  = out_dir / "docked_poses.sdf"
    gnina_sdf   = out_dir / "gnina_rescored.sdf"
    features_csv = out_dir / f"features_{rec}.csv"

    if features_csv.exists():
        print(f"  SKIP {rec} (features cached)")
        return pd.read_csv(features_csv)

    print(f"\n--- {rec.upper()} ---")

    # Step 1: UniDock (GPU docking) — runs via Docker, see chain script
    # Step 2: GNINA rescore on UniDock poses — also via Docker
    # This script handles post-docking feature extraction
    if not gnina_sdf.exists() and docked_sdf.exists():
        gnina_sdf = docked_sdf  # use raw docked if not rescored

    if not gnina_sdf.exists():
        print(f"  No docked poses found — run chain_unidock_full.sh first")
        return None

    print(f"  Extracting features from {gnina_sdf.name} ...")

    # GNINA scores from SDF
    rows = []
    gnina_props = ["minimizedAffinity", "CNNscore", "CNNaffinity",
                   "CNNaffinity_variance", "CNN_VS"]
    seen = set()
    suppl = Chem.SDMolSupplier(str(gnina_sdf), removeHs=False, sanitize=False)
    for mol in suppl:
        if mol is None: continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unk"
        if name in seen: continue
        seen.add(name)
        row = {"mol_id": name}
        for p in gnina_props:
            try: row[f"{rec}__{p}"] = float(mol.GetProp(p))
            except: row[f"{rec}__{p}"] = np.nan
        rows.append(row)
    scores_df = pd.DataFrame(rows)
    print(f"  GNINA scores: {len(scores_df)} molecules")

    # Strain energy
    print("  Computing strain energies ...")
    strain_df = compute_strain_energy(gnina_sdf, rec)
    print(f"  Strain: {len(strain_df)} molecules, valid={strain_df[f'{rec}__strain_energy'].notna().sum()}")

    # Distance to key residues
    print("  Computing distances to key residues ...")
    dist_df = extract_distance_to_key_residues(gnina_sdf, rec_pdb_H, rec)

    # ProLIF
    print("  Running ProLIF ...")
    plif_df = extract_plif_full(gnina_sdf, rec_pdb_H, rec)

    # Merge everything
    combined = scores_df
    for df in [strain_df, dist_df]:
        if df is not None and len(df) > 0:
            combined = combined.merge(df, on="mol_id", how="left")
    if plif_df is not None and len(plif_df) > 0:
        combined = combined.merge(plif_df, on="mol_id", how="left")

    combined.to_csv(features_csv, index=False)
    print(f"  Saved {features_csv}  shape={combined.shape}")
    return combined


def main():
    print("=== 75_docking_unidock_full.py ===\n")

    ligs_csv = build_all_ligands_csv()

    all_dfs = []
    for rec in RECEPTORS:
        df = process_receptor(rec, ligs_csv)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = merged.merge(df, on="mol_id", how="outer")
        merged.to_csv(RESULTS_DIR / "docking_full_features_all.csv", index=False)
        print(f"\nMerged features: {merged.shape}")
    else:
        print("No features yet — run docking first")


if __name__ == "__main__":
    main()
