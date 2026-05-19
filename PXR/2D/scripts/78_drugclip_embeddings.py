"""
78_drugclip_embeddings.py

Extract DrugCLIP 128-d ligand embeddings and pocket-ligand similarity scores
from UniDock docked poses for PXR receptors 2o9i, 8r81, 8eqz.

Runs in: drugclip_env (Python 3.9, unicore, torch 2.8+cuda)

Strategy per receptor:
  - Take best UniDock pose (first conformer) per molecule from docked SDF
  - Pass 3D coordinates directly to DrugCLIP (no re-conformer generation)
  - Extract pocket residues within 6 Å of crystal ligand centroid from receptor PDB
  - Get 128-d normalized mol embedding + 128-d pocket embedding
  - Similarity score = mol_emb @ pocket_emb (cosine, both L2-normalised)

Outputs:
  features/drugclip_emb_{receptor}_train.npy   (n_train × 128)
  features/drugclip_emb_{receptor}_test.npy    (n_test  × 128)
  features/drugclip_sim_{receptor}_train.npy   (n_train,) cosine similarity
  features/drugclip_sim_{receptor}_test.npy    (n_test,)
  features/drugclip_emb_train.npy              (n_train × 128*3 concat, or avg)
  features/drugclip_emb_test.npy

Run:
  wsl -e bash -c "/home/jeremy/mambaforge/envs/drugclip_env/bin/python -u \\
    /mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR/78_drugclip_embeddings.py"
"""
from __future__ import annotations
import os, sys, pickle, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import lmdb
import torch
from tqdm import tqdm

# ── DrugCLIP path setup ────────────────────────────────────────────────────
DRUGCLIP_DIR = Path("/home/jeremy/drugclip")
sys.path.insert(0, str(DRUGCLIP_DIR))   # parent of unimol/ so "unimol.tasks" resolves

import unimol.tasks   # registers @register_task("drugclip")
import unimol.models  # registers @register_model_architecture("drugclip", ...)

from unicore import checkpoint_utils, tasks, options
import unicore.utils

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.WARNING,
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path("/mnt/c/Users/jeremy/Documents/Scripts/Python/OpenADMET/PXR")
FEAT_DIR    = ROOT / "features"
DOCK_DIR    = ROOT / "results" / "docking_full"
TEMPL_DIR   = ROOT / "3D" / "data" / "templates"
CKPT        = Path("/home/jeremy/drugclip_weights/checkpoint_best.pt")
DATA_DIR    = DRUGCLIP_DIR / "data"       # contains dict_mol.txt / dict_pkt.txt
TMP_DIR     = Path("/tmp/drugclip_pxr")

RECEPTORS   = ["2o9i", "8r81", "8eqz"]
POCKET_CUT  = 6.0   # Å from crystal ligand centroid for pocket extraction
MAX_ATOMS   = 256   # max pocket atoms (same as DrugCLIP default)

FEAT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports that need drugclip_env ─────────────────────────────────────────
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
RDLogger.DisableLog("rdApp.*")


# ══════════════════════════════════════════════════════════════════════════
# Data preparation helpers
# ══════════════════════════════════════════════════════════════════════════

def sdf_to_lmdb(sdf_path: Path, out_lmdb: Path) -> list[str]:
    """Convert docked multi-mol SDF to DrugCLIP mol LMDB.

    Stores the SDF _Name (e.g. 'train_0000', 'test_0042') in each record
    so we can map back to OADMET IDs via row index.
    Returns list of SDF names in LMDB order.
    """
    records, names = [], []
    seen = set()
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        # Strip pose suffix: 'train_0000_1' → 'train_0000'
        base = "_".join(name.split("_")[:2]) if name.count("_") >= 2 else name
        if base in seen:
            continue
        seen.add(base)
        coords     = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
        atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
        smi        = Chem.MolToSmiles(mol)
        records.append({"atoms": atom_types, "coordinates": [coords],
                        "smi": smi, "sdf_name": base})
        names.append(base)

    env = lmdb.open(str(out_lmdb), subdir=False, lock=False,
                    readahead=False, meminit=False, map_size=1 << 40)
    with env.begin(write=True) as txn:
        for i, rec in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(rec))
    env.close()
    print(f"  {out_lmdb.name}: {len(records)} molecules")
    return names


def _append_sdf_to_lmdb(sdf_path: Path, out_lmdb: Path):
    """Append molecules from an additional SDF into an existing LMDB."""
    env = lmdb.open(str(out_lmdb), subdir=False, lock=False,
                    readahead=False, meminit=False, map_size=1 << 40)
    with env.begin() as txn:
        offset = txn.stat()["entries"]

    records = []
    seen_names = set()
    # Read existing names to avoid duplicates
    with env.begin() as txn:
        for k in txn.cursor().iternext(values=False):
            d = pickle.loads(txn.get(k))
            seen_names.add(d.get("sdf_name", ""))

    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        base = "_".join(name.split("_")[:2]) if name.count("_") >= 2 else name
        if base in seen_names:
            continue
        seen_names.add(base)
        coords     = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
        atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
        records.append({"atoms": atom_types, "coordinates": [coords],
                        "smi": Chem.MolToSmiles(mol), "sdf_name": base})

    with env.begin(write=True) as txn:
        for i, rec in enumerate(records):
            txn.put(str(offset + i).encode(), pickle.dumps(rec))
    env.close()
    print(f"  Appended {len(records)} molecules from {sdf_path.name}")


def pdb_to_pocket_lmdb(pdb_path: Path, ref_sdf: Path, out_lmdb: Path,
                        receptor_name: str, cutoff: float = POCKET_CUT,
                        max_atoms: int = MAX_ATOMS):
    """Extract binding pocket from PDB and write to DrugCLIP pocket LMDB.

    Pocket = all protein heavy atoms within `cutoff` Å of any crystal ligand atom.
    """
    # Load reference ligand to get binding site centroid / atom positions
    ref_mol = next(Chem.SDMolSupplier(str(ref_sdf), removeHs=True))
    if ref_mol is None or ref_mol.GetNumConformers() == 0:
        raise ValueError(f"Cannot read reference ligand from {ref_sdf}")
    ref_coords = np.array(ref_mol.GetConformer(0).GetPositions())  # (n_ref, 3)

    # Load PDB protein atoms
    pdb = PandasPdb().read_pdb(str(pdb_path))
    atoms_df = pdb.df["ATOM"]

    protein_coords = atoms_df[["x_coord", "y_coord", "z_coord"]].values  # (N, 3)
    atom_names     = atoms_df["atom_name"].tolist()

    # Find atoms within cutoff of any ref_ligand atom
    dists = np.min(
        np.linalg.norm(protein_coords[:, None, :] - ref_coords[None, :, :], axis=-1),
        axis=1
    )
    mask = dists < cutoff
    pocket_atoms  = [atom_names[i] for i in range(len(atom_names)) if mask[i]]
    pocket_coords = protein_coords[mask]

    # Truncate to max_atoms if needed
    if len(pocket_atoms) > max_atoms:
        pocket_atoms  = pocket_atoms[:max_atoms]
        pocket_coords = pocket_coords[:max_atoms]

    print(f"  Pocket {receptor_name}: {len(pocket_atoms)} atoms (cutoff={cutoff}Å)")

    rec = {"pocket": receptor_name,
           "pocket_atoms": pocket_atoms,
           "pocket_coordinates": pocket_coords.astype(np.float32)}

    env = lmdb.open(str(out_lmdb), subdir=False, lock=False,
                    readahead=False, meminit=False, map_size=1 << 30)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(rec))
    env.close()


# ══════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════

def load_model():
    """Load DrugCLIP model via unicore checkpoint utils."""
    import argparse
    # Minimal args namespace that setup_task + build_model need
    args = argparse.Namespace(
        task="drugclip",
        data=str(DATA_DIR),
        seed=1,
        arch="drugclip",
        max_seq_len=512,
        max_pocket_atoms=MAX_ATOMS,
        finetune_mol_model=None,
        finetune_pocket_model=None,
        dist_threshold=6.0,
        reg=False,
        test_model=False,
        # unicore model defaults needed by drugclip_architecture
        encoder_layers=15,
        encoder_embed_dim=512,
        encoder_ffn_embed_dim=2048,
        encoder_attention_heads=64,
        dropout=0.1,
        emb_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len_mol=512,
        max_seq_len_pocket=512,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        post_ln=False,
        masked_token_loss=-1,
        masked_coord_loss=-1,
        masked_dist_loss=-1,
        x_norm_loss=-1,
        delta_pair_repr_norm_loss=-1,
    )

    task = tasks.setup_task(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(str(CKPT))

    # Build model from checkpoint args if available
    if "args" in state:
        ckpt_args = state["args"]
        for k, v in vars(ckpt_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    model = model.cuda().half().eval()
    print(f"  DrugCLIP model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, task


# ══════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════

def encode_mols(model, task, mol_lmdb: Path) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, sdf_names) — shape (N, 128).
    sdf_names are the original SDF _Name values (e.g. 'train_0000').
    """
    cache = mol_lmdb.with_suffix(".emb.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Read sdf_names directly from LMDB (don't rely on dataset's smi_name)
    env = lmdb.open(str(mol_lmdb), subdir=False, readonly=True, lock=False)
    txn = env.begin()
    sdf_name_list = []
    for k in txn.cursor().iternext(values=False):
        d = pickle.loads(txn.get(k))
        sdf_name_list.append(d.get("sdf_name", d.get("smi", str(k))))
    env.close()

    mol_dataset = task.load_retrieval_mols_dataset(str(mol_lmdb), "atoms", "coordinates")
    loader = torch.utils.data.DataLoader(
        mol_dataset, batch_size=64, collate_fn=mol_dataset.collater,
        num_workers=4, pin_memory=True,
    )

    mol_reps = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  encoding mols"):
            batch = unicore.utils.move_to_cuda(batch)
            st   = batch["net_input"]["mol_src_tokens"]
            dist = batch["net_input"]["mol_src_distance"]
            et   = batch["net_input"]["mol_src_edge_type"]
            pad  = st.eq(model.mol_model.padding_idx)
            x    = model.mol_model.embed_tokens(st)
            n    = dist.size(-1)
            gbf  = model.mol_model.gbf(dist, et)
            gbf  = model.mol_model.gbf_proj(gbf).permute(0,3,1,2).contiguous().view(-1,n,n)
            out  = model.mol_model.encoder(x, padding_mask=pad, attn_mask=gbf)
            cls  = out[0][:, 0, :]
            emb  = model.mol_project(cls)
            emb  = emb / emb.norm(dim=-1, keepdim=True)
            mol_reps.append(emb.float().cpu().numpy())

    mol_reps = np.concatenate(mol_reps, axis=0)
    # Use sdf_names read directly from LMDB (order-preserving, no SMILES matching)
    assert len(mol_reps) == len(sdf_name_list), \
        f"Embedding count {len(mol_reps)} != LMDB count {len(sdf_name_list)}"
    with open(cache, "wb") as f:
        pickle.dump((mol_reps, sdf_name_list), f)
    return mol_reps, sdf_name_list


def encode_pocket(model, task, pocket_lmdb: Path) -> np.ndarray:
    """Return pocket embedding — shape (128,)."""
    pocket_dataset = task.load_pockets_dataset(str(pocket_lmdb))
    loader = torch.utils.data.DataLoader(
        pocket_dataset, batch_size=1, collate_fn=pocket_dataset.collater,
    )
    with torch.no_grad():
        for batch in loader:
            batch  = unicore.utils.move_to_cuda(batch)
            st     = batch["net_input"]["pocket_src_tokens"]
            dist   = batch["net_input"]["pocket_src_distance"]
            et     = batch["net_input"]["pocket_src_edge_type"]
            pad    = st.eq(model.pocket_model.padding_idx)
            x      = model.pocket_model.embed_tokens(st)
            n      = dist.size(-1)
            gbf    = model.pocket_model.gbf(dist, et)
            gbf    = model.pocket_model.gbf_proj(gbf).permute(0,3,1,2).contiguous().view(-1,n,n)
            out    = model.pocket_model.encoder(x, padding_mask=pad, attn_mask=gbf)
            cls    = out[0][:, 0, :]
            emb    = model.pocket_project(cls)
            emb    = emb / emb.norm(dim=-1, keepdim=True)
            return emb.float().cpu().numpy()[0]  # (128,)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def build_sdf_to_oadmet_map(train_df, test_df, id_col):
    """Map 'train_XXXX'/'test_XXXX' SDF names → OADMET IDs via row index.

    train_XXXX → train_df.iloc[XXXX][id_col]
    test_XXXX  → test_df.iloc[XXXX][id_col]
    """
    mapping = {}
    for i, oid in enumerate(train_df[id_col]):
        mapping[f"train_{i:04d}"] = str(oid)
    for i, oid in enumerate(test_df[id_col]):
        mapping[f"test_{i:04d}"]  = str(oid)
    return mapping


def main():
    print("=== 78_drugclip_embeddings.py ===\n")

    train_csv = ROOT / "data" / "train_final.csv"
    if not train_csv.exists():
        train_csv = ROOT / "data" / "train_curated.csv"
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(ROOT / "data" / "test_raw.csv")
    id_col   = "OCNT_ID" if "OCNT_ID" in train_df.columns else "Molecule Name"
    train_ids = train_df[id_col].tolist()
    test_ids  = test_df[id_col].tolist()
    # Index-based mapping: 'train_0000' → OADMET ID (row 0 of train_df)
    sdf_to_oadmet = build_sdf_to_oadmet_map(train_df, test_df, id_col)
    print(f"Train: {len(train_ids)}  Test: {len(test_ids)}")

    # Load model once
    print("\nLoading DrugCLIP model ...")
    model, task = load_model()

    all_train_embs, all_test_embs = [], []

    for receptor in RECEPTORS:
        print(f"\n{'='*50}")
        print(f"Receptor: {receptor}")

        dock_subdir = DOCK_DIR / receptor / "docked_poses.sdf"
        sdf_path    = dock_subdir / "docked_poses.sdf"
        pdb_path    = TEMPL_DIR / receptor / "receptor_H.pdb"
        if not pdb_path.exists():
            pdb_path = TEMPL_DIR / receptor / "receptor.pdb"
        ref_sdf     = TEMPL_DIR / receptor / "reference_crystal.sdf"

        if not sdf_path.exists():
            print(f"  WARNING: {sdf_path} not found, skipping")
            all_train_embs.append(np.zeros((len(train_ids), 128), dtype=np.float32))
            all_test_embs.append(np.zeros((len(test_ids), 128), dtype=np.float32))
            continue

        # ── Prepare LMDBs ──
        print("  Preparing molecule LMDB from docked poses ...")
        mol_lmdb_all = TMP_DIR / f"mols_{receptor}_all_v2.lmdb"  # v2 stores sdf_name
        # Check for separate test-only docked SDF (e.g. from 80_dock_test_2o9i.py)
        test_sdf = dock_subdir.parent / "test_docked_poses.sdf"
        if not mol_lmdb_all.exists():
            sdf_to_lmdb(sdf_path, mol_lmdb_all)
            # Append test poses if available (may be in separate file)
            if test_sdf.exists():
                print(f"  Appending test poses from {test_sdf.name} ...")
                _append_sdf_to_lmdb(test_sdf, mol_lmdb_all)
        else:
            env = lmdb.open(str(mol_lmdb_all), subdir=False, readonly=True, lock=False)
            n = env.begin().stat()["entries"]
            env.close()
            print(f"  {mol_lmdb_all.name}: {n} (cached)")

        print("  Preparing pocket LMDB ...")
        pocket_lmdb = TMP_DIR / f"pocket_{receptor}.lmdb"
        if not pocket_lmdb.exists():
            pdb_to_pocket_lmdb(pdb_path, ref_sdf, pocket_lmdb, receptor)

        # ── Encode ──
        print("  Encoding molecules ...")
        mol_embs, sdf_names = encode_mols(model, task, mol_lmdb_all)

        # Map sdf_names ('train_0000') → OADMET IDs via row index
        name_to_emb = {}
        for sdf_name, emb in zip(sdf_names, mol_embs):
            oadmet_id = sdf_to_oadmet.get(sdf_name)
            if oadmet_id:
                name_to_emb[oadmet_id] = emb

        print("  Encoding pocket ...")
        pocket_emb = encode_pocket(model, task, pocket_lmdb)  # (128,)

        # ── Build aligned feature arrays ──
        def build_aligned(ids):
            out = np.zeros((len(ids), 128), dtype=np.float32)
            missing = 0
            for i, oid in enumerate(ids):
                if oid in name_to_emb:
                    out[i] = name_to_emb[oid]
                else:
                    missing += 1
            if missing:
                print(f"    WARNING: {missing} molecules missing embeddings (set to zero)")
            return out

        tr_emb = build_aligned(train_ids)   # (n_train, 128)
        te_emb = build_aligned(test_ids)    # (n_test,  128)

        # Similarity scores (cosine, since both L2-normalised)
        tr_sim = tr_emb @ pocket_emb        # (n_train,)
        te_sim = te_emb @ pocket_emb        # (n_test,)

        # Save per-receptor
        np.save(FEAT_DIR / f"drugclip_emb_{receptor}_train.npy", tr_emb)
        np.save(FEAT_DIR / f"drugclip_emb_{receptor}_test.npy",  te_emb)
        np.save(FEAT_DIR / f"drugclip_sim_{receptor}_train.npy", tr_sim)
        np.save(FEAT_DIR / f"drugclip_sim_{receptor}_test.npy",  te_sim)
        print(f"  Sim score  mean={tr_sim.mean():.4f}  std={tr_sim.std():.4f}")

        all_train_embs.append(tr_emb)
        all_test_embs.append(te_emb)

    # ── Aggregate across receptors ──
    print("\nAggregating across receptors ...")
    # Concatenate (n × 384) and also save mean (n × 128)
    tr_concat = np.concatenate(all_train_embs, axis=1)   # (n_train, 384)
    te_concat = np.concatenate(all_test_embs,  axis=1)   # (n_test,  384)
    tr_mean   = np.mean(all_train_embs, axis=0)           # (n_train, 128)
    te_mean   = np.mean(all_test_embs,  axis=0)           # (n_test,  128)

    np.save(FEAT_DIR / "drugclip_concat_train.npy", tr_concat)
    np.save(FEAT_DIR / "drugclip_concat_test.npy",  te_concat)
    np.save(FEAT_DIR / "drugclip_emb_train.npy",    tr_mean)
    np.save(FEAT_DIR / "drugclip_emb_test.npy",     te_mean)

    print(f"\nSaved features:")
    print(f"  drugclip_emb_{{receptor}}_{{split}}.npy  — 128-d embedding per receptor")
    print(f"  drugclip_sim_{{receptor}}_{{split}}.npy  — cosine similarity score per receptor")
    print(f"  drugclip_concat_{{split}}.npy           — 384-d concat (3 receptors)")
    print(f"  drugclip_emb_{{split}}.npy              — 128-d mean across receptors")
    print("\nDone.")


def _read_names_from_lmdb(lmdb_path: Path) -> list[str]:
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    txn = env.begin()
    names = []
    for k in txn.cursor().iternext(values=False):
        d = pickle.loads(txn.get(k))
        names.append(d.get("smi", ""))
    env.close()
    return names


if __name__ == "__main__":
    main()
