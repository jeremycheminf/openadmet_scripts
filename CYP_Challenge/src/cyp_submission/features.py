"""Featurization: ECFP4 (2048-bit Morgan) + RDKit2D (217 descriptors), both via
plain RDKit — no external dependencies beyond what's in requirements.txt."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(smiles_list: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _ECFP4_GEN.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def rdkit2d(smiles_list: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append(Descriptors.CalcMolDescriptors(mol) if mol is not None else {})
    return pd.DataFrame(rows)


def featurize(smiles_list: list[str]) -> pd.DataFrame:
    fp = ecfp4(smiles_list)
    fp_df = pd.DataFrame(fp, columns=[f"ecfp4_{i}" for i in range(fp.shape[1])])
    desc_df = rdkit2d(smiles_list).reset_index(drop=True)
    return pd.concat([fp_df, desc_df], axis=1)
