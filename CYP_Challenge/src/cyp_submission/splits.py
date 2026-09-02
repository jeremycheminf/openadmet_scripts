"""Butina (scaffold-cluster) k-fold splitting.

The blind test set has zero SMILES overlap with train and is an "analog-expansion"
set, so random k-fold would overestimate CV performance — cluster-based folds are a
closer proxy to the real train/test shift.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def butina_clusters(smiles_list: list[str], cutoff: float = 0.6) -> np.ndarray:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [_FP_GEN.GetFingerprint(m) if m is not None else None for m in mols]

    n = len(fps)
    dists = []
    for i in range(1, n):
        if fps[i] is None:
            sims = [0.0] * i
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(i)])
        dists.extend(1 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, 1 - cutoff, isDistData=True)
    cluster_id = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_id[m] = cid
    return cluster_id


def butina_kfold(smiles_list: list[str], n_folds: int = 5, cutoff: float = 0.6,
                  seed: int = 42) -> np.ndarray:
    cluster_id = butina_clusters(smiles_list, cutoff=cutoff)
    rng = np.random.default_rng(seed)

    unique_clusters, counts = np.unique(cluster_id, return_counts=True)
    order = np.argsort(-counts)
    unique_clusters, counts = unique_clusters[order], counts[order]

    fold_sizes = np.zeros(n_folds, dtype=int)
    cluster_to_fold = {}
    for cid, cnt in zip(unique_clusters, counts):
        min_size = fold_sizes.min()
        candidates = np.flatnonzero(fold_sizes == min_size)
        fold = rng.choice(candidates)
        cluster_to_fold[cid] = fold
        fold_sizes[fold] += cnt

    return np.array([cluster_to_fold[c] for c in cluster_id])
