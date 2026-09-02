"""Caruana ensemble selection (Caruana et al. 2004): forward stepwise selection
*with replacement* (a strong candidate can be picked multiple times, giving it more
weight without a continuous optimizer that can overfit/reallocate weight
destructively on correlated candidates), sorted top-N init, bagged over random
subsets of the candidate library. Generalizes to both continuous regression targets
and boolean classification votes (see scripts/06_ensemble.py for both uses).

    weights = caruana_bagged_ensemble(oof_matrix, y_true, score_fn, minimize=True)
    final_test_pred = test_matrix @ weights   # weights sum to 1
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _select_with_replacement(oof: np.ndarray, y: np.ndarray, score_fn, minimize: bool,
                              n_init: int, max_iter: int) -> list[int]:
    n_candidates = oof.shape[1]
    individual = np.array([score_fn(y, oof[:, i]) for i in range(n_candidates)])
    order = np.argsort(individual) if minimize else np.argsort(-individual)

    selected = list(order[:n_init])
    current_sum = oof[:, selected].sum(axis=1)
    current_score = score_fn(y, current_sum / len(selected))

    for _ in range(max_iter):
        best_idx, best_score = None, current_score
        for i in range(n_candidates):
            trial_pred = (current_sum + oof[:, i]) / (len(selected) + 1)
            s = score_fn(y, trial_pred)
            if (minimize and s < best_score) or (not minimize and s > best_score):
                best_score, best_idx = s, i
        if best_idx is None:
            break
        selected.append(best_idx)
        current_sum = current_sum + oof[:, best_idx]
        current_score = best_score

    return selected


def caruana_bagged_ensemble(oof: np.ndarray, y: np.ndarray, score_fn, *, minimize: bool = True,
                             n_bags: int = 20, bag_frac: float = 0.5, n_init: int = 1,
                             max_iter: int = 50, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_candidates = oof.shape[1]
    bag_size = max(1, int(round(bag_frac * n_candidates)))

    counts = Counter()
    for _ in range(n_bags):
        subset = rng.choice(n_candidates, size=bag_size, replace=False)
        selected_local = _select_with_replacement(
            oof[:, subset], y, score_fn, minimize, n_init=min(n_init, bag_size), max_iter=max_iter,
        )
        for local_idx in selected_local:
            counts[subset[local_idx]] += 1

    weights = np.zeros(n_candidates)
    if not counts:
        weights[:] = 1.0 / n_candidates
    else:
        for idx, c in counts.items():
            weights[idx] = c
        weights /= weights.sum()
    return weights
