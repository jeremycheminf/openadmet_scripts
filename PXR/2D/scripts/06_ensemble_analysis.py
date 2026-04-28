"""
Post-CV analysis:
  1. Paired Wilcoxon signed-rank tests between all model pairs (+ BH FDR correction)
  2. OOF-based ensemble evaluation with smart (non-brute-force) weighting:
       - Simple mean / median of top-K
       - Performance-weighted (1/RMSE_cv, softmax)
       - Non-negative least squares (NNLS) — convex optimisation, no search
       - ElasticNetCV meta-learner
       - Greedy ensemble selection (Caruana et al. 2004) — O(K²), finds compact ensembles
  3. Saves results/ensemble_analysis.csv and results/pairwise_wilcoxon.csv
     and produces results/ensemble_comparison.png

Run AFTER 04_cv_benchmark.py.
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils import FEATURE_DIR, DATA_DIR

warnings.filterwarnings("ignore")

RESULTS_DIR = ROOT / "results"
OOF_DIR     = RESULTS_DIR / "oof_predictions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_pred)
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def oof_rmse(y_true: np.ndarray, pred_matrix: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ensemble RMSE on OOF rows where all preds are non-NaN."""
    mask = ~np.isnan(pred_matrix).any(axis=1)
    ensemble = pred_matrix[mask] @ weights
    return float(np.sqrt(mean_squared_error(y_true[mask], ensemble)))


# ---------------------------------------------------------------------------
# 1. Paired Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def pairwise_wilcoxon(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every pair of models, run a paired Wilcoxon signed-rank test on
    per-fold RMSE values (same fold = same data split = paired).
    Returns a tidy DataFrame with BH-FDR-corrected p-values.
    """
    # Pivot to (fold_key, model) matrix of per-fold RMSE
    cv_df = cv_df.copy()
    cv_df["fold_key"] = cv_df["repeat"].astype(str) + "_" + cv_df["fold"].astype(str)
    pivot = cv_df.pivot_table(index="fold_key", columns="model", values="RMSE")
    models = pivot.columns.tolist()

    rows = []
    for m1, m2 in combinations(models, 2):
        paired = pivot[[m1, m2]].dropna()
        if len(paired) < 5:
            continue
        d = paired[m1].values - paired[m2].values
        if (d == 0).all():
            continue
        stat, p = wilcoxon(d, alternative="two-sided")
        rows.append({
            "model_A": m1,
            "model_B": m2,
            "rmse_A":  paired[m1].mean(),
            "rmse_B":  paired[m2].mean(),
            "delta_rmse": paired[m1].mean() - paired[m2].mean(),
            "W_stat": stat,
            "p_value": p,
            "n_pairs": len(paired),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # BH FDR correction
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(res["p_value"].values, method="fdr_bh")
    res["p_adj_BH"] = p_adj
    res["significant"] = res["p_adj_BH"] < 0.05
    return res.sort_values("p_value")


# ---------------------------------------------------------------------------
# 2. Ensemble strategies
# ---------------------------------------------------------------------------

def strategy_mean(pred_matrix: np.ndarray, model_names: list[str],
                  **_) -> np.ndarray:
    mask = ~np.isnan(pred_matrix).any(axis=1)
    w = np.ones(pred_matrix.shape[1]) / pred_matrix.shape[1]
    return w, "mean_all"


def strategy_topk_mean(pred_matrix: np.ndarray, model_names: list[str],
                       cv_rmse: dict[str, float], k: int = 5, **_):
    ranked = sorted(model_names, key=lambda m: cv_rmse.get(m, 999))[:k]
    idx = [model_names.index(m) for m in ranked]
    w = np.zeros(len(model_names))
    w[idx] = 1.0 / len(idx)
    return w, f"mean_top{k}"


def strategy_median(pred_matrix: np.ndarray, model_names: list[str],
                    y_true: np.ndarray, **_):
    """Median is not a linear combination — evaluate directly."""
    mask = ~np.isnan(pred_matrix).any(axis=1)
    pred = np.median(pred_matrix[mask], axis=1)
    r = float(np.sqrt(mean_squared_error(y_true[mask], pred)))
    return None, "median_all", r


def strategy_inv_rmse(pred_matrix: np.ndarray, model_names: list[str],
                      cv_rmse: dict[str, float], **_):
    """Weight ∝ 1 / CV_RMSE (better model gets higher weight)."""
    rmses = np.array([cv_rmse.get(m, 1.0) for m in model_names])
    w = (1.0 / rmses)
    w /= w.sum()
    return w, "inv_rmse_weighted"


def strategy_softmax_neg_rmse(pred_matrix: np.ndarray, model_names: list[str],
                               cv_rmse: dict[str, float], temp: float = 10.0, **_):
    """softmax(-RMSE * temp) — sharper than inv_rmse; temp controls peakedness."""
    rmses = np.array([cv_rmse.get(m, 1.0) for m in model_names])
    logits = -rmses * temp
    logits -= logits.max()  # numerical stability
    w = np.exp(logits)
    w /= w.sum()
    return w, "softmax_neg_rmse"


def strategy_nnls(pred_matrix: np.ndarray, model_names: list[str],
                  y_true: np.ndarray, **_):
    """
    Non-negative least squares: min ||y - Xw||² s.t. w ≥ 0.
    Convex, single closed-form solve — no search required.
    """
    mask = ~np.isnan(pred_matrix).any(axis=1)
    X = pred_matrix[mask]
    y = y_true[mask]
    w_raw, _ = nnls(X, y)
    if w_raw.sum() == 0:
        w = np.ones(len(model_names)) / len(model_names)
    else:
        w = w_raw / w_raw.sum()
    return w, "nnls"


def strategy_elasticnet(pred_matrix: np.ndarray, model_names: list[str],
                        y_true: np.ndarray, **_):
    """ElasticNetCV meta-learner fitted on OOF predictions."""
    mask = ~np.isnan(pred_matrix).any(axis=1)
    X = pred_matrix[mask]
    y = y_true[mask]
    meta = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        cv=5, max_iter=10000, n_jobs=-1,
        positive=True,  # keep weights non-negative
    )
    meta.fit(X, y)
    w = meta.coef_
    if w.sum() > 0:
        w = w / w.sum()
    return w, "elasticnet_oof"


def strategy_greedy_caruana(pred_matrix: np.ndarray, model_names: list[str],
                             y_true: np.ndarray, max_size: int = 10, **_):
    """
    Greedy ensemble selection (Caruana et al. 2004).
    Start from best model; at each step add the model that most reduces RMSE.
    Allows repetition (a model can be selected multiple times — equivalent to
    fractional weighting). Stops when RMSE no longer improves or max_size reached.
    """
    mask = ~np.isnan(pred_matrix).any(axis=1)
    X = pred_matrix[mask]
    y = y_true[mask]
    n_models = X.shape[1]

    # Start with best single model
    base_rmses = [np.sqrt(mean_squared_error(y, X[:, j])) for j in range(n_models)]
    best_start = int(np.argmin(base_rmses))
    ensemble_sum = X[:, best_start].copy()
    selected = [best_start]
    best_rmse = base_rmses[best_start]

    for step in range(1, max_size):
        gains = []
        for j in range(n_models):
            candidate_sum = ensemble_sum + X[:, j]
            candidate_mean = candidate_sum / (step + 1)
            r = np.sqrt(mean_squared_error(y, candidate_mean))
            gains.append(r)
        best_j = int(np.argmin(gains))
        new_rmse = gains[best_j]
        if new_rmse >= best_rmse - 1e-6:
            break  # no improvement
        ensemble_sum += X[:, best_j]
        selected.append(best_j)
        best_rmse = new_rmse

    counts = np.bincount(selected, minlength=n_models).astype(float)
    w = counts / counts.sum()
    selected_names = [model_names[j] for j in set(selected)]
    return w, f"greedy_caruana(n={len(selected)})"


# ---------------------------------------------------------------------------
# 3. Evaluate all strategies and report
# ---------------------------------------------------------------------------

def evaluate_strategies(oof_df: pd.DataFrame, cv_summary: pd.DataFrame) -> pd.DataFrame:
    y_true = oof_df["y_true"].values
    model_cols = [c for c in oof_df.columns if c != "y_true"]

    if not model_cols:
        print("No model columns in OOF file.")
        return pd.DataFrame()

    pred_matrix = oof_df[model_cols].values
    cv_rmse = cv_summary["RMSE_mean"].to_dict()

    print(f"\nEvaluating ensemble strategies on {len(model_cols)} models, "
          f"{(~np.isnan(pred_matrix).any(axis=1)).sum()} complete OOF rows\n")

    # Single-model baselines
    results = []
    for i, m in enumerate(model_cols):
        mask = ~np.isnan(pred_matrix[:, i])
        if mask.sum() < 10:
            continue
        r = np.sqrt(mean_squared_error(y_true[mask], pred_matrix[mask, i]))
        results.append({"strategy": f"[single] {m}", "oof_rmse": r,
                        "n_models": 1, "weights": {m: 1.0}})

    # Ensemble strategies
    kwargs = dict(
        pred_matrix=pred_matrix, model_names=model_cols,
        y_true=y_true, cv_rmse=cv_rmse
    )

    linear_strategies = [
        strategy_mean,
        lambda **kw: strategy_topk_mean(k=3, **kw),
        lambda **kw: strategy_topk_mean(k=5, **kw),
        strategy_inv_rmse,
        strategy_softmax_neg_rmse,
        strategy_nnls,
        strategy_elasticnet,
        strategy_greedy_caruana,
    ]

    for fn in linear_strategies:
        try:
            out = fn(**kwargs)
            if len(out) == 3:
                # non-linear strategy (e.g. median) returns (None, name, rmse)
                _, name, r = out
                results.append({"strategy": name, "oof_rmse": r,
                                 "n_models": len(model_cols), "weights": {}})
            else:
                w, name = out
                r = oof_rmse(y_true, pred_matrix, w)
                active = {model_cols[i]: round(float(w[i]), 4)
                          for i in np.where(w > 1e-4)[0]}
                results.append({"strategy": name, "oof_rmse": r,
                                 "n_models": int((w > 1e-4).sum()),
                                 "weights": active})
        except Exception as e:
            print(f"  Strategy failed: {e}")

    # Median
    try:
        _, name, r = strategy_median(**kwargs)
        results.append({"strategy": name, "oof_rmse": r,
                         "n_models": len(model_cols), "weights": {}})
    except Exception as e:
        print(f"  Median failed: {e}")

    df = pd.DataFrame(results).sort_values("oof_rmse")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Ensemble Analysis & Pairwise Statistical Tests ===\n")

    cv_path  = RESULTS_DIR / "cv_results.csv"
    oof_path = OOF_DIR / "oof_predictions.csv"
    sum_path = RESULTS_DIR / "cv_summary.csv"

    for p in [cv_path, oof_path, sum_path]:
        if not p.exists():
            print(f"ERROR: {p} not found — run 04_cv_benchmark.py first.")
            sys.exit(1)

    cv_df      = pd.read_csv(cv_path)
    oof_df     = pd.read_csv(oof_path)
    cv_summary = pd.read_csv(sum_path, index_col=0)

    print(f"CV results: {len(cv_df)} rows, {cv_df['model'].nunique()} models")
    print(f"OOF file:   {oof_df.shape}")

    # ------------------------------------------------------------------ #
    # 1. Paired Wilcoxon tests
    # ------------------------------------------------------------------ #
    print("\n--- Pairwise Wilcoxon signed-rank tests (BH FDR) ---")
    wilcox_df = pairwise_wilcoxon(cv_df)
    if not wilcox_df.empty:
        print(wilcox_df[["model_A", "model_B", "rmse_A", "rmse_B",
                          "delta_rmse", "p_value", "p_adj_BH", "significant"]]
              .round(5).to_string(index=False))
        wilcox_df.to_csv(RESULTS_DIR / "pairwise_wilcoxon.csv", index=False)
        print(f"\nSaved pairwise_wilcoxon.csv  ({len(wilcox_df)} pairs)")

        sig = wilcox_df[wilcox_df["significant"]]
        print(f"Significant pairs (p_adj < 0.05): {len(sig)}/{len(wilcox_df)}")
    else:
        print("No pairs to compare.")

    # ------------------------------------------------------------------ #
    # 2. Ensemble evaluation
    # ------------------------------------------------------------------ #
    print("\n--- Ensemble strategy comparison ---")
    ens_df = evaluate_strategies(oof_df, cv_summary)

    print("\nRanked by OOF RMSE:")
    print(ens_df[["strategy", "oof_rmse", "n_models"]].round(5).to_string(index=False))

    ens_df.to_csv(RESULTS_DIR / "ensemble_analysis.csv", index=False)
    print(f"\nSaved ensemble_analysis.csv")

    # Weight breakdown for top ensemble strategies
    print("\nWeight breakdown for top ensemble strategies:")
    for _, row in ens_df.head(8).iterrows():
        if row["weights"] and row["strategy"].startswith("[single]") is False:
            print(f"  {row['strategy']}  (RMSE={row['oof_rmse']:.5f})")
            for m, w in sorted(row["weights"].items(), key=lambda x: -x[1]):
                print(f"      {w:.4f}  {m}")

    # ------------------------------------------------------------------ #
    # 3. Plots
    # ------------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # --- Left: per-model boxplot (from CV results) ---
        ax = axes[0]
        order = cv_summary.sort_values("RMSE_mean").index.tolist()
        palette = sns.color_palette("Blues_r", len(order))
        sns.boxplot(data=cv_df, x="model", y="RMSE", order=order, ax=ax,
                    palette=palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_title("3×5-fold Butina CV RMSE per model")
        ax.axhline(cv_summary["RMSE_mean"].min(), color="red",
                   linestyle="--", alpha=0.5, label="best single")
        ax.legend(fontsize=8)

        # --- Right: ensemble strategy comparison ---
        ax = axes[1]
        plot_df = ens_df[~ens_df["strategy"].str.startswith("[single]")].head(15)
        colors = ["#2196F3" if r < cv_summary["RMSE_mean"].min() else "#90CAF9"
                  for r in plot_df["oof_rmse"]]
        ax.barh(plot_df["strategy"][::-1], plot_df["oof_rmse"][::-1], color=colors[::-1])
        ax.axvline(cv_summary["RMSE_mean"].min(), color="red", linestyle="--",
                   alpha=0.7, label=f"best single = {cv_summary['RMSE_mean'].min():.4f}")
        ax.set_xlabel("OOF RMSE")
        ax.set_title("Ensemble strategy OOF RMSE (lower = better)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ensemble_comparison.png", dpi=150)
        plt.close()
        print("Saved ensemble_comparison.png")

        # --- Pairwise significance heatmap ---
        if not wilcox_df.empty:
            models = cv_summary.sort_values("RMSE_mean").index.tolist()
            n = len(models)
            mat = pd.DataFrame(np.ones((n, n)), index=models, columns=models)
            for _, row in wilcox_df.iterrows():
                m1, m2 = row["model_A"], row["model_B"]
                if m1 in mat.index and m2 in mat.columns:
                    mat.loc[m1, m2] = row["p_adj_BH"]
                    mat.loc[m2, m1] = row["p_adj_BH"]

            fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
            mask = mat.values == 1.0
            np.fill_diagonal(mask, True)
            sns.heatmap(
                -np.log10(mat.clip(1e-10, 1.0).values),
                xticklabels=models, yticklabels=models,
                annot=True, fmt=".1f", cmap="RdYlGn",
                ax=ax, mask=mask,
                cbar_kws={"label": "-log10(p_adj BH)"},
            )
            ax.set_title("Pairwise Wilcoxon −log10(p_adj)  [BH FDR]  "
                         "— higher = more significant")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "pairwise_significance.png", dpi=150)
            plt.close()
            print("Saved pairwise_significance.png")

    except Exception as e:
        print(f"  Plot error: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    best_single = cv_summary["RMSE_mean"].min()
    best_single_name = cv_summary["RMSE_mean"].idxmin()
    best_ens = ens_df[~ens_df["strategy"].str.startswith("[single]")].iloc[0]
    improvement = best_single - best_ens["oof_rmse"]
    print(f"\nBest single model:  {best_single_name}  RMSE={best_single:.5f}")
    print(f"Best ensemble:      {best_ens['strategy']}  RMSE={best_ens['oof_rmse']:.5f}")
    if improvement > 0:
        print(f"Ensemble improvement: {improvement:.5f} ({improvement/best_single*100:.1f}%)")
    else:
        print("No ensemble improvement over best single model on OOF data.")


if __name__ == "__main__":
    main()
