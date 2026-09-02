"""Post-hoc linear recalibration ("placement correction").

Per the leaderboard's own write-up (SuperCowPowers/workbench blog, see README
references): rescale predictions toward the mean by k = Pearson correlation rho —
the OLS-optimal linear fit of y on pred. Fit on OOF predictions, apply the same two
scalars to that model's test predictions.
"""

from __future__ import annotations

import numpy as np


def fit_placement_correction(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(y_pred, y_true, deg=1)
    return float(slope), float(intercept)


def apply_placement_correction(y_pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * y_pred + intercept
