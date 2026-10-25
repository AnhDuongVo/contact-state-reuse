"""Correlation statistics with resampling-based uncertainty."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, n_perm: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Spearman rho, percentile bootstrap 95 % CI and two-sided permutation p-value.

    With few points (for example 3 seeds x 3 severities) the interval will be
    wide; that is the honest outcome, not a reason to report rho alone.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_perm": float("nan"), "n": n}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boots.append(spearmanr(x[idx], y[idx]).statistic)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1))
    return {
        "rho": rho,
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "p_perm": p,
        "n": n,
    }
