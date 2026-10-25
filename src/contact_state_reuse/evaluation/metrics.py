"""Episode metrics and bootstrap summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

EPISODE_METRICS = (
    "success",
    "return",
    "peak_force",
    "impulse",
    "force_violation",
    "time_to_success",
    "jam_or_failure",
    "steps",
    "first_contact_step",
)


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval over the given values (NaNs dropped)."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(
    df: pd.DataFrame, by: list[str], metrics: Iterable[str] = EPISODE_METRICS, n_boot: int = 2000
) -> pd.DataFrame:
    """Tidy table: one row per group and metric with mean and 95 % bootstrap CI over episodes."""
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if m not in g:
                continue
            mean, lo, hi = bootstrap_ci(g[m].to_numpy(dtype=float), n_boot=n_boot)
            rows.append(
                {
                    **dict(zip(by, key)),
                    "metric": m,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": int(np.isfinite(g[m].to_numpy(dtype=float)).sum()),
                }
            )
    return pd.DataFrame(rows)


def seed_level(df: pd.DataFrame, by: list[str], metrics: Iterable[str] = EPISODE_METRICS) -> pd.DataFrame:
    """Per-seed means, then mean and standard deviation across seeds."""
    if "seed" not in df:
        raise KeyError("seed column required")
    per_seed = df.groupby([*by, "seed"])[list(metrics)].mean().reset_index()
    agg = per_seed.groupby(by)[list(metrics)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def paired_difference(
    df: pd.DataFrame,
    condition_col: str,
    a: str,
    b: str,
    metric: str,
    keys: tuple[str, ...] = ("seed", "domain", "episode"),
    n_boot: int = 2000,
) -> dict[str, float]:
    """Mean of (a - b) over matched episodes (same seed/domain/episode index) with a bootstrap CI."""
    da = df[df[condition_col] == a].set_index(list(keys))[metric]
    db = df[df[condition_col] == b].set_index(list(keys))[metric]
    j = pd.concat([da.rename("a"), db.rename("b")], axis=1, join="inner").dropna()
    mean, lo, hi = bootstrap_ci((j.a - j.b).to_numpy(), n_boot=n_boot)
    return {"a": a, "b": b, "metric": metric, "diff": mean, "ci_low": lo, "ci_high": hi, "n": int(len(j))}
