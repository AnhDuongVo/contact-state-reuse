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
    "peak_force_sum",
    "compliance_pre_contact",
    "compliance_in_contact",
    "approach_speed",
    "masked_force",
    "masked_steps",
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


# ---------------------------------------------------------------- v0.2
# Training-run-aware uncertainty. Evaluation episodes are shared by every
# trained model (same evaluation seeds), and every condition is trained with
# the same set of training seeds, so results form a crossed seeds x episodes
# design. The crossed ("pigeonhole") bootstrap resamples training seeds and
# evaluation episodes independently (Owen, 2007), so intervals reflect
# variation between training runs as well as between evaluation episodes, and
# keep the pairing of conditions on the same seed and episode.


def crossed_bootstrap(mat: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Mean and percentile CI of ``mat`` [seeds, episodes] (NaNs ignored)."""
    mat = np.asarray(mat, float)
    S, E = mat.shape
    rng = np.random.default_rng(seed)
    means = []
    for b0 in range(0, n_boot, 250):
        nb = min(250, n_boot - b0)
        si = rng.integers(0, S, (nb, S))
        ei = rng.integers(0, E, (nb, E))
        vals = mat[si[:, :, None], ei[:, None, :]]
        with np.errstate(all="ignore"):
            means.append(np.nanmean(vals.reshape(nb, -1), 1))
    means = np.concatenate(means)
    return float(np.nanmean(mat)), float(np.nanquantile(means, alpha / 2)), float(np.nanquantile(means, 1 - alpha / 2))


def _matrix(df: pd.DataFrame, metric: str, episode_cols: tuple[str, ...] = ("episode",)) -> pd.DataFrame:
    return df.pivot_table(index="seed", columns=list(episode_cols), values=metric, aggfunc="mean")


def seed_episode_summary(
    df: pd.DataFrame, metric: str, n_boot: int = 2000, episode_cols: tuple[str, ...] = ("episode",)
) -> dict:
    """Mean, crossed-bootstrap CI and per-seed means for one condition (and domain).
    ``episode_cols`` identifies an evaluation episode, e.g. ("domain", "episode")
    when pooling over domains."""
    m = _matrix(df, metric, episode_cols)
    mean, lo, hi = crossed_bootstrap(m.to_numpy(), n_boot)
    per_seed = m.mean(1)
    return {
        "metric": metric,
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
        "per_seed": [round(float(v), 4) for v in per_seed.to_numpy()],
        "n_seeds": int(m.shape[0]),
        "n_episodes": int(m.shape[1]),
    }


def paired_crossed(
    df: pd.DataFrame,
    condition_col: str,
    a: str,
    b: str,
    metric: str,
    n_boot: int = 2000,
    episode_cols: tuple[str, ...] = ("episode",),
) -> dict:
    """Paired difference a - b on matched (seed, episode) cells with a crossed bootstrap CI,
    plus per-seed differences and how many seeds favour a (higher success, lower force)."""
    ma = _matrix(df[df[condition_col] == a], metric, episode_cols)
    mb = _matrix(df[df[condition_col] == b], metric, episode_cols)
    seeds = ma.index.intersection(mb.index)
    eps = ma.columns.intersection(mb.columns)
    d = ma.loc[seeds, eps].to_numpy() - mb.loc[seeds, eps].to_numpy()
    mean, lo, hi = crossed_bootstrap(d, n_boot)
    per_seed = np.nanmean(d, 1)
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "per_seed": [round(float(v), 4) for v in per_seed],
        "seeds_a_better": int(np.sum(per_seed > 0)) if metric in ("success",) else int(np.sum(per_seed < 0)),
        "n_seeds": int(len(seeds)),
        "n_episodes": int(len(eps)),
    }
