"""Representation-reuse score (definition in docs/methods.md).

For phase p with paired source/shifted blocks X_p, Y_p:

    R_p = (CKA(X_p, Y_p) - mean_null_p) / (1 - mean_null_p)

where mean_null_p is the mean CKA under episode permutation. R_p = 0 means
the shifted-domain latents of a situation are no more similar to its
source-domain latents than to those of a random other episode in the same
phase; R_p = 1 means identical geometry up to rotation and scale. Negative
values are possible. The reuse score is the mean of R_p over phases with at
least ``min_episodes`` paired episodes. A leave-one-episode-out jackknife
gives an approximate 95 % interval (``n_boot`` > 0 switches it on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .latents import LatentSet
from .similarity import PHASES_FOR_REUSE, cka_blocks, cka_null, phase_matched_blocks


@dataclass
class ReuseResult:
    score: float
    ci: tuple[float, float]
    per_phase: dict[int, dict[str, float]] = field(default_factory=dict)


def reuse_score(
    src: LatentSet,
    tgt: LatentSet,
    phases: tuple[int, ...] = PHASES_FOR_REUSE,
    n_points: int = 4,
    n_perm: int = 50,
    n_boot: int = 200,
    min_episodes: int = 5,
    seed: int = 0,
) -> ReuseResult:
    blocks = {}
    per_phase = {}
    for p in phases:
        xb, yb, eps = phase_matched_blocks(src, tgt, p, n_points)
        if len(eps) < min_episodes:
            continue
        cka = cka_blocks(xb, yb)
        null = cka_null(xb, yb, n_perm, seed + p)
        mu = float(np.nanmean(null))
        r = (cka - mu) / (1 - mu) if mu < 1 else float("nan")
        per_phase[p] = {
            "cka": cka,
            "null_mean": mu,
            "null_sd": float(np.nanstd(null)),
            "null_q05": float(np.nanquantile(null, 0.05)),
            "null_q50": float(np.nanquantile(null, 0.5)),
            "null_q95": float(np.nanquantile(null, 0.95)),
            "cka_minus_null": float(cka - mu),
            "p_above_null": float((np.sum(null >= cka) + 1) / (len(null) + 1)),
            "reuse": r,
            "n_episodes": int(len(eps)),
        }
        blocks[p] = (xb, yb, mu, eps)
    if not blocks:
        return ReuseResult(float("nan"), (float("nan"), float("nan")), per_phase)
    score = float(np.nanmean([v["reuse"] for v in per_phase.values()]))
    if n_boot <= 0:
        return ReuseResult(score, (float("nan"), float("nan")), per_phase)
    # Leave-one-episode-out jackknife interval. (An episode bootstrap is biased
    # upwards for CKA: duplicated episodes add identical rows to both blocks
    # and inflate the similarity, so its intervals can exclude the estimate.)
    episodes = np.unique(np.concatenate([e for *_, e in blocks.values()]))
    n = len(episodes)
    jack = []
    for ep in episodes:
        vals = []
        for xb, yb, mu, eps in blocks.values():
            keep = eps != ep
            vals.append((cka_blocks(xb[keep], yb[keep]) - mu) / (1 - mu))
        jack.append(np.nanmean(vals))
    jack = np.asarray(jack)
    se = float(np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2)))
    return ReuseResult(score, (score - 1.96 * se, score + 1.96 * se), per_phase)
