"""Representation-reuse score (definition in docs/methods.md).

For phase p with paired source/shifted blocks X_p, Y_p:

    R_p = (CKA(X_p, Y_p) - mean_null_p) / (1 - mean_null_p)

where mean_null_p is the mean CKA under episode permutation. R_p = 0 means
the shifted-domain latents of a situation are no more similar to its
source-domain latents than to those of a random other episode in the same
phase; R_p = 1 means identical geometry up to rotation and scale. Negative
values are possible. The reuse score is the mean of R_p over phases with at
least ``min_episodes`` paired episodes. A bootstrap over episodes gives a CI.
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
            "reuse": r,
            "n_episodes": int(len(eps)),
        }
        blocks[p] = (xb, yb, mu)
    if not blocks:
        return ReuseResult(float("nan"), (float("nan"), float("nan")), per_phase)
    score = float(np.nanmean([v["reuse"] for v in per_phase.values()]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        vals = []
        for xb, yb, mu in blocks.values():
            idx = rng.integers(0, len(xb), len(xb))
            vals.append((cka_blocks(xb[idx], yb[idx]) - mu) / (1 - mu))
        boots.append(np.nanmean(vals))
    return ReuseResult(score, (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))), per_phase)
