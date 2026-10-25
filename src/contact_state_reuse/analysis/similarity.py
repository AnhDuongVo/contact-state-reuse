"""Representation similarity: linear CKA, phase-matched pairing and a permutation null.

Linear CKA (Kornblith et al., 2019) between matrices X [n, p] and Y [n, q]
whose rows are paired observations:

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)

with column-centred Xc, Yc. It is invariant to orthogonal transforms and
isotropic scaling, so two models (or one model in two domains) can be
compared without aligning latent axes. It lies in [0, 1].

Pairing across domains: datasets generated with the same seed in two domains
share initial conditions (see ``data.generate``). For each simulator phase,
the latent trajectory of episode e within that phase is resampled to
``n_points`` positions (by linear interpolation of the row index), and the
rows of X (source) and Y (shifted) are matched by (episode, position).

Null: the episode order of Y is permuted (phase content and within-episode
structure are preserved, the episode correspondence is broken). CKA under
the null measures how similar *any* two same-phase latent sets are; the excess
over it measures whether the same physical situation is encoded the same way
across domains.
"""

from __future__ import annotations

import numpy as np

from .latents import LatentSet

PHASES_FOR_REUSE = (0, 1, 2, 3)  # approach, first_contact, alignment, insertion


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA needs paired rows")
    if x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(0, keepdims=True)
    yc = y - y.mean(0, keepdims=True)
    num = np.linalg.norm(yc.T @ xc, "fro") ** 2
    den = np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro")
    return float(num / den) if den > 0 else float("nan")


def _resample(z: np.ndarray, n_points: int) -> np.ndarray:
    if len(z) == 1:
        return np.repeat(z, n_points, 0)
    pos = np.linspace(0, len(z) - 1, n_points)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(z) - 1)
    w = (pos - lo)[:, None]
    return z[lo] * (1 - w) + z[hi] * w


def phase_matched_blocks(src: LatentSet, tgt: LatentSet, phase: int, n_points: int = 4):
    """Return ``(X, Y, episodes)`` with X, Y of shape [E, n_points, H] for episodes
    present in both sets that visit ``phase`` in both domains."""
    xs, ys, eps = [], [], []
    for e in np.intersect1d(np.unique(src.episode), np.unique(tgt.episode)):
        ms = (src.episode == e) & (src.mode == phase)
        mt = (tgt.episode == e) & (tgt.mode == phase)
        if ms.sum() == 0 or mt.sum() == 0:
            continue
        xs.append(_resample(src.z[ms], n_points))
        ys.append(_resample(tgt.z[mt], n_points))
        eps.append(e)
    if not xs:
        h = src.z.shape[1]
        return np.zeros((0, n_points, h)), np.zeros((0, n_points, h)), np.zeros(0, int)
    return np.stack(xs), np.stack(ys), np.asarray(eps)


def cka_blocks(xb: np.ndarray, yb: np.ndarray) -> float:
    return linear_cka(xb.reshape(-1, xb.shape[-1]), yb.reshape(-1, yb.shape[-1]))


def cka_null(xb: np.ndarray, yb: np.ndarray, n_perm: int = 50, seed: int = 0) -> np.ndarray:
    """CKA after permuting the episode order of ``yb`` (derangements only)."""
    rng = np.random.default_rng(seed)
    E = len(xb)
    out = []
    for _ in range(n_perm):
        perm = rng.permutation(E)
        while E > 1 and np.any(perm == np.arange(E)):
            perm = rng.permutation(E)
        out.append(cka_blocks(xb, yb[perm]))
    return np.asarray(out)
