"""Latent feature ablation.

A conservative intervention on the trained model, not a lesion and not a
causal claim about "units": selected latent dimensions are clamped to their
dataset mean during closed-loop control (readout and, for GRU cores, the
recurrent state), and task metrics are compared across three dimension sets
of equal size:

* ``contact``   dimensions with the largest |weight| in a contact-mode probe
* ``movement``  dimensions with the largest |weight| in a ridge regression
                predicting the expert action (a control for "important for
                control in general")
* ``random``    uniformly random dimensions (several draws)

If clamping contact-ranked dimensions hurts more than clamping equally many
movement-ranked or random dimensions, those dimensions carry information the
policy uses. The converse does not show the information is unused, because
of redundancy across dimensions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .probes import probe_weights


def rank_contact_dims(z: np.ndarray, modes: np.ndarray, seed: int = 0) -> np.ndarray:
    return np.argsort(-probe_weights(z, modes, seed=seed))


def rank_movement_dims(z: np.ndarray, actions: np.ndarray) -> np.ndarray:
    zs = StandardScaler().fit_transform(z)
    coef = Ridge(alpha=1.0).fit(zs, actions).coef_
    return np.argsort(-np.abs(coef).sum(0))


def clamp_fn(dims: np.ndarray, means: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
    idx = torch.as_tensor(np.asarray(dims), dtype=torch.long)
    vals = torch.as_tensor(np.asarray(means)[np.asarray(dims)], dtype=torch.float32)

    def fn(z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        z[:, idx] = vals.to(z.device)
        return z

    return fn


def dimension_sets(
    z: np.ndarray, modes: np.ndarray, actions: np.ndarray, k: int, n_random: int = 3, seed: int = 0
) -> dict[str, np.ndarray]:
    """Equal-size dimension sets for the ablation comparison."""
    if not 0 < k <= z.shape[1]:
        raise ValueError(f"k must be in [1, {z.shape[1]}]")
    rng = np.random.default_rng(seed)
    sets = {"contact": rank_contact_dims(z, modes, seed)[:k], "movement": rank_movement_dims(z, actions)[:k]}
    for i in range(n_random):
        sets[f"random{i}"] = rng.choice(z.shape[1], k, replace=False)
    return sets
