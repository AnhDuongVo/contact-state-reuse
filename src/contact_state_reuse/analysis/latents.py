"""Latent extraction and event alignment.

Latents are computed by running a frozen model over *recorded* trajectories
(teacher forcing with the recorded observations and actions). Using recorded
oracle trajectories rather than the model's own rollouts means the phases
being compared are produced by the same behaviour in every domain, so latent
differences reflect the model's representation of the physics, not
differences in its own behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import Normalizer, TrajectoryData, build_episode_tensors


@dataclass
class LatentSet:
    """Row-aligned latents and analysis labels (one row per dataset row)."""

    z: np.ndarray  # [N, H]
    episode: np.ndarray  # [N] episode index within the source dataset
    t: np.ndarray  # [N]
    mode: np.ndarray  # [N] simulator-defined contact mode
    force_norm: np.ndarray  # [N] true contact force magnitude
    success: np.ndarray  # [N] episode outcome (bool)
    domain: np.ndarray  # [N] concrete domain name
    t_from_first_contact: np.ndarray  # [N] t - first contact step (NaN if never)
    t_to_end: np.ndarray  # [N] t - final step

    def subset(self, mask: np.ndarray) -> LatentSet:
        return LatentSet(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})

    def __len__(self) -> int:
        return len(self.z)


@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    normalizer: Normalizer,
    data: TrajectoryData,
    episodes: np.ndarray | None = None,
    batch: int = 64,
) -> LatentSet:
    """Run ``model`` over the given episodes and collect ``out['latent']``."""
    episodes = np.arange(data.n_episodes) if episodes is None else np.asarray(episodes)
    model.eval()
    device = next(model.parameters()).device
    zs, rows_all = [], []
    for i in range(0, len(episodes), batch):
        eps = episodes[i : i + batch]
        tens = build_episode_tensors(data, eps, normalizer)
        out = model(tens.obs.to(device), tens.prev_action.to(device))
        lat = out["latent"].cpu().numpy()
        for j, ep in enumerate(eps):
            rows = data.episode_rows(int(ep))
            n = rows.stop - rows.start
            zs.append(lat[j, :n])
            rows_all.append(np.arange(rows.start, rows.stop))
    rows = np.concatenate(rows_all)
    a = data.arrays
    ep = a["episode_id"][rows]
    t = a["t"][rows].astype(np.float64)
    first = np.full(data.n_episodes, np.nan)
    fc_rows = np.flatnonzero(a["first_contact"])
    first[a["episode_id"][fc_rows]] = a["t"][fc_rows]
    return LatentSet(
        z=np.concatenate(zs).astype(np.float32),
        episode=ep,
        t=t,
        mode=a["contact_mode"][rows].astype(int),
        force_norm=a["contact_force_norm"][rows],
        success=a["ep_success"][ep],
        domain=a["ep_domain"][ep],
        t_from_first_contact=t - first[ep],
        t_to_end=t - a["ep_length"][ep],
    )


def align(lat: LatentSet, event: str, before: int, after: int) -> tuple[np.ndarray, np.ndarray]:
    """Event-aligned latent array [E, before + after + 1, H] (NaN-padded) and episode ids.

    event: ``"start"`` (t = 0), ``"first_contact"`` or ``"end"`` (final step).
    Episodes without the event are skipped.
    """
    key = {"start": "t", "first_contact": "t_from_first_contact", "end": "t_to_end"}[event]
    rel = getattr(lat, key)
    eps = np.unique(lat.episode[np.isfinite(rel)])
    W = before + after + 1
    out = np.full((len(eps), W, lat.z.shape[1]), np.nan, np.float32)
    for i, e in enumerate(eps):
        m = (lat.episode == e) & np.isfinite(rel)
        r = rel[m].astype(int)
        keep = (r >= -before) & (r <= after)
        out[i, r[keep] + before] = lat.z[m][keep]
    return out, eps


def reference_latents(data: TrajectoryData, kind: str, normalizer: Normalizer | None = None) -> LatentSet:
    """Reference "representations" with the same rows and labels as ``extract_latents``.

    obs       standardised observation vector (all modalities)
    proprio   standardised proprioception only
    progress  time step and time since first contact (task progress clock)

    ``normalizer`` standardises the observations (fit on ``data`` if None).
    """
    a = data.arrays
    rows = np.arange(len(a["t"]))
    if normalizer is None:
        normalizer = Normalizer.fit(a["obs"])
    ep = a["episode_id"][rows]
    t = a["t"][rows].astype(np.float64)
    first = np.full(data.n_episodes, np.nan)
    fc_rows = np.flatnonzero(a["first_contact"])
    first[a["episode_id"][fc_rows]] = a["t"][fc_rows]
    tf = t - first[ep]
    if kind == "obs":
        z = normalizer(a["obs"])
    elif kind == "proprio":
        z = normalizer(a["obs"])[:, data.layout["proprio"]]
    elif kind == "progress":
        z = np.stack([t, np.nan_to_num(np.clip(tf, -40, 160), nan=-40.0)], 1)
    else:
        raise ValueError(f"unknown reference {kind!r}")
    return LatentSet(
        z=np.asarray(z, np.float32),
        episode=ep,
        t=t,
        mode=a["contact_mode"][rows].astype(int),
        force_norm=a["contact_force_norm"][rows],
        success=a["ep_success"][ep],
        domain=a["ep_domain"][ep],
        t_from_first_contact=tf,
        t_to_end=t - a["ep_length"][ep],
    )
