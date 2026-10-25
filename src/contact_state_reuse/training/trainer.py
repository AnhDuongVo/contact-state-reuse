"""Training loop shared by all models (fixed update budget, episode batches)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.dataset import (
    EpisodeTensors,
    Normalizer,
    TrajectoryData,
    build_episode_tensors,
    episode_split,
    load_dataset,
)
from ..models.common import ModelConfig, n_params
from ..models.factory import build_model, load_checkpoint, save_checkpoint
from ..utils.io import get_device, save_json
from ..utils.seeding import seed_everything
from .losses import compute_losses, weighted_terms


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1500
    batch_episodes: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.15
    seed: int = 0
    device: str = "auto"
    log_every: int = 100
    bc_on_failures: bool = False
    max_episodes: int = 0  # if > 0, use only this many training episodes (sample-efficiency)
    # v0.2
    episode_filter: str = "all"  # "success": train every loss on successful episodes only
    force_aug: float = 0.0  # fraction of training episodes added again with corrupted force input
    log_grad_norms: bool = True  # gradient norm of each weighted loss term on the shared trunk


@dataclass
class TrainResult:
    checkpoint: Path
    history: list[dict[str, float]]
    model_config: ModelConfig
    n_params: int
    train_episodes: int


def _cat_tensors(parts: list[EpisodeTensors]) -> EpisodeTensors:
    L = max(p.obs.shape[1] for p in parts)

    def pad(t: torch.Tensor) -> torch.Tensor:
        if t.shape[1] == L:
            return t
        shape = list(t.shape)
        shape[1] = L - t.shape[1]
        return torch.cat([t, torch.zeros(shape, dtype=t.dtype)], 1)

    return EpisodeTensors(
        **{k: torch.cat([pad(getattr(p, k)) for p in parts], 0) for k in EpisodeTensors.__dataclass_fields__}
    )


def prepare_data(
    datasets: list[TrajectoryData], cfg: TrainConfig, mcfg: ModelConfig, normalizer: Normalizer | None = None
):
    """Episode-level split of each dataset, normaliser fit on training episodes."""
    if cfg.episode_filter not in ("all", "success"):
        raise ValueError(f"episode_filter must be 'all' or 'success', got {cfg.episode_filter!r}")
    splits = []
    for i, d in enumerate(datasets):
        tr, va = episode_split(d.n_episodes, cfg.val_fraction, cfg.seed + 1000 * i)
        if cfg.episode_filter == "success":
            ok = d.arrays["ep_success"].astype(bool)
            tr, va = tr[ok[tr]], va[ok[va]]
        if cfg.max_episodes > 0:
            tr = np.random.default_rng(cfg.seed).permutation(tr)[: cfg.max_episodes]
            tr = np.sort(tr)
        splits.append((tr, va))
    if normalizer is None:
        obs = np.concatenate(
            [
                d.arrays["obs"][
                    np.concatenate([np.arange(d.episode_rows(int(e)).start, d.episode_rows(int(e)).stop) for e in tr])
                ]
                for d, (tr, _) in zip(datasets, splits)
            ]
        )
        fs = datasets[0].layout["force"]
        normalizer = Normalizer.fit(obs, mcfg.force_transform, (fs.start, fs.stop))
    kw = dict(force_horizon=mcfg.force_horizon, mode_horizon=mcfg.mode_horizon, bc_on_failures=cfg.bc_on_failures)
    train = _cat_tensors([build_episode_tensors(d, tr, normalizer, **kw) for d, (tr, _) in zip(datasets, splits)])
    if cfg.force_aug > 0:
        train = add_force_corruption(train, normalizer, datasets[0].layout["force"], cfg.force_aug, cfg.seed)
    vals = [build_episode_tensors(d, va, normalizer, **kw) for d, (_, va) in zip(datasets, splits) if len(va)]
    val = _cat_tensors(vals) if vals else None
    return train, val, normalizer, splits


# sensor-fault model of the "sensor_noise" domain (domains.SHIFTS), used for
# fault-aware training: noise 1-2 N, bias 1-2 N, 2-step delay, 10 % dropout
FAULT_AUG = {"noise": (1.0, 2.0), "bias": (1.0, 2.0), "delay": 2, "dropout": 0.1}


def add_force_corruption(
    train: EpisodeTensors, norm: Normalizer, force_slice: slice, frac: float, seed: int
) -> EpisodeTensors:
    """Fault-aware training: append copies of a fraction of the training
    episodes whose force *input* is corrupted with the sensor-fault model.
    Prediction targets (future force) keep the original readings."""
    if norm.force_transform != "none":
        raise ValueError("force augmentation is implemented for untransformed force only")
    rng = np.random.default_rng(seed + 4242)
    n = len(train)
    ids = np.sort(rng.permutation(n)[: max(1, int(round(frac * n)))])
    copy = train.index(torch.as_tensor(ids))
    obs = copy.obs.clone().numpy()
    s, e = force_slice.start, force_slice.stop
    mu, sd = norm.mean[s:e], norm.std[s:e]
    scale = np.array([1.0, 1.0, 0.01])
    for i in range(len(ids)):
        L = int(copy.valid[i].sum())
        raw = obs[i, :L, s:e] * sd + mu
        d = rng.normal(size=3)
        d /= np.linalg.norm(d) + 1e-12
        bias = d * rng.uniform(*FAULT_AUG["bias"]) * scale
        noise = rng.normal(size=(L, 3)) * rng.uniform(*FAULT_AUG["noise"]) * scale
        k = FAULT_AUG["delay"]
        delayed = np.concatenate([np.repeat(raw[:1], k, 0), raw[: L - k]]) if k < L else raw
        out = delayed + bias + noise
        out[rng.random(L) < FAULT_AUG["dropout"]] = 0.0
        obs[i, :L, s:e] = (out - mu) / sd
    copy = EpisodeTensors(**{**{k_: getattr(copy, k_) for k_ in EpisodeTensors.__dataclass_fields__}})
    copy.obs = torch.from_numpy(obs.astype(np.float32))
    return _cat_tensors([train, copy])


def _trunk_params(model) -> list[torch.nn.Parameter]:
    mods = [model.encoder] + ([model.core] if hasattr(model, "core") else [])
    return [p for m in mods for p in m.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate_losses(
    model, data: EpisodeTensors, mcfg: ModelConfig, device: torch.device, batch: int = 64
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for i in range(0, len(data), batch):
        b = data.index(torch.arange(i, min(i + batch, len(data)))).to(device)
        losses = compute_losses(model(b.obs, b.prev_action), b, mcfg)
        w = int(b.obs.shape[0])
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v) * w
        n += w
    return {k: v / max(n, 1) for k, v in sums.items()}


def train_model(
    dataset_paths: list[str | Path],
    mcfg: ModelConfig,
    cfg: TrainConfig,
    out_dir: str | Path,
    layout: dict[str, slice] | None = None,
    init_checkpoint: str | Path | None = None,
    tag: str = "model",
    verbose: bool = True,
) -> TrainResult:
    """Train one model and write ``<out_dir>/<tag>.pt`` and ``<tag>_history.json``.

    ``init_checkpoint`` fine-tunes an existing model (its normaliser is reused
    so the input scaling does not change between pre-training and adaptation).
    """
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    datasets = [load_dataset(p) for p in dataset_paths]
    layout = layout or datasets[0].layout
    normalizer = None
    if init_checkpoint is not None:
        model, mcfg_loaded, normalizer, _ = load_checkpoint(init_checkpoint, device)
        mcfg = mcfg_loaded
    else:
        model, mcfg = build_model(layout, mcfg)
        model.to(device)
    train, val, normalizer, splits = prepare_data(datasets, cfg, mcfg, normalizer)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1), eta_min=cfg.lr * 0.05)
    gen = torch.Generator().manual_seed(cfg.seed)
    history: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        ids = torch.randint(0, len(train), (min(cfg.batch_episodes, len(train)),), generator=gen)
        b = train.index(ids).to(device)
        losses = compute_losses(model(b.obs, b.prev_action), b, mcfg)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at step {step}: { ({k: float(v) for k, v in losses.items()}) }")
        log_now = step % cfg.log_every == 0 or step == cfg.steps
        grad_norms: dict[str, float] = {}
        if log_now and cfg.log_grad_norms:
            params = _trunk_params(model)
            for k, term in weighted_terms(losses, mcfg).items():
                gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                grad_norms[f"grad_{k}"] = float(
                    torch.sqrt(sum(((g.detach() ** 2).sum() for g in gs if g is not None), torch.tensor(0.0)))
                )
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if log_now:
            row: dict[str, Any] = {"step": step, "time_s": round(time.time() - t0, 2)}
            row.update({f"train_{k}": float(v.detach()) for k, v in losses.items()})
            row.update(grad_norms)
            if val is not None:
                row.update({f"val_{k}": v for k, v in evaluate_losses(model, val, mcfg, device).items()})
            history.append(row)
            if verbose:
                keys = [k for k in ("train_action", "val_action", "val_obs", "val_force", "val_contact") if k in row]
                print(f"[{tag}] step {step}/{cfg.steps} " + " ".join(f"{k}={row[k]:.4f}" for k in keys), flush=True)
    out_dir = Path(out_dir)
    ckpt = out_dir / f"{tag}.pt"
    model.cpu()
    save_checkpoint(
        ckpt,
        model,
        mcfg,
        normalizer,
        layout,
        extra={
            "train_config": asdict(cfg),
            "datasets": [str(p) for p in dataset_paths],
            "train_episodes": [s[0].tolist() for s in splits],
            "val_episodes": [s[1].tolist() for s in splits],
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        },
    )
    save_json(
        {
            "tag": tag,
            "model_config": mcfg.to_dict(),
            "train_config": asdict(cfg),
            "n_params": n_params(model),
            "history": history,
        },
        out_dir / f"{tag}_history.json",
    )
    return TrainResult(ckpt, history, mcfg, n_params(model), int(sum(len(s[0]) for s in splits)))
