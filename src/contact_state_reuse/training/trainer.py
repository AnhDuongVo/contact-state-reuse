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
from .losses import compute_losses


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
    splits = []
    for i, d in enumerate(datasets):
        tr, va = episode_split(d.n_episodes, cfg.val_fraction, cfg.seed + 1000 * i)
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
        normalizer = Normalizer.fit(obs)
    kw = dict(force_horizon=mcfg.force_horizon, mode_horizon=mcfg.mode_horizon, bc_on_failures=cfg.bc_on_failures)
    train = _cat_tensors([build_episode_tensors(d, tr, normalizer, **kw) for d, (tr, _) in zip(datasets, splits)])
    vals = [build_episode_tensors(d, va, normalizer, **kw) for d, (_, va) in zip(datasets, splits) if len(va)]
    val = _cat_tensors(vals) if vals else None
    return train, val, normalizer, splits


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
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps:
            row: dict[str, Any] = {"step": step, "time_s": round(time.time() - t0, 2)}
            row.update({f"train_{k}": float(v.detach()) for k, v in losses.items()})
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
