"""Model construction, matched-capacity sizing and checkpoint I/O."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..data.dataset import Normalizer
from .common import ModelConfig, n_params
from .policies import PredictiveContactWorldModel, ReactivePolicy, SequencePolicy

_CLASSES = {"reactive": ReactivePolicy, "sequence": SequencePolicy, "world_model": PredictiveContactWorldModel}


def _build(layout: dict[str, slice], cfg: ModelConfig) -> nn.Module:
    return _CLASSES[cfg.kind](layout, cfg)


def build_model(layout: dict[str, slice], cfg: ModelConfig) -> tuple[nn.Module, ModelConfig]:
    """Build a model; if ``cfg.param_budget > 0`` choose ``hidden`` so the
    parameter count is closest to the budget (search over even widths, or
    multiples of ``n_heads`` for the Transformer core).

    Returns the model and the (possibly updated) config.
    """
    if cfg.param_budget > 0:
        best, best_err = cfg.hidden, float("inf")
        step = cfg.n_heads * 2 if cfg.core == "transformer" else 2
        for h in range(16, 513, step):
            n = n_params(_build(layout, replace(cfg, hidden=h)))
            err = abs(n - cfg.param_budget)
            if err < best_err:
                best, best_err = h, err
            if n > cfg.param_budget:
                break
        cfg = replace(cfg, hidden=best)
    return _build(layout, cfg), cfg


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: ModelConfig,
    normalizer: Normalizer,
    layout: dict[str, slice],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": cfg.to_dict(),
            "normalizer": normalizer.state_dict(),
            "layout": {k: [v.start, v.stop] for k, v in layout.items()},
            "n_params": n_params(model),
            "class": type(model).__name__,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Return ``(model, config, normalizer, checkpoint_dict)``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = dict(ckpt["model_config"])
    c["modalities"] = tuple(c["modalities"])
    cfg = ModelConfig(**c)
    layout = {k: slice(*v) for k, v in ckpt["layout"].items()}
    model = _build(layout, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, Normalizer.from_state_dict(ckpt["normalizer"]), ckpt
