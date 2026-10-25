"""Multitask loss. Every term is returned separately for logging.

total = action
      + lambda_obs     * next-observation MSE        (self-supervised)
      + lambda_force   * future-force Gaussian NLL   (self-supervised)
      + lambda_contact * future contact-mode CE      (simulator-supervised;
                                                       only if contact_supervision)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.dataset import EpisodeTensors
from ..models.common import ModelConfig


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def compute_losses(out: dict[str, torch.Tensor], batch: EpisodeTensors, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    action = _masked_mean(((out["action"] - batch.target_action) ** 2).mean(-1), batch.action_mask)
    losses = {"action": action}
    total = action
    if "next_obs" in out:
        obs = _masked_mean(((out["next_obs"] - batch.next_obs) ** 2).mean(-1), batch.next_mask)
        mu, logvar = out["force_mean"], out["force_logvar"]
        nll = 0.5 * (logvar + (batch.future_force - mu) ** 2 * torch.exp(-logvar))
        force = _masked_mean(nll.mean(-1), batch.future_force_mask)
        force_mse = _masked_mean(((batch.future_force - mu) ** 2).mean(-1), batch.future_force_mask)
        ce = F.cross_entropy(out["contact_logits"].flatten(0, 1), batch.future_mode.flatten(), reduction="none")
        contact = _masked_mean(ce.view_as(batch.valid), batch.valid)
        losses.update(obs=obs, force=force, force_mse=force_mse, contact=contact)
        total = total + cfg.lambda_obs * obs + cfg.lambda_force * force
        if cfg.contact_supervision:
            total = total + cfg.lambda_contact * contact
    losses["total"] = total
    return losses
