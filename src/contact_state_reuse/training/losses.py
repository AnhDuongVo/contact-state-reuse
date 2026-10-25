"""Multitask loss. Every term is returned separately for logging.

total = action
      + lambda_obs     * next-observation MSE        (self-supervised)
      + lambda_force   * future-force Gaussian NLL   (self-supervised)
      + lambda_contact * future contact-mode CE      (simulator-supervised;
                                                       only if contact_supervision)
      + lambda_recon   * current-observation MSE     (non-predictive control)

Scaling: actions are in [-1, 1]; observation and force targets are
standardised per feature with the training-set normaliser. Each term is a
mean over its feature dimensions and over valid (unpadded) time steps of the
batch, so all terms are per-step, per-dimension averages. Equal weights do not
imply equal gradient contributions; the trainer logs the gradient norm of each
weighted term on the shared trunk (``grad_<term>``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.dataset import EpisodeTensors
from ..models.common import ModelConfig


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def weighted_terms(losses: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """The weighted terms that enter ``total`` (for gradient-norm logging)."""
    w = {"action": 1.0, "obs": cfg.lambda_obs, "force": cfg.lambda_force, "recon": cfg.lambda_recon}
    if cfg.contact_supervision:
        w["contact"] = cfg.lambda_contact
    return {k: losses[k] * v for k, v in w.items() if k in losses and v > 0}


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
        if "recon" in out:
            recon = _masked_mean(((out["recon"] - batch.obs) ** 2).mean(-1), batch.valid)
            losses["recon"] = recon
            total = total + cfg.lambda_recon * recon
    losses["total"] = total
    return losses
