"""The three model families.

All models share the interface

    out = model(obs [B, L, D], prev_action [B, L, 4])
    out["action"]  [B, L, 4] in [-1, 1]
    out["latent"]  [B, L, H] (representation used for analysis)

and, for closed-loop control, ``state = model.init_state(B)`` followed by
``out, state = model.step(obs_t, prev_action_t, state)``.

ReactivePolicy               current observation only (no memory, no prev. action)
SequencePolicy               fused history -> temporal core -> action; no prediction heads
PredictiveContactWorldModel  same trunk + prediction heads (next observation,
                             future force mean/log-variance, optional future
                             contact mode); the action head reads z_t
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..envs.labels import N_MODES
from .common import N_ACTION, FusionEncoder, ModelConfig, TemporalCore, mlp


class ReactivePolicy(nn.Module):
    """Behaviour-cloning baseline on the current observation only."""

    def __init__(self, layout: dict[str, slice], cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FusionEncoder(layout, cfg.modalities, cfg.embed, cfg.hidden, use_action=False)
        self.head = mlp(cfg.hidden, cfg.hidden, N_ACTION, n_hidden=1, dropout=cfg.dropout)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)
        return {"action": torch.tanh(self.head(z)), "latent": z}

    def init_state(self, batch: int, device: torch.device | None = None):
        return None

    def step(self, obs: torch.Tensor, prev_action: torch.Tensor, state=None):
        out = self.forward(obs[:, None])
        return {k: v[:, 0] for k, v in out.items()}, None


class SequencePolicy(nn.Module):
    """Non-predictive sequence baseline: history encoder + action head only."""

    def __init__(self, layout: dict[str, slice], cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FusionEncoder(layout, cfg.modalities, cfg.embed, cfg.hidden, use_action=True)
        self.core = TemporalCore(cfg.hidden, cfg.core, cfg.n_layers, cfg.n_heads, cfg.max_context, cfg.dropout)
        self.action_head = mlp(cfg.hidden, cfg.hidden, N_ACTION, n_hidden=1, dropout=cfg.dropout)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.core(self.encoder(obs, prev_action))
        return self.heads(z)

    def heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"action": torch.tanh(self.action_head(z)), "latent": z}

    def init_state(self, batch: int, device: torch.device | None = None):
        if self.cfg.core == "gru":
            return torch.zeros(self.cfg.n_layers, batch, self.cfg.hidden, device=device)
        return None

    def step(self, obs: torch.Tensor, prev_action: torch.Tensor, state):
        z, state = self.core.step(self.encoder(obs, prev_action), state)
        return self.heads(z), state


class PredictiveContactWorldModel(SequencePolicy):
    """Sequence trunk with self-supervised prediction heads.

    Heads on z_t:
      * next observation (extero + proprio, standardised)         [lambda_obs]
      * future force over ``force_horizon`` steps: mean, log-var  [lambda_force]
      * contact mode at t + ``mode_horizon`` (simulator labels)    [lambda_contact,
        only in configurations with ``contact_supervision: true``]
      * action (behaviour cloning)
    """

    def __init__(self, layout: dict[str, slice], cfg: ModelConfig) -> None:
        super().__init__(layout, cfg)
        n_pred = (layout["extero"].stop - layout["extero"].start) + (layout["proprio"].stop - layout["proprio"].start)
        h = cfg.hidden
        self.obs_head = mlp(h, h, n_pred, n_hidden=1)
        self.force_head = mlp(h, h, 2 * cfg.force_horizon * 3, n_hidden=1)
        self.contact_head = mlp(h, h, N_MODES, n_hidden=1)

    def heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().heads(z)
        out["next_obs"] = self.obs_head(z)
        f = self.force_head(z).unflatten(-1, (self.cfg.force_horizon, 2, 3))
        out["force_mean"] = f[..., 0, :]
        out["force_logvar"] = f[..., 1, :].clamp(-8.0, 6.0)
        out["contact_logits"] = self.contact_head(z)
        return out
