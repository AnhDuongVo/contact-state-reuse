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

    def _prev(self, prev_action: torch.Tensor) -> torch.Tensor:
        return prev_action if self.cfg.use_prev_action else torch.zeros_like(prev_action)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.core(self.encoder(obs, self._prev(prev_action)))
        return self.heads(z)

    def heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"action": torch.tanh(self.action_head(z)), "latent": z}

    def init_state(self, batch: int, device: torch.device | None = None):
        if self.cfg.core == "gru":
            return torch.zeros(self.cfg.n_layers, batch, self.cfg.hidden, device=device)
        return None

    def step(self, obs: torch.Tensor, prev_action: torch.Tensor, state):
        z, state = self.core.step(self.encoder(obs, self._prev(prev_action)), state)
        return self.heads(z), state


class PredictiveContactWorldModel(SequencePolicy):
    """Sequence trunk with self-supervised prediction heads.

    Heads on z_t:
      * next observation (extero + proprio, standardised)         [lambda_obs]
      * future force over ``force_horizon`` steps: mean, log-var  [lambda_force]
      * contact mode at t + ``mode_horizon`` (simulator labels)    [lambda_contact,
        only in configurations with ``contact_supervision: true``]
      * current observation (reconstruction control)              [lambda_recon,
        head built only when lambda_recon > 0]
      * action (behaviour cloning)

    With all lambdas at 0 this class is a sequence behaviour-cloning policy
    with the same trunk as the predictive variants (the "fixed trunk" design
    of the v0.2 factorial experiment). ``aux_stopgrad`` detaches the latent
    read by the prediction heads.
    """

    def __init__(self, layout: dict[str, slice], cfg: ModelConfig) -> None:
        super().__init__(layout, cfg)
        n_pred = (layout["extero"].stop - layout["extero"].start) + (layout["proprio"].stop - layout["proprio"].start)
        h = cfg.hidden
        self.obs_head = mlp(h, h, n_pred, n_hidden=1)
        self.force_head = mlp(h, h, 2 * cfg.force_horizon * 3, n_hidden=1)
        self.contact_head = mlp(h, h, N_MODES, n_hidden=1)
        if cfg.lambda_recon > 0:
            self.recon_head = mlp(h, h, layout["force"].stop, n_hidden=1)
        if cfg.aux_core:
            self.aux_rnn = TemporalCore(h, cfg.core, cfg.n_layers, cfg.n_heads, cfg.max_context, cfg.dropout)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.cfg.aux_core:
            return super().forward(obs, prev_action)
        e = self.encoder(obs, self._prev(prev_action))
        return self.heads(self.core(e), self.aux_rnn(e))

    def init_state(self, batch: int, device: torch.device | None = None):
        s = super().init_state(batch, device)
        return (s, super().init_state(batch, device)) if self.cfg.aux_core else s

    def step(self, obs: torch.Tensor, prev_action: torch.Tensor, state):
        if not self.cfg.aux_core:
            return super().step(obs, prev_action, state)
        e = self.encoder(obs, self._prev(prev_action))
        z, s_main = self.core.step(e, state[0])
        za, s_aux = self.aux_rnn.step(e, state[1])
        return self.heads(z, za), (s_main, s_aux)

    def heads(self, z: torch.Tensor, za: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = super().heads(z)
        if za is None:
            za = z.detach() if self.cfg.aux_stopgrad else z
        if self.cfg.lambda_recon > 0:
            out["recon"] = self.recon_head(za)
        out["next_obs"] = self.obs_head(za)
        f = self.force_head(za).unflatten(-1, (self.cfg.force_horizon, 2, 3))
        out["force_mean"] = f[..., 0, :]
        out["force_logvar"] = f[..., 1, :].clamp(-8.0, 6.0)
        out["contact_logits"] = self.contact_head(za)
        return out
