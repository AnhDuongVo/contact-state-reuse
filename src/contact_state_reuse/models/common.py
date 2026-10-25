"""Shared building blocks: model configuration, fusion encoder, temporal cores."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

MODALITIES = ("extero", "proprio", "force")
N_ACTION = 4


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and objective of one model.

    kind              ``reactive`` | ``sequence`` | ``world_model``
    modalities        observation groups given to the model (others are zeroed)
    core              temporal core for sequence/world models: ``gru`` | ``transformer``
    param_budget      if > 0, ``hidden`` is adjusted so the total parameter count
                      is as close as possible to this budget (matched capacity)
    lambda_obs        weight of next-observation prediction (self-supervised)
    lambda_force      weight of future-force prediction (self-supervised)
    lambda_contact    weight of future contact-mode prediction. Uses
                      simulator-defined labels, so it requires
                      ``contact_supervision: true`` to make that explicit.
    """

    kind: str = "world_model"
    modalities: tuple[str, ...] = MODALITIES
    hidden: int = 128
    embed: int = 32
    core: str = "gru"
    n_layers: int = 1
    n_heads: int = 4
    max_context: int = 32
    dropout: float = 0.0
    force_horizon: int = 5
    mode_horizon: int = 3
    lambda_obs: float = 1.0
    lambda_force: float = 1.0
    lambda_contact: float = 0.0
    contact_supervision: bool = False
    param_budget: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("reactive", "sequence", "world_model"):
            raise ValueError(f"unknown model kind {self.kind!r}")
        bad = set(self.modalities) - set(MODALITIES)
        if bad:
            raise ValueError(f"unknown modalities {sorted(bad)}")
        if self.core not in ("gru", "transformer"):
            raise ValueError(f"unknown core {self.core!r}")
        if self.lambda_contact > 0 and not self.contact_supervision:
            raise ValueError(
                "lambda_contact > 0 uses simulator contact labels; set contact_supervision: true "
                "to mark this configuration as simulator-supervised"
            )
        if self.kind != "world_model" and (self.lambda_obs or self.lambda_force or self.lambda_contact):
            object.__setattr__(self, "lambda_obs", 0.0)
            object.__setattr__(self, "lambda_force", 0.0)
            object.__setattr__(self, "lambda_contact", 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modalities"] = list(self.modalities)
        return d


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class FusionEncoder(nn.Module):
    """Modality-specific projection + LayerNorm, masking, then a fused MLP.

    Inputs are already standardised per feature (``data.Normalizer``). Each
    modality gets its own linear embedding and LayerNorm so that a modality
    with large variance cannot dominate the fused representation. Modalities
    not in ``modalities`` are multiplied by a zero mask, so all observation
    ablations share identical parameters. ``set_mask`` changes the mask at
    evaluation time (modality dropout).
    """

    def __init__(
        self, layout: dict[str, slice], modalities: tuple[str, ...], embed: int, hidden: int, use_action: bool
    ) -> None:
        super().__init__()
        self.layout = {k: (v.start, v.stop) for k, v in layout.items()}
        self.proj = nn.ModuleDict({m: nn.Linear(self.layout[m][1] - self.layout[m][0], embed) for m in MODALITIES})
        self.norm = nn.ModuleDict({m: nn.LayerNorm(embed) for m in MODALITIES})
        self.use_action = use_action
        in_dim = embed * len(MODALITIES) + (embed if use_action else 0)
        if use_action:
            self.act_proj = nn.Linear(N_ACTION, embed)
        self.fuse = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.register_buffer("mask", torch.tensor([1.0 if m in modalities else 0.0 for m in MODALITIES]))

    def set_mask(self, modalities: tuple[str, ...]) -> None:
        self.mask.copy_(torch.tensor([1.0 if m in modalities else 0.0 for m in MODALITIES]))

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        parts = []
        for i, m in enumerate(MODALITIES):
            s, e = self.layout[m]
            parts.append(self.norm[m](self.proj[m](obs[..., s:e])) * self.mask[i])
        if self.use_action:
            if prev_action is None:
                raise ValueError("this encoder expects the previous action")
            parts.append(self.act_proj(prev_action))
        return self.fuse(torch.cat(parts, dim=-1))


class TemporalCore(nn.Module):
    """GRU (default) or small causal Transformer over fused features.

    ``forward`` processes a whole sequence [B, L, H]; ``step`` processes one
    time step with an explicit state for closed-loop rollouts.
    """

    def __init__(
        self,
        hidden: int,
        kind: str = "gru",
        n_layers: int = 1,
        n_heads: int = 4,
        max_context: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kind, self.hidden, self.max_context = kind, hidden, max_context
        if kind == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(hidden, n_heads, 2 * hidden, dropout, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.pos = nn.Parameter(torch.zeros(1, max_context, hidden))
            nn.init.normal_(self.pos, std=0.02)

    def _tf_window(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device), diagonal=1)
        return self.tf(x + self.pos[:, :n], mask=mask, is_causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            return self.rnn(x)[0]
        # sliding causal window: each position attends to at most max_context steps
        L = x.shape[1]
        if self.max_context >= L:
            return self._tf_window(x)
        outs = []
        for t in range(L):
            lo = max(0, t + 1 - self.max_context)
            outs.append(self._tf_window(x[:, lo : t + 1])[:, -1])
        return torch.stack(outs, 1)

    def step(self, x: torch.Tensor, state):
        if self.kind == "gru":
            out, h = self.rnn(x[:, None], state)
            return out[:, 0], h
        buf = x[:, None] if state is None else torch.cat([state, x[:, None]], 1)[:, -self.max_context :]
        return self._tf_window(buf)[:, -1], buf


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
