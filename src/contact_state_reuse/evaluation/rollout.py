"""Closed-loop evaluation: batched rollouts, evaluation-time force corruption,
optional latent-feature ablation, and per-episode metrics.

Evaluation seeds come from ``derive_seed(seed, "eval", i)`` and therefore never
coincide with dataset seeds (``derive_seed(seed, i)``). Episode ``i`` of an
evaluation set is identical for every model, so comparisons are paired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..data.dataset import Normalizer
from ..envs.expert import ExpertConfig, ScriptedExpert
from ..envs.insertion_env import ContactInsertionEnv, EnvConfig
from ..utils.seeding import derive_seed

CORRUPTION_MODES = ("none", "zero", "noise", "bias", "delay", "first_contact_window", "random_window")


@dataclass(frozen=True)
class ForceCorruption:
    """Evaluation-time corruption of the *observed* force channel.

    none                  unchanged
    zero                  force reading replaced by 0 for the whole episode
    noise                 extra Gaussian noise (std ``magnitude`` N, 1 % for torque)
    bias                  constant offset of ``magnitude`` N on Fx and Fz
    delay                 reading delayed by ``delay`` control steps
    first_contact_window  readings zeroed for ``window`` steps starting at the
                          step where the simulator registers first contact
                          (the harness uses privileged timing to *schedule* the
                          intervention; the policy never sees it)
    random_window         same-length zero window starting at a uniformly
                          random step in [0, ``random_max_start``); whether it
                          overlapped first contact is recorded per episode
    """

    mode: str = "none"
    window: int = 6
    magnitude: float = 2.0
    delay: int = 3
    random_max_start: int = 80

    def __post_init__(self) -> None:
        if self.mode not in CORRUPTION_MODES:
            raise ValueError(f"unknown corruption mode {self.mode!r}; choose from {CORRUPTION_MODES}")

    @property
    def name(self) -> str:
        return self.mode if self.mode in ("none", "zero") else f"{self.mode}"


class _CorruptionState:
    def __init__(self, spec: ForceCorruption, rng: np.random.Generator, force_slice: slice) -> None:
        self.spec, self.rng, self.sl = spec, rng, force_slice
        self.buffer: list[np.ndarray] = []
        self.window_start: int | None = None
        if spec.mode == "random_window":
            self.window_start = int(rng.integers(0, spec.random_max_start))
        self.first_contact_step: int | None = None

    def apply(self, obs: np.ndarray, t: int, first_contact_now: bool) -> np.ndarray:
        s = self.spec
        if s.mode == "none":
            return obs
        o = obs.copy()
        f = o[self.sl]
        if first_contact_now and self.first_contact_step is None:
            self.first_contact_step = t
            if s.mode == "first_contact_window":
                self.window_start = t
        if s.mode == "zero":
            f[:] = 0.0
        elif s.mode == "noise":
            f += self.rng.normal(size=3) * np.array([s.magnitude, s.magnitude, 0.01 * s.magnitude])
        elif s.mode == "bias":
            f[:2] += s.magnitude
        elif s.mode == "delay":
            self.buffer.append(f.copy())
            f[:] = self.buffer[max(0, len(self.buffer) - 1 - s.delay)]
        elif (
            s.mode in ("first_contact_window", "random_window")
            and self.window_start is not None
            and self.window_start <= t < self.window_start + s.window
        ):
            f[:] = 0.0
        o[self.sl] = f
        return o

    def overlapped_first_contact(self) -> bool | None:
        if self.spec.mode != "random_window" or self.first_contact_step is None:
            return None
        a0, a1 = self.window_start, self.window_start + self.spec.window
        b0, b1 = self.first_contact_step, self.first_contact_step + self.spec.window
        return bool(a0 < b1 and b0 < a1)


class PolicyAgent:
    """Runs a trained model over a batch of environments.

    ``latent_fn`` (optional) maps the latent z_t [N, H] to a modified latent
    before the heads read it; for GRU cores the same change is applied to the
    recurrent state, so the modification persists over time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer: Normalizer,
        device: torch.device | str = "cpu",
        latent_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model, self.norm, self.device, self.latent_fn = model.eval(), normalizer, torch.device(device), latent_fn

    def reset(self, n: int) -> None:
        self.state = self.model.init_state(n, self.device)
        self.prev_action = torch.zeros(n, 4, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = torch.from_numpy(self.norm(obs)).to(self.device)
        m = self.model
        if self.latent_fn is None or not hasattr(m, "core"):
            out, self.state = m.step(x, self.prev_action, self.state)
        else:
            z, state = m.core.step(m.encoder(x, self.prev_action), self.state)
            z = self.latent_fn(z)
            if m.cfg.core == "gru":
                state = state.clone()
                state[-1] = z
            self.state = state
            out = m.heads(z)
        self.prev_action = out["action"]
        return out["action"].cpu().numpy(), {k: v.cpu().numpy() for k, v in out.items()}


def run_episodes(
    agent: PolicyAgent | str,
    n_episodes: int,
    *,
    domain: str = "source",
    severity: float = 1.0,
    seed: int = 0,
    env_cfg: EnvConfig | None = None,
    corruption: ForceCorruption | None = None,
    batch: int = 32,
    record: bool = False,
    expert_cfg: ExpertConfig | None = None,
) -> list[dict[str, Any]]:
    """Roll out ``agent`` (a ``PolicyAgent`` or the string ``"expert"``) and return one dict per episode."""
    env_cfg = env_cfg or EnvConfig()
    corruption = corruption or ForceCorruption()
    results: list[dict[str, Any]] = []
    for b0 in range(0, n_episodes, batch):
        ids = list(range(b0, min(b0 + batch, n_episodes)))
        envs = [ContactInsertionEnv(env_cfg, domain=domain, severity=severity) for _ in ids]
        obs, infos, cors, experts, traces = [], [], [], [], []
        for env, i in zip(envs, ids):
            es = derive_seed(seed, "eval", i)
            o, info = env.reset(seed=es, options={"domain": domain, "severity": severity})
            cs = _CorruptionState(corruption, np.random.default_rng(derive_seed(es, "corrupt")), env.layout["force"])
            obs.append(cs.apply(o, 0, False))
            infos.append(info)
            cors.append(cs)
            if agent == "expert":
                ex = ScriptedExpert(expert_cfg or ExpertConfig())
                ex.reset(env, np.random.default_rng(derive_seed(es, "expert")), "nominal")
                experts.append(ex)
            traces.append(
                {
                    "modes": [info["contact_mode"]],
                    "force_norm": [info["contact_force_norm"]],
                    "rewards": [],
                    "latent": [],
                    "pred": [],
                    "obs": [obs[-1]],
                    "actions": [],
                }
            )
        if agent != "expert":
            agent.reset(len(ids))
        active = np.ones(len(ids), bool)
        stats = [{"return": 0.0, "impulse": 0.0, "peak": 0.0, "first": None} for _ in ids]
        final_info: list[dict[str, Any] | None] = [None] * len(ids)
        while active.any():
            if agent == "expert":
                acts = np.stack(
                    [ex.act(env) if a else np.zeros(4, np.float32) for ex, env, a in zip(experts, envs, active)]
                )
                extra: dict[str, np.ndarray] = {}
            else:
                acts, extra = agent.act(np.stack(obs))
            for j, env in enumerate(envs):
                if not active[j]:
                    continue
                if record:
                    if "latent" in extra:
                        traces[j]["latent"].append(extra["latent"][j])
                    if "force_mean" in extra:
                        traces[j]["pred"].append(extra["force_mean"][j])
                    traces[j]["actions"].append(acts[j])
                o, r, term, trunc, info = env.step(acts[j])
                st = stats[j]
                st["return"] += r
                st["impulse"] += info["contact_force_norm"] * env_cfg.control_dt if info["in_contact"] else 0.0
                st["peak"] = max(st["peak"], info["peak_force"])
                if info["first_contact"]:
                    st["first"] = info["t"]
                obs[j] = cors[j].apply(o, info["t"], info["first_contact"])
                traces[j]["modes"].append(info["contact_mode"])
                traces[j]["force_norm"].append(info["contact_force_norm"])
                traces[j]["rewards"].append(r)
                traces[j]["obs"].append(obs[j])
                if term or trunc:
                    active[j] = False
                    final_info[j] = info
        for j, (env, i) in enumerate(zip(envs, ids)):
            info, st = final_info[j], stats[j]
            row = {
                "episode": i,
                "domain": domain,
                "severity": severity,
                "corruption": corruption.name,
                "success": float(info["success"]),
                "return": st["return"],
                "peak_force": st["peak"],
                "impulse": st["impulse"],
                "force_violation": float(st["peak"] > env_cfg.safe_force),
                "time_to_success": env.t * env_cfg.control_dt if info["success"] else np.nan,
                "jam_or_failure": float(info["jam"] or info["damage"]),
                "steps": float(env.t),
                "first_contact_step": float(st["first"]) if st["first"] is not None else np.nan,
                "window_overlapped_first_contact": cors[j].overlapped_first_contact(),
                **{f"param_{k}": v for k, v in env.params.to_dict().items()},
            }
            if record:
                tr = traces[j]
                row["trace"] = {
                    "modes": np.asarray(tr["modes"]),
                    "force_norm": np.asarray(tr["force_norm"]),
                    "latent": np.asarray(tr["latent"]) if tr["latent"] else None,
                    "pred_force": np.asarray(tr["pred"]) if tr["pred"] else None,
                    "obs": np.asarray(tr["obs"]),
                    "actions": np.asarray(tr["actions"]),
                }
            results.append(row)
    return results


def first_contact_prediction_scores(
    results: list[dict[str, Any]], normalizer: Normalizer, force_slice: slice, horizon: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Score/label pairs for "first contact within the next ``horizon`` steps".

    Score: predicted future contact-force magnitude (max over the model's force
    horizon, de-standardised to newtons). Only steps before first contact are
    used. Requires ``run_episodes(..., record=True)`` with a world model.
    """
    scores, labels = [], []
    mu, sd = normalizer.mean[force_slice], normalizer.std[force_slice]
    for r in results:
        tr = r.get("trace")
        if tr is None or tr["pred_force"] is None:
            continue
        fc = r["first_contact_step"]
        pred = tr["pred_force"] * sd + mu  # [T, H, 3]
        mag = np.hypot(pred[..., 0], pred[..., 1]).max(-1)
        T = len(mag)
        last = int(fc) if np.isfinite(fc) else T
        for t in range(min(last, T)):
            scores.append(mag[t])
            labels.append(float(np.isfinite(fc) and fc - (t + 1) < horizon))
    return np.asarray(scores), np.asarray(labels)
