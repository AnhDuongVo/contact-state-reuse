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

CORRUPTION_MODES = (
    "none",
    "zero",
    "noise",
    "bias",
    "delay",
    "scale",
    "first_contact_window",
    "onset_offset_window",
    "onset_only",
    "hold_onset",
    "random_window",
)


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
    scale                 reading multiplied by ``magnitude`` for the whole episode
    onset_offset_window   zero window starting ``offset`` steps after first
                          contact (later-contact control for the onset window)
    onset_only            force kept only in the onset window, zeroed everywhere else
    hold_onset            during the onset window the last pre-contact reading is
                          held instead of zeroed (a "stale" rather than "zero" reading)

    For every episode the harness records how many steps were actually masked
    (``masked_steps``; 0 if the episode ended before the window) and the
    summed magnitude of the readings that were removed (``masked_force``), so
    windows can be compared by the amount of force signal they removed.
    """

    mode: str = "none"
    window: int = 6
    magnitude: float = 2.0
    delay: int = 3
    random_max_start: int = 80
    offset: int = 0

    def __post_init__(self) -> None:
        if self.mode not in CORRUPTION_MODES:
            raise ValueError(f"unknown corruption mode {self.mode!r}; choose from {CORRUPTION_MODES}")

    @property
    def name(self) -> str:
        if self.mode == "onset_offset_window":
            return f"onset+{self.offset}"
        if self.mode == "scale":
            return f"scale{self.magnitude:g}"
        return self.mode


class _CorruptionState:
    def __init__(self, spec: ForceCorruption, rng: np.random.Generator, force_slice: slice) -> None:
        self.spec, self.rng, self.sl = spec, rng, force_slice
        self.buffer: list[np.ndarray] = []
        self.window_start: int | None = None
        if spec.mode == "random_window":
            self.window_start = int(rng.integers(0, spec.random_max_start))
        self.first_contact_step: int | None = None
        self.masked_steps = 0
        self.masked_force = 0.0
        self.last_pre_contact: np.ndarray | None = None

    def _mask(self, f: np.ndarray, value: np.ndarray | float = 0.0) -> None:
        self.masked_steps += 1
        self.masked_force += float(np.hypot(f[0], f[1]))
        f[:] = value

    def apply(self, obs: np.ndarray, t: int, first_contact_now: bool) -> np.ndarray:
        s = self.spec
        if s.mode == "none":
            return obs
        o = obs.copy()
        f = o[self.sl]
        if first_contact_now and self.first_contact_step is None:
            self.first_contact_step = t
            if s.mode in ("first_contact_window", "hold_onset", "onset_only"):
                self.window_start = t
            elif s.mode == "onset_offset_window":
                self.window_start = t + s.offset
        in_window = self.window_start is not None and self.window_start <= t < self.window_start + s.window
        if s.mode == "zero":
            self._mask(f)
        elif s.mode == "scale":
            f *= s.magnitude
        elif s.mode == "onset_only":
            if not in_window:
                self._mask(f)
        elif s.mode == "hold_onset":
            if in_window and self.last_pre_contact is not None:
                self._mask(f, self.last_pre_contact)
            elif self.first_contact_step is None:
                self.last_pre_contact = f.copy()
        elif s.mode == "noise":
            f += self.rng.normal(size=3) * np.array([s.magnitude, s.magnitude, 0.01 * s.magnitude])
        elif s.mode == "bias":
            f[:2] += s.magnitude
        elif s.mode == "delay":
            self.buffer.append(f.copy())
            f[:] = self.buffer[max(0, len(self.buffer) - 1 - s.delay)]
        elif s.mode in ("first_contact_window", "random_window", "onset_offset_window") and in_window:
            self._mask(f)
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
        if self.latent_fn is not None and getattr(m.cfg, "aux_core", False):
            raise NotImplementedError("latent ablation is not implemented for models with a separate auxiliary core")
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
    compliance_override: float | None = None,
) -> list[dict[str, Any]]:
    """Roll out ``agent`` (a ``PolicyAgent`` or the string ``"expert"``) and return one dict per episode.

    ``compliance_override`` replaces the compliance command of every action
    (fixed-compliance control). Behaviour metrics per episode: mean commanded
    compliance (0 = soft, 1 = stiff) before and after first contact, peg speed
    over the last 5 steps before first contact, time to first contact.
    """
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
        stats = [
            {
                "return": 0.0,
                "impulse": 0.0,
                "peak": 0.0,
                "peak_sum": 0.0,
                "first": None,
                "c_pre": [],
                "c_post": [],
                "speeds": [],
            }
            for _ in ids
        ]
        final_info: list[dict[str, Any] | None] = [None] * len(ids)
        while active.any():
            if agent == "expert":
                acts = np.stack(
                    [ex.act(env) if a else np.zeros(4, np.float32) for ex, env, a in zip(experts, envs, active)]
                )
                extra: dict[str, np.ndarray] = {}
            else:
                acts, extra = agent.act(np.stack(obs))
            if compliance_override is not None:
                acts = acts.copy()
                acts[:, 3] = compliance_override
                if agent != "expert":
                    agent.prev_action = torch.as_tensor(acts, dtype=torch.float32, device=agent.device)
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
                st["peak_sum"] = max(st["peak_sum"], info.get("peak_force_sum", 0.0))
                c01 = (float(np.clip(acts[j][3], -1, 1)) + 1.0) / 2.0
                (st["c_post"] if st["first"] is not None else st["c_pre"]).append(c01)
                if st["first"] is None:
                    st["speeds"].append(float(np.hypot(*env.data.qvel[:2])))
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
                "peak_force_sum": st["peak_sum"],
                "masked_steps": float(cors[j].masked_steps),
                "masked_force": float(cors[j].masked_force),
                "compliance_pre_contact": float(np.mean(st["c_pre"])) if st["c_pre"] else np.nan,
                "compliance_in_contact": float(np.mean(st["c_post"])) if st["c_post"] else np.nan,
                "approach_speed": float(np.mean(st["speeds"][-5:])) if st["speeds"] else np.nan,
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
    results: list[dict[str, Any]],
    normalizer: Normalizer,
    force_slice: slice,
    horizon: int = 3,
    extero_slice: slice | None = None,
    with_baselines: bool = False,
):
    """Score/label pairs for "first contact within the next ``horizon`` steps".

    Score: predicted future contact-force magnitude (max over the model's force
    horizon, de-standardised to newtons). Only steps *before* first contact are
    used (frames already in contact are excluded), so the score measures
    anticipation, not contact persistence. Requires
    ``run_episodes(..., record=True)`` with a world model.

    With ``with_baselines`` also returns scores of two simple predictors on the
    same steps: the current observed force magnitude, and the negative
    estimated distance between peg tip and slot entrance (from the pose
    estimate), as a dict ``{"model": ..., "current_force": ..., "distance": ...}``.
    """
    scores, labels, cur, dist = [], [], [], []
    mu, sd = normalizer.mean[force_slice], normalizer.std[force_slice]
    for r in results:
        tr = r.get("trace")
        if tr is None or tr["pred_force"] is None:
            continue
        fc = r["first_contact_step"]
        if normalizer.force_transform != "none":
            continue
        pred = tr["pred_force"] * sd + mu  # [T, H, 3]
        mag = np.hypot(pred[..., 0], pred[..., 1]).max(-1)
        T = len(mag)
        last = int(fc) if np.isfinite(fc) else T
        obs = tr["obs"]
        for t in range(min(last, T)):
            scores.append(mag[t])
            labels.append(float(np.isfinite(fc) and fc - (t + 1) < horizon))
            if with_baselines:
                o = obs[t]
                cur.append(float(np.hypot(*o[force_slice][:2])))
                if extero_slice is not None:
                    e = o[extero_slice]
                    dist.append(-float(np.hypot(e[0], e[1])))
    if not with_baselines:
        return np.asarray(scores), np.asarray(labels)
    return {"model": np.asarray(scores), "current_force": np.asarray(cur), "distance": np.asarray(dist)}, np.asarray(
        labels
    )
