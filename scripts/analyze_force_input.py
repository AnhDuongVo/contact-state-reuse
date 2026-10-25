"""Why does force as a plain input hurt behaviour cloning? Two diagnostics.

1. Force statistics: observed force readings in the (successful) training
   demonstrations versus readings the trained policies encounter in their own
   closed-loop rollouts (source domain), overall and by simulator phase.
2. Input sensitivity: on held-out demonstration episodes (teacher forcing),
   the gradient of the action with respect to the current force reading
   (per newton), on free-space and in-contact steps.

    python scripts/analyze_force_input.py --run outputs/E_factorial_medium/run1 --out results/revision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from contact_state_reuse.data.dataset import build_episode_tensors, load_dataset
from contact_state_reuse.envs.labels import MODE_NAMES
from contact_state_reuse.evaluation.rollout import PolicyAgent, run_episodes
from contact_state_reuse.models.factory import load_checkpoint

CONDITIONS = ("bc_F", "obs_F", "obs_F_ftgt", "bc_F_log", "bc_F_binary", "bc_F_aug", "obs_F_ftgt_aug", "bc_F_noprev")


def force_mag(obs: np.ndarray, sl: slice) -> np.ndarray:
    return np.hypot(obs[:, sl.start], obs[:, sl.start + 1])


def quantiles(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "frac_above_1N": float(np.mean(x > 1.0)),
        "q50": float(np.quantile(x, 0.5)),
        "q90": float(np.quantile(x, 0.9)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def sensitivity(model, norm, data, episodes, force_sl: slice) -> dict:
    """Mean ||d a_t / d F_t|| (per newton, Fx and Fz) on free-space and contact steps."""
    t = build_episode_tensors(data, episodes, norm)
    obs = t.obs.clone().requires_grad_(True)
    out = model(obs, t.prev_action)["action"]  # [E, L, 4]
    valid = t.valid.bool()
    sd = torch.as_tensor(norm.std[force_sl.start : force_sl.start + 2])
    grads = []
    for k in range(4):
        g = torch.autograd.grad(out[..., k][valid].sum(), obs, retain_graph=True)[0]
        grads.append(g[..., force_sl.start : force_sl.start + 2] / sd)  # d a_k / d F (N)
    g = torch.stack(grads, -1)  # [E, L, 2, 4]
    gn = torch.linalg.matrix_norm(g).detach().numpy()  # Frobenius over (Fx, Fz) x action dims
    contact = np.isin(t.mode.numpy(), [1, 2, 3])
    v = valid.numpy()
    return {
        "free_space": float(gn[v & ~contact].mean()),
        "in_contact": float(gn[v & contact].mean()),
        "all": float(gn[v].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/E_factorial_medium/run1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out", default="results/revision")
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    ck0 = torch.load(run / "checkpoints" / f"bc_F_seed{a.seeds[0]}.pt", weights_only=False)
    data = load_dataset(ck0["datasets"][0])
    sl = data.layout["force"]
    ok = data.arrays["ep_success"].astype(bool)
    val_eps = np.asarray([e for e in ck0["val_episodes"][0] if ok[e]])
    train_eps = np.asarray([e for e in ck0["train_episodes"][0] if ok[e]])
    rows = np.concatenate(
        [np.arange(data.episode_rows(int(e)).start, data.episode_rows(int(e)).stop) for e in train_eps]
    )
    demo = force_mag(data.arrays["obs"][rows], sl)
    demo_modes = data.arrays["contact_mode"][rows]
    res: dict = {"demonstrations": {"all": quantiles(demo)}, "rollouts": {}, "sensitivity": {}}
    for m in range(4):
        res["demonstrations"][MODE_NAMES[m]] = quantiles(demo[demo_modes == m])
    q99 = res["demonstrations"]["all"]["q99"]
    sens_rows = []
    for cond in CONDITIONS:
        roll, roll_modes = [], []
        for seed in a.seeds:
            ck = run / "checkpoints" / f"{cond}_seed{seed}.pt"
            if not ck.exists():
                continue
            model, mcfg, norm, _ = load_checkpoint(ck)
            for r in run_episodes(PolicyAgent(model, norm), a.episodes, domain="source", seed=123, record=True):
                roll.append(force_mag(r["trace"]["obs"], sl))
                roll_modes.append(r["trace"]["modes"])
            s = sensitivity(model, norm, data, val_eps, sl)
            sens_rows.append({"condition": cond, "seed": seed, **s})
        if roll:
            x, mo = np.concatenate(roll), np.concatenate(roll_modes)
            res["rollouts"][cond] = {"all": quantiles(x), "frac_above_demo_q99": float(np.mean(x > q99))}
            for m in range(4):
                if (mo == m).any():
                    res["rollouts"][cond][MODE_NAMES[m]] = quantiles(x[mo == m])
        print(cond, "done", flush=True)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out / "force_input_sensitivity.csv", index=False)
    res["sensitivity"] = {
        c: {
            k: [float(sens[sens.condition == c][k].mean()), float(sens[sens.condition == c][k].std(ddof=1))]
            for k in ("free_space", "in_contact")
        }
        for c in sens.condition.unique()
    }
    (out / "force_input_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["sensitivity"], indent=1))


if __name__ == "__main__":
    main()
