"""Evaluate the scripted controllers as baseline policies.

privileged      the data-generation oracle (true slot pose and contact wrench)
non_privileged  the same controller using only the observation vector
search          the non-privileged controller with the hole-search behaviour

    python scripts/eval_scripted.py --out results/revision/scripted_baselines.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from contact_state_reuse.envs.expert import ExpertConfig
from contact_state_reuse.evaluation.rollout import run_episodes

CONTROLLERS = {
    "privileged": ExpertConfig(),
    "non_privileged": ExpertConfig(privileged=False, aim_error=0.001),
    "search": ExpertConfig(privileged=False, search=True, aim_error=0.001),
}
DOMAINS = (
    "source",
    "shifted_friction",
    "tight_clearance",
    "sensor_noise",
    "occlusion",
    "search",
    "search+shifted_friction",
    "search+tight_clearance",
    "search+sensor_noise",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="results/revision/scripted_baselines.csv")
    a = p.parse_args()
    rows = []
    for name, cfg in CONTROLLERS.items():
        for dom in DOMAINS:
            for r in run_episodes("expert", a.episodes, domain=dom, seed=a.seed, expert_cfg=cfg):
                r["controller"] = name
                rows.append(r)
            print(name, dom, flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_name(out.stem + "_episodes.csv"), index=False)
    s = df.groupby(["controller", "domain"])[["success", "peak_force", "jam_or_failure", "steps"]].mean()
    s.reset_index().to_csv(out, index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
