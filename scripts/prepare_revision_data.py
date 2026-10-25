"""Generate every dataset used by the v0.2 experiments before launching parallel shards.

python scripts/prepare_revision_data.py --scale medium
"""

from __future__ import annotations

import argparse

from contact_state_reuse.evaluation.experiments import ensure_dataset
from contact_state_reuse.evaluation.revision import prepare_reuse_v2_data
from contact_state_reuse.utils import load_config

CONDITION_EXPERIMENTS = ("E_factorial", "G_nonpriv", "G_search")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="medium")
    p.add_argument("--configs", default="configs/experiments")
    a = p.parse_args()
    for name in CONDITION_EXPERIMENTS:
        cfg = load_config(f"{a.configs}/{name}.yaml", a.scale)
        for cond in cfg["conditions"]:
            c = {**cfg, "data": {**cfg["data"], **cond.get("data", {})}}
            d = c["data"]
            print(ensure_dataset(c, "train", d.get("train_domain", "source"), d["n_train"], d.get("seed", 0)))
    cfg = load_config(f"{a.configs}/L_ablation.yaml", a.scale)
    d = cfg["data"]
    print(ensure_dataset(cfg, "analysis", "source", d["n_analysis"], d.get("analysis_seed", 777), p_perturbed=0.0))
    for k, v in prepare_reuse_v2_data(load_config(f"{a.configs}/R_reuse.yaml", a.scale)).items():
        print(k, v)


if __name__ == "__main__":
    main()
