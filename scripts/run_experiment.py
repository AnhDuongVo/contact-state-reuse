"""Run experiment A, B, C or D from its YAML config.

Examples:
    python scripts/run_experiment.py configs/experiments/A_modality.yaml --scale quick
    python scripts/run_experiment.py configs/experiments/D_reuse.yaml --scale full
"""

from __future__ import annotations

import argparse
import time

from contact_state_reuse.evaluation.experiments import run_experiment
from contact_state_reuse.utils import load_config, make_run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--scale", choices=["quick", "full"], default="quick")
    p.add_argument("--out", default=None, help="output root (default: config output_root)")
    a = p.parse_args()
    cfg = load_config(a.config, a.scale)
    run = make_run_dir(a.out or cfg.get("output_root", "outputs"), f"{cfg['experiment']}_{a.scale}")
    t0 = time.time()
    print(f"[{cfg['experiment']}|{a.scale}] writing to {run}", flush=True)
    run_experiment(cfg, run)
    print(f"[{cfg['experiment']}|{a.scale}] done in {time.time() - t0:.0f}s -> {run}", flush=True)


if __name__ == "__main__":
    main()
