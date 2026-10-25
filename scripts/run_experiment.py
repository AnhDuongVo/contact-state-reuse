"""Run experiment A, B, C or D from its YAML config.

Examples:
    python scripts/run_experiment.py configs/experiments/A_modality.yaml --scale quick
    python scripts/run_experiment.py configs/experiments/D_reuse.yaml --scale full
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from contact_state_reuse.evaluation.experiments import run_experiment
from contact_state_reuse.utils import load_config, make_run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--scale", choices=["quick", "medium", "full"], default="quick")
    p.add_argument("--out", default=None, help="output root (default: config output_root)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the most recent run of this experiment/scale (finished models and evaluations are reused)",
    )
    p.add_argument("--run-dir", default=None, help="explicit run directory (created if missing)")
    p.add_argument(
        "--shard",
        default=None,
        help="i/n: run only every n-th (seed, condition) job starting at i; combine with --run-dir, then run "
        "once more without --shard to aggregate",
    )
    p.add_argument("--threads", type=int, default=int(os.environ.get("CSR_THREADS", "1")))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config(a.config, a.scale)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        cfg["_shard"] = (i, n)
    root = Path(a.out or cfg.get("output_root", "outputs")) / f"{cfg['experiment']}_{a.scale}"
    previous = sorted(p for p in root.glob("*") if p.is_dir()) if a.resume else []
    if a.run_dir:
        run = Path(a.run_dir)
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = previous[-1] if previous else make_run_dir(root.parent, root.name)
    t0 = time.time()
    print(f"[{cfg['experiment']}|{a.scale}] writing to {run}", flush=True)
    run_experiment(cfg, run)
    print(f"[{cfg['experiment']}|{a.scale}] done in {time.time() - t0:.0f}s -> {run}", flush=True)


if __name__ == "__main__":
    main()
