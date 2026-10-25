"""Evaluate one or more checkpoints on named domains, optionally with force corruption.

Examples:
    python scripts/evaluate.py --checkpoints outputs/train/sequence_seed0.pt --episodes 30
    python scripts/evaluate.py --checkpoints a.pt b.pt --domains source tight_clearance \
        --corruptions none zero first_contact_window random_window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from contact_state_reuse.analysis.plotting import plot_condition_metric
from contact_state_reuse.evaluation import (
    CORRUPTION_MODES,
    EVAL_DOMAINS,
    ForceCorruption,
    evaluate_checkpoint,
    write_outputs,
)
from contact_state_reuse.utils import make_run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--domains", nargs="+", default=list(EVAL_DOMAINS))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123, help="evaluation seed (disjoint from data seeds)")
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--corruptions", nargs="+", default=["none"], choices=CORRUPTION_MODES)
    p.add_argument("--out", default="outputs")
    p.add_argument("--name", default="evaluate")
    a = p.parse_args()
    run = make_run_dir(a.out, a.name)
    cors = tuple(ForceCorruption(m) for m in a.corruptions)
    dfs = [
        evaluate_checkpoint(
            c,
            tuple(a.domains),
            a.episodes,
            a.seed,
            cors,
            a.severity,
            labels={"condition": Path(c).stem},
            contact_auroc=True,
        )
        for c in a.checkpoints
    ]
    summary = write_outputs(run, pd.concat(dfs, ignore_index=True), ["condition", "domain", "corruption"])
    for m in ("success", "peak_force"):
        plot_condition_metric(
            summary[summary.corruption == a.corruptions[0]],
            m,
            "domain",
            "condition",
            run / "figures" / f"{m}.png",
            f"{m} (mean, 95% bootstrap CI)",
        )
    print(summary[summary.metric.isin(["success", "peak_force"])].to_string(index=False))
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
