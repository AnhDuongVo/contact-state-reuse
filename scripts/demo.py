"""Tiny end-to-end pipeline for `make demo` (a few minutes on CPU).

generate data -> train a sequence baseline and a predictive world model ->
evaluate on source and tight clearance -> representation analysis -> figures.
The numbers it prints are a pipeline check on very little data, not results.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from contact_state_reuse.analysis.pipeline import analyze_representations
from contact_state_reuse.analysis.plotting import plot_condition_metric, plot_episode_overview, plot_training_curves
from contact_state_reuse.data import DatasetConfig, generate_dataset, load_dataset
from contact_state_reuse.evaluation import evaluate_checkpoint, write_outputs
from contact_state_reuse.models import ModelConfig
from contact_state_reuse.training import TrainConfig, train_model
from contact_state_reuse.utils import from_dict, load_config, make_run_dir, seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    t0 = time.time()
    run = make_run_dir(cfg["output_root"], "demo")
    d = cfg["data"]
    print(f"[demo] output: {run}")
    train_path = generate_dataset(
        DatasetConfig(name="demo_train", domain="source", n_episodes=d["n_train"], seed=cfg["seed"], out_dir=d["dir"])
    )
    src = generate_dataset(
        DatasetConfig(
            name="demo_analysis_source",
            domain="source",
            n_episodes=d["n_analysis"],
            seed=777,
            p_perturbed=0.0,
            out_dir=d["dir"],
        )
    )
    tgt = generate_dataset(
        DatasetConfig(
            name="demo_analysis_tight",
            domain="tight_clearance",
            n_episodes=d["n_analysis"],
            seed=777,
            p_perturbed=0.0,
            out_dir=d["dir"],
        )
    )
    plot_episode_overview(load_dataset(train_path), run / "figures" / "dataset_episodes.png")
    tcfg = from_dict(TrainConfig, {**cfg["train"], "seed": cfg["seed"]})
    ckpts, hist, dfs = {}, {}, []
    for name, kind in (("sequence_bc", "sequence"), ("world_model", "world_model")):
        mcfg = from_dict(ModelConfig, {**cfg["model_defaults"], "kind": kind})
        res = train_model([train_path], mcfg, tcfg, run / "checkpoints", tag=name)
        ckpts[name], hist[name] = res.checkpoint, res.history
        ev = cfg["eval"]
        dfs.append(
            evaluate_checkpoint(
                res.checkpoint,
                tuple(ev["domains"]),
                ev["n_episodes"],
                ev["seed"],
                labels={"condition": name, "seed": cfg["seed"]},
                contact_auroc=True,
            )
        )
    plot_training_curves(hist, run / "figures" / "training_curves.png")
    summary = write_outputs(
        run,
        pd.concat(dfs, ignore_index=True),
        ["condition", "domain", "corruption"],
        {"note": "demo scale: pipeline check, not a result"},
    )
    plot_condition_metric(
        summary,
        "success",
        "domain",
        "condition",
        run / "figures" / "success.png",
        "demo: success rate (tiny data; not a result)",
        "success rate",
    )
    analyze_representations(ckpts["world_model"], src, tgt, run / "analysis", ablation_episodes=8)
    print(
        summary[summary.metric.isin(["success", "peak_force"])][
            ["condition", "domain", "metric", "mean", "ci_low", "ci_high", "n"]
        ].to_string(index=False)
    )
    print(
        f"[demo] finished in {time.time() - t0:.0f}s; figures in {run / 'figures'} and {run / 'analysis' / 'figures'}"
    )


if __name__ == "__main__":
    main()
