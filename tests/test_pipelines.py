"""End-to-end pipeline checks at toy scale (marked slow; run with `make test`)."""

import pytest

from contact_state_reuse.evaluation.experiments import run_condition_experiment, run_force_timing, run_reuse


def _base(tmp_path):
    return {
        "experiment": "toy",
        "seeds": [0],
        "verbose": False,
        "data": {"dir": str(tmp_path / "data"), "n_train": 10, "n_analysis": 8, "seed": 0, "p_perturbed": 0.2},
        "model_defaults": {"hidden": 24},
        "train": {"steps": 5, "batch_episodes": 4, "log_every": 5, "val_fraction": 0.2},
        "eval": {"domains": ["source", "tight_clearance"], "n_episodes": 2, "seed": 1},
    }


@pytest.mark.slow
def test_condition_pipeline_writes_outputs(tmp_path):
    cfg = _base(tmp_path)
    cfg["conditions"] = [
        {"name": "seq_EP", "model": {"kind": "sequence", "modalities": ["extero", "proprio"]}},
        {"name": "seq_EPF", "model": {"kind": "sequence"}},
    ]
    cfg["comparisons"] = [["seq_EPF", "seq_EP"]]
    run = tmp_path / "run"
    (run / "figures").mkdir(parents=True)
    df = run_condition_experiment(cfg, run)
    assert len(df) == 2 * 2 * 2
    for f in ("episode_metrics.csv", "results.csv", "metrics.json", "paired_differences.csv", "figures/success.png"):
        assert (run / f).exists(), f


@pytest.mark.slow
def test_force_timing_pipeline(tmp_path):
    cfg = _base(tmp_path)
    cfg["conditions"] = [{"name": "wm", "model": {"kind": "world_model"}}]
    run = tmp_path / "run"
    (run / "figures").mkdir(parents=True)
    df = run_force_timing(cfg, run)
    assert set(df.corruption) == {"none", "first_contact_window", "random_window", "zero"}


@pytest.mark.slow
def test_reuse_pipeline(tmp_path):
    cfg = _base(tmp_path)
    cfg["conditions"] = [{"name": "wm", "model": {"kind": "world_model"}}]
    cfg["reuse"] = {"severities": [1.0], "budgets": [2], "finetune_steps": 3, "n_perm": 5, "n_boot": 5}
    run = tmp_path / "run"
    (run / "figures").mkdir(parents=True)
    run_reuse(cfg, run)
    for f in ("reuse_points.csv", "cka_by_phase.csv", "adaptation_curves.csv", "reuse_correlations.json"):
        assert (run / f).exists(), f
