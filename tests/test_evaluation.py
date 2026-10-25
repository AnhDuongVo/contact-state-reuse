import json

import numpy as np
import pandas as pd

from contact_state_reuse.evaluation import (
    ForceCorruption,
    bootstrap_ci,
    evaluate_checkpoint,
    run_episodes,
    write_outputs,
)
from contact_state_reuse.evaluation.rollout import _CorruptionState
from contact_state_reuse.models import ModelConfig
from contact_state_reuse.training import TrainConfig, train_model


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci(np.r_[np.zeros(50), np.ones(50)])
    assert lo < 0.5 < hi and m == 0.5


def test_corruption_window_timing():
    sl = slice(0, 3)
    cs = _CorruptionState(ForceCorruption("first_contact_window", window=2), np.random.default_rng(0), sl)
    ones = np.ones(3, np.float32)
    assert cs.apply(ones, 1, False).sum() == 3
    assert cs.apply(ones, 2, True).sum() == 0  # contact onset
    assert cs.apply(ones, 3, False).sum() == 0
    assert cs.apply(ones, 4, False).sum() == 3  # window over


def test_expert_rollouts_deterministic():
    a = run_episodes("expert", 3, seed=1)
    b = run_episodes("expert", 3, seed=1)
    assert [r["success"] for r in a] == [r["success"] for r in b]
    assert [round(r["peak_force"], 6) for r in a] == [round(r["peak_force"], 6) for r in b]


def test_evaluation_outputs(tiny_dataset_path, tmp_path):
    res = train_model(
        [tiny_dataset_path],
        ModelConfig(kind="world_model", hidden=24),
        TrainConfig(steps=5, log_every=5, val_fraction=0.0),
        tmp_path,
        tag="m",
        verbose=False,
    )
    df = evaluate_checkpoint(
        res.checkpoint,
        ("source",),
        2,
        corruptions=(ForceCorruption(), ForceCorruption("zero")),
        labels={"condition": "m", "seed": 0},
        contact_auroc=True,
    )
    assert len(df) == 4 and set(df.corruption) == {"none", "zero"}
    summary = write_outputs(tmp_path / "run", df, ["condition", "domain", "corruption"])
    for f in ("episode_metrics.csv", "results.csv", "metrics.json"):
        assert (tmp_path / "run" / f).exists()
    assert {"mean", "ci_low", "ci_high", "n"} <= set(summary.columns)
    assert "summary" in json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(pd.read_csv(tmp_path / "run" / "episode_metrics.csv")) == 4
