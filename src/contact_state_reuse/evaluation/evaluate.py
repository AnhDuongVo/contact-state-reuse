"""Evaluate checkpoints on named domains and write the standard output files.

Outputs in a run directory:
    episode_metrics.csv  one row per evaluated episode
    results.csv          tidy summary: group columns, metric, mean, 95 % bootstrap CI, n
    metrics.json         the same summary nested as JSON, plus run metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..envs.insertion_env import EnvConfig
from ..models.factory import load_checkpoint
from ..utils.io import get_device, git_version, save_json
from .metrics import EPISODE_METRICS, summarize
from .rollout import ForceCorruption, PolicyAgent, first_contact_prediction_scores, run_episodes

EVAL_DOMAINS = ("source", "shifted_friction", "tight_clearance", "sensor_noise", "occlusion")


def evaluate_checkpoint(
    checkpoint: str | Path,
    domains: tuple[str, ...] = EVAL_DOMAINS,
    n_episodes: int = 50,
    seed: int = 0,
    corruptions: tuple[ForceCorruption, ...] = (ForceCorruption(),),
    severity: float = 1.0,
    env_cfg: EnvConfig | None = None,
    device: str = "cpu",
    labels: dict[str, Any] | None = None,
    contact_auroc: bool = False,
    compliance_override: float | None = None,
) -> pd.DataFrame:
    """Roll out one checkpoint on every (domain, corruption) pair; returns per-episode rows."""
    model, mcfg, norm, ckpt = load_checkpoint(checkpoint, get_device(device))
    agent = PolicyAgent(model, norm, get_device(device))
    rows = []
    for dom in domains:
        for cor in corruptions:
            res = run_episodes(
                agent,
                n_episodes,
                domain=dom,
                severity=severity,
                seed=seed,
                env_cfg=env_cfg,
                corruption=cor,
                record=contact_auroc and mcfg.kind == "world_model" and mcfg.lambda_force > 0,
                compliance_override=compliance_override,
            )
            aurocs = {"first_contact_auroc": np.nan, "auroc_current_force": np.nan, "auroc_distance": np.nan}
            # only meaningful when the force head was trained
            if contact_auroc and mcfg.kind == "world_model" and mcfg.lambda_force > 0:
                from sklearn.metrics import roc_auc_score

                layout = {k: slice(*v) for k, v in ckpt["layout"].items()}
                sc, y = first_contact_prediction_scores(
                    res, norm, layout["force"], extero_slice=layout["extero"], with_baselines=True
                )
                if len(np.unique(y)) == 2:
                    for key, name in (
                        ("model", "first_contact_auroc"),
                        ("current_force", "auroc_current_force"),
                        ("distance", "auroc_distance"),
                    ):
                        if len(sc[key]) == len(y):
                            aurocs[name] = float(roc_auc_score(y, sc[key]))
            for r in res:
                r.pop("trace", None)
                r.update(aurocs)
                r.update(labels or {})
                rows.append(r)
    return pd.DataFrame(rows)


def write_outputs(
    run_dir: str | Path, episodes: pd.DataFrame, group_cols: list[str], extra_meta: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Write episode_metrics.csv, results.csv and metrics.json; return the summary."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(run_dir / "episode_metrics.csv", index=False)
    metrics = [m for m in EPISODE_METRICS if m in episodes]
    summary = summarize(episodes, group_cols, metrics)
    summary.to_csv(run_dir / "results.csv", index=False)
    nested: dict[str, Any] = {}
    for _, r in summary.iterrows():
        key = " | ".join(f"{c}={r[c]}" for c in group_cols)
        nested.setdefault(key, {})[r["metric"]] = {
            "mean": r["mean"],
            "ci95": [r["ci_low"], r["ci_high"]],
            "n": int(r["n"]),
        }
    save_json(
        {"code_version": git_version(), "group_columns": group_cols, "summary": nested, **(extra_meta or {})},
        run_dir / "metrics.json",
    )
    return summary
