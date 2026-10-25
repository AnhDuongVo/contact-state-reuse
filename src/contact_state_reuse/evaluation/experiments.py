"""Experiment pipelines A-D (see docs/methods.md for the protocol).

Each ``run_*`` function takes a loaded YAML config (``utils.config.load_config``)
and a run directory, trains what it needs, evaluates, writes the standard
output files and figures, and returns the per-episode DataFrame.

A  modality ablation        same capacity, with vs without force, several model kinds
B  objective ablation       sequence BC vs predictive objectives (+ optional simulator-supervised aux)
C  force timing             force zeroed around first contact vs a matched random window
D  representation reuse     pre-train on source, measure reuse under graded shifts,
                            fine-tune, and relate reuse to adaptation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..analysis.latents import extract_latents
from ..analysis.plotting import (
    plot_cka_by_phase,
    plot_condition_metric,
    plot_confusion,
    plot_curves,
    plot_pca_trajectories,
    plot_training_curves,
    plot_xy_scatter,
)
from ..analysis.probes import cross_domain_probe, grouped_cv_probe
from ..analysis.reuse import reuse_score
from ..analysis.stats import spearman_with_ci
from ..data.dataset import load_dataset
from ..data.generate import DatasetConfig, generate_dataset
from ..envs.insertion_env import EnvConfig
from ..models.common import ModelConfig
from ..models.factory import load_checkpoint
from ..training.trainer import TrainConfig, train_model
from ..utils.config import from_dict
from ..utils.io import save_json
from ..utils.seeding import seed_everything
from .evaluate import evaluate_checkpoint, write_outputs
from .metrics import paired_difference, summarize
from .rollout import ForceCorruption


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (avoids the numpy 1/2 trapz/trapezoid rename)."""
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


# ----------------------------------------------------------------- helpers
def ensure_dataset(
    cfg: dict[str, Any],
    name: str,
    domain: str,
    n_episodes: int,
    seed: int,
    severity: float = 1.0,
    p_perturbed: float | None = None,
) -> Path:
    """Generate a dataset once and cache it under ``cfg['data']['dir']``."""
    d = cfg["data"]
    p_pert = d.get("p_perturbed", 0.2) if p_perturbed is None else p_perturbed
    tag = f"{name}_{domain}_sev{severity:g}_n{n_episodes}_s{seed}_p{p_pert:g}_d{d.get('dart_sigma', 0.1):g}"
    path = Path(d["dir"]) / f"{tag}.npz"
    if not path.exists():
        generate_dataset(
            DatasetConfig(
                name=tag,
                domain=domain,
                severity=severity,
                n_episodes=n_episodes,
                seed=seed,
                p_perturbed=p_pert,
                dart_sigma=d.get("dart_sigma", 0.1),
                out_dir=d["dir"],
            ),
            EnvConfig(),
            verbose=False,
        )
    return path


def model_config(cfg: dict[str, Any], overrides: dict[str, Any]) -> ModelConfig:
    merged = {**cfg.get("model_defaults", {}), **overrides}
    if "modalities" in merged:
        merged["modalities"] = tuple(merged["modalities"])
    return from_dict(ModelConfig, merged)


def train_config(cfg: dict[str, Any], seed: int, **overrides: Any) -> TrainConfig:
    return from_dict(TrainConfig, {**cfg.get("train", {}), "seed": seed, **overrides})


def corruption_list(specs: list[dict[str, Any]] | None) -> tuple[ForceCorruption, ...]:
    return tuple(ForceCorruption(**s) for s in (specs or [{"mode": "none"}]))


def _save_config(cfg: dict[str, Any], run_dir: Path) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _train_and_eval_conditions(
    cfg: dict[str, Any], run_dir: Path, conditions: list[dict[str, Any]], corruptions: tuple[ForceCorruption, ...]
) -> tuple[pd.DataFrame, dict]:
    ev = cfg["eval"]
    rows, histories = [], {}
    data_cfg = cfg["data"]
    train_path = ensure_dataset(
        cfg, "train", data_cfg.get("train_domain", "source"), data_cfg["n_train"], data_cfg.get("seed", 0)
    )
    for seed in cfg["seeds"]:
        for cond in conditions:
            mcfg = model_config(cfg, cond["model"])
            tag = f"{cond['name']}_seed{seed}"
            res = train_model(
                [train_path],
                mcfg,
                train_config(cfg, seed),
                run_dir / "checkpoints",
                tag=tag,
                verbose=cfg.get("verbose", True),
            )
            histories[tag] = res.history
            df = evaluate_checkpoint(
                res.checkpoint,
                tuple(ev["domains"]),
                ev["n_episodes"],
                seed=ev.get("seed", 123),
                corruptions=corruptions,
                contact_auroc=ev.get("contact_auroc", False),
                labels={
                    "condition": cond["name"],
                    "seed": seed,
                    "n_params": res.n_params,
                    "hidden": res.model_config.hidden,
                    "kind": res.model_config.kind,
                },
            )
            rows.append(df)
            print(
                f"  {tag}: params={res.n_params} success(source)="
                f"{df[df.domain == ev['domains'][0]].success.mean():.2f}",
                flush=True,
            )
    return pd.concat(rows, ignore_index=True), histories


# --------------------------------------------------------------- A and B
def run_condition_experiment(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    """Shared driver for experiments A (modalities) and B (objectives)."""
    seed_everything(cfg["seeds"][0])
    _save_config(cfg, run_dir)
    eps, histories = _train_and_eval_conditions(
        cfg, run_dir, cfg["conditions"], corruption_list(cfg["eval"].get("corruptions"))
    )
    summary = write_outputs(
        run_dir, eps, ["condition", "domain", "corruption"], {"experiment": cfg["experiment"], "seeds": cfg["seeds"]}
    )
    figs = run_dir / "figures"
    for metric, label in (
        ("success", "success rate"),
        ("peak_force", "peak contact force [N]"),
        ("force_violation", f"P(peak > {EnvConfig().safe_force:g} N)"),
        ("jam_or_failure", "jam / failure rate"),
    ):
        plot_condition_metric(
            summary,
            metric,
            "domain",
            "condition",
            figs / f"{metric}.png",
            f"{cfg['experiment']}: {label} (mean, 95% bootstrap CI over episodes)",
            label,
            x_order=list(cfg["eval"]["domains"]),
            hue_order=[c["name"] for c in cfg["conditions"]],
        )
    plot_training_curves(histories, figs / "training_curves.png")
    comparisons = []
    for a, b in cfg.get("comparisons", []):
        for dom in cfg["eval"]["domains"]:
            sub = eps[eps.domain == dom]
            for m in ("success", "peak_force"):
                comparisons.append({"domain": dom, **paired_difference(sub, "condition", a, b, m)})
    if comparisons:
        pd.DataFrame(comparisons).to_csv(run_dir / "paired_differences.csv", index=False)
    return eps


# ----------------------------------------------------------------------- C
def run_force_timing(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    """Experiment C: zero the force channel around first contact vs a matched random window."""
    seed_everything(cfg["seeds"][0])
    _save_config(cfg, run_dir)
    win = cfg["eval"].get("window", 6)
    corr = (
        ForceCorruption("none"),
        ForceCorruption("first_contact_window", window=win),
        ForceCorruption("random_window", window=win, random_max_start=cfg["eval"].get("random_max_start", 80)),
        ForceCorruption("zero"),
    )
    eps, _ = _train_and_eval_conditions(cfg, run_dir, cfg["conditions"], corr)
    summary = write_outputs(
        run_dir, eps, ["condition", "domain", "corruption"], {"experiment": cfg["experiment"], "window": win}
    )
    nonoverlap = eps[(eps.corruption != "random_window") | (eps.window_overlapped_first_contact == False)]  # noqa: E712
    summarize(nonoverlap, ["condition", "domain", "corruption"]).to_csv(
        run_dir / "results_random_nonoverlap.csv", index=False
    )
    rows = []
    for cond in eps.condition.unique():
        for dom in eps.domain.unique():
            sub = eps[(eps.condition == cond) & (eps.domain == dom)]
            for m in ("success", "peak_force", "force_violation"):
                rows.append(
                    {
                        "condition": cond,
                        "domain": dom,
                        **paired_difference(sub, "corruption", "first_contact_window", "random_window", m),
                    }
                )
    pd.DataFrame(rows).to_csv(run_dir / "paired_differences.csv", index=False)
    figs = run_dir / "figures"
    for metric in ("success", "peak_force", "force_violation"):
        for dom in eps.domain.unique():
            s = summary[summary.domain == dom]
            plot_condition_metric(
                s,
                metric,
                "corruption",
                "condition",
                figs / f"{metric}_{dom}.png",
                f"C: {metric}, {dom} (force zeroed for {win} steps)",
                metric,
            )
    return eps


# ----------------------------------------------------------------------- D
def run_reuse(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    """Experiment D: representation reuse under graded shifts and adaptation."""
    seed_everything(cfg["seeds"][0])
    _save_config(cfg, run_dir)
    d, ev, rcfg = cfg["data"], cfg["eval"], cfg["reuse"]
    shift = rcfg.get("shift", "tight_clearance")
    figs, ck = run_dir / "figures", run_dir / "checkpoints"
    train_path = ensure_dataset(cfg, "train", "source", d["n_train"], d.get("seed", 0))
    # paired analysis sets: same seed, same initial conditions, different physics
    an_seed = d.get("analysis_seed", 777)
    src_an = load_dataset(ensure_dataset(cfg, "analysis", "source", d["n_analysis"], an_seed, p_perturbed=0.0))
    shifted_an = {
        s: load_dataset(ensure_dataset(cfg, "analysis", shift, d["n_analysis"], an_seed, severity=s, p_perturbed=0.0))
        for s in rcfg["severities"]
    }
    ft_paths = {
        s: ensure_dataset(cfg, "finetune", shift, max(rcfg["budgets"]) + 8, d.get("finetune_seed", 555), severity=s)
        for s in rcfg["severities"]
    }
    points, episodes, cka_rows, curves = [], [], [], []
    for seed in cfg["seeds"]:
        for cond in cfg["conditions"]:
            mcfg = model_config(cfg, cond["model"])
            tag = f"{cond['name']}_seed{seed}"
            pre = train_model(
                [train_path], mcfg, train_config(cfg, seed), ck, tag=tag, verbose=cfg.get("verbose", True)
            )
            model, _, norm, _ = load_checkpoint(pre.checkpoint)
            lat_src = extract_latents(model, norm, src_an)
            probe_src = grouped_cv_probe(lat_src.z, lat_src.mode, lat_src.episode, seed=seed)
            if seed == cfg["seeds"][0]:
                plot_pca_trajectories(lat_src, figs / f"pca_{cond['name']}_source.png", title=f"{cond['name']}, source")
                plot_confusion(
                    probe_src.confusion,
                    figs / f"probe_confusion_{cond['name']}_source.png",
                    f"{cond['name']}: within-source probe (episode-grouped CV)",
                )
            for sev in rcfg["severities"]:
                lat_t = extract_latents(model, norm, shifted_an[sev])
                rr = reuse_score(
                    lat_src, lat_t, n_perm=rcfg.get("n_perm", 50), n_boot=rcfg.get("n_boot", 200), seed=seed
                )
                xprobe = cross_domain_probe(lat_src.z, lat_src.mode, lat_t.z, lat_t.mode, seed=seed)
                for p, v in rr.per_phase.items():
                    cka_rows.append({"model": cond["name"], "seed": seed, "severity": sev, "phase": p, **v})
                # zero-shot and adaptation
                zs = evaluate_checkpoint(
                    pre.checkpoint,
                    (shift,),
                    ev["n_episodes"],
                    seed=ev.get("seed", 123),
                    severity=sev,
                    labels={"condition": cond["name"], "seed": seed, "budget": 0},
                )
                curve = [{"budget": 0, "success": zs.success.mean(), "peak_force": zs.peak_force.mean()}]
                episodes.append(zs.assign(severity=sev))
                for budget in rcfg["budgets"]:
                    ft = train_model(
                        [ft_paths[sev]],
                        mcfg,
                        train_config(
                            cfg,
                            seed,
                            steps=rcfg["finetune_steps"],
                            max_episodes=budget,
                            val_fraction=0.0,
                            lr=rcfg.get("finetune_lr", 3e-4),
                        ),
                        ck,
                        init_checkpoint=pre.checkpoint,
                        tag=f"{tag}_sev{sev:g}_ft{budget}",
                        verbose=False,
                    )
                    e = evaluate_checkpoint(
                        ft.checkpoint,
                        (shift,),
                        ev["n_episodes"],
                        seed=ev.get("seed", 123),
                        severity=sev,
                        labels={"condition": cond["name"], "seed": seed, "budget": budget},
                    )
                    episodes.append(e.assign(severity=sev))
                    curve.append({"budget": budget, "success": e.success.mean(), "peak_force": e.peak_force.mean()})
                c = pd.DataFrame(curve).assign(model=cond["name"], seed=seed, severity=sev)
                curves.append(c)
                b = c.budget.to_numpy(float)
                auc = (
                    float(_trapz(c.success.to_numpy(float), b) / (b.max() - b.min()))
                    if b.max() > b.min()
                    else float(c.success.mean())
                )
                points.append(
                    {
                        "model": cond["name"],
                        "seed": seed,
                        "severity": sev,
                        "reuse": rr.score,
                        "reuse_ci_low": rr.ci[0],
                        "reuse_ci_high": rr.ci[1],
                        "probe_source_bacc": probe_src.balanced_accuracy,
                        "probe_cross_domain_bacc": xprobe.balanced_accuracy,
                        "zero_shot_success": curve[0]["success"],
                        "adaptation_auc": auc,
                        "final_success": curve[-1]["success"],
                        "peak_force_change": curve[-1]["peak_force"] - curve[0]["peak_force"],
                    }
                )
                print(
                    f"  {tag} sev={sev:g}: reuse={rr.score:.3f} xprobe={xprobe.balanced_accuracy:.2f} "
                    f"zero-shot={curve[0]['success']:.2f} auc={auc:.2f}",
                    flush=True,
                )
    eps = pd.concat(episodes, ignore_index=True)
    write_outputs(run_dir, eps, ["condition", "severity", "budget"], {"experiment": cfg["experiment"], "shift": shift})
    pts = pd.DataFrame(points)
    pts.to_csv(run_dir / "reuse_points.csv", index=False)
    cka = pd.DataFrame(cka_rows)
    cka.to_csv(run_dir / "cka_by_phase.csv", index=False)
    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_csv(run_dir / "adaptation_curves.csv", index=False)
    corr = {
        "reuse_vs_adaptation_auc": spearman_with_ci(pts.reuse, pts.adaptation_auc),
        "reuse_vs_peak_force_change": spearman_with_ci(pts.reuse, pts.peak_force_change),
        "reuse_vs_zero_shot_success": spearman_with_ci(pts.reuse, pts.zero_shot_success),
        "note": "Spearman correlations across (model, seed, severity) points; associations, not causal effects.",
    }
    save_json(corr, run_dir / "reuse_correlations.json")
    sev_max = max(rcfg["severities"])
    plot_cka_by_phase(
        cka[cka.severity == sev_max].groupby(["model", "phase"])[["cka", "null_mean", "null_sd"]].mean().reset_index(),
        figs / "cka_by_phase.png",
        f"source vs {shift} (severity {sev_max:g})",
    )
    r = corr["reuse_vs_adaptation_auc"]
    plot_xy_scatter(
        pts,
        "reuse",
        "adaptation_auc",
        "model",
        figs / "reuse_vs_adaptation.png",
        "reuse score vs adaptation efficiency",
        f"Spearman rho={r['rho']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}], n={r['n']}",
    )
    for sev in rcfg["severities"]:
        plot_curves(
            curves_df[curves_df.severity == sev],
            "budget",
            "success",
            "model",
            figs / f"adaptation_curve_sev{sev:g}.png",
            f"adaptation on {shift} (severity {sev:g})",
            "success rate",
        )
    return eps


def run_experiment(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    kind = cfg.get("kind")
    runners = {"conditions": run_condition_experiment, "force_timing": run_force_timing, "reuse": run_reuse}
    if kind not in runners:
        raise ValueError(f"config 'kind' must be one of {sorted(runners)}, got {kind!r}")
    return runners[kind](cfg, run_dir)


__all__ = ["ensure_dataset", "run_condition_experiment", "run_experiment", "run_force_timing", "run_reuse"]
