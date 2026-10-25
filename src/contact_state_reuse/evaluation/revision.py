"""Experiments added in v0.2 (revision after review).

T  force-timing controls      evaluation-only, on checkpoints of experiment E:
                              onset window vs later windows, onset-only, hold vs
                              zero, random windows, plus fixed-compliance runs
L  latent ablation            coordinate-wise clamping on every seed
R  reuse and adaptation       exploratory; a shift with a recoverable zero-shot
                              deficit, gain-over-zero-shot metrics, a
                              source-domain fine-tuning control, reference
                              representations, held-out-pair probes and the
                              permutation-null distributions

Every function reads a YAML config (see ``configs/experiments/``), caches
each evaluation under ``<run_dir>/evals`` and each fine-tuned model under
``<run_dir>/checkpoints``, and supports ``cfg['_shard'] = (i, n)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..analysis.ablation import clamp_fn, dimension_sets
from ..analysis.latents import extract_latents, reference_latents
from ..analysis.probes import cross_domain_probe, grouped_cv_probe, paired_heldout_probe
from ..analysis.reuse import reuse_score
from ..analysis.similarity import PHASES_FOR_REUSE
from ..data.dataset import Normalizer, load_dataset
from ..models.factory import build_model, load_checkpoint
from ..utils.io import load_json, save_json
from .experiments import (
    _eval_cached,
    _save_config,
    _train_cached,
    _trapz,
    corruption_list,
    ensure_dataset,
    train_config,
    write_outputs,
)
from .rollout import PolicyAgent, run_episodes


def _jobs(cfg: dict[str, Any], items: list) -> list:
    shard = cfg.get("_shard")
    return items[shard[0] :: shard[1]] if shard is not None else items


def _ckpt(cfg: dict[str, Any], cond: str, seed: int) -> Path:
    path = Path(cfg["source_run"]) / "checkpoints" / f"{cond}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run experiment E first)")
    return path


# ----------------------------------------------------------------------- T
def run_timing_v2(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    _save_config(cfg, run_dir)
    ev = cfg["eval"]
    corr = corruption_list(cfg["corruptions"])
    rows = []
    jobs = [(s, c) for s in cfg["seeds"] for c in cfg["conditions"]]
    for seed, cond in _jobs(cfg, jobs):
        ck = _ckpt(cfg, cond, seed)
        lab = {"condition": cond, "seed": seed, "compliance_override": "policy"}
        rows.append(
            _eval_cached(
                run_dir,
                f"{cond}_seed{seed}",
                ck,
                tuple(ev["domains"]),
                ev["n_episodes"],
                seed=ev.get("seed", 123),
                corruptions=corr,
                labels=lab,
            )
        )
        for c in cfg.get("compliance_overrides", []):
            rows.append(
                _eval_cached(
                    run_dir,
                    f"{cond}_seed{seed}_c{c:g}",
                    ck,
                    tuple(ev["domains"]),
                    ev["n_episodes"],
                    seed=ev.get("seed", 123),
                    compliance_override=float(c),
                    labels={**lab, "compliance_override": f"{c:g}"},
                )
            )
        print(f"  T {cond} seed {seed} done", flush=True)
    eps = pd.concat(rows, ignore_index=True)
    if cfg.get("_shard") is None:
        write_outputs(
            run_dir,
            eps,
            ["condition", "domain", "corruption", "compliance_override"],
            {"experiment": cfg["experiment"]},
        )
    return eps


# ----------------------------------------------------------------------- L
def run_ablation_seeds(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    """Clamp equal-size sets of latent dimensions (contact-ranked,
    movement-ranked, random draws) during closed-loop control, for every seed."""
    _save_config(cfg, run_dir)
    d, ab = cfg["data"], cfg["ablation"]
    an = load_dataset(
        ensure_dataset(cfg, "analysis", "source", d["n_analysis"], d.get("analysis_seed", 777), p_perturbed=0.0)
    )
    rows = []
    jobs = [(s, c) for s in cfg["seeds"] for c in cfg["conditions"]]
    for seed, cond in _jobs(cfg, jobs):
        key = Path(run_dir) / "evals" / f"{cond}_seed{seed}.csv"
        if key.exists():
            rows.append(pd.read_csv(key))
            continue
        model, _, norm, _ = load_checkpoint(_ckpt(cfg, cond, seed))
        lat = extract_latents(model, norm, an)
        k = max(1, int(round(ab.get("fraction", 0.25) * lat.z.shape[1])))
        sets = dimension_sets(lat.z, lat.mode, an.arrays["expert_action"], k, n_random=ab.get("n_random", 3), seed=seed)
        means = lat.z.mean(0)
        out = []
        for dom in ab["domains"]:
            for name, dims in [("none", np.array([], int)), *sets.items()]:
                agent = PolicyAgent(model, norm, latent_fn=clamp_fn(dims, means) if len(dims) else None)
                for r in run_episodes(agent, ab["n_episodes"], domain=dom, seed=cfg["eval"].get("seed", 123)):
                    r.update(
                        ablation="random" if name.startswith("random") else name,
                        ablation_draw=name,
                        condition=cond,
                        seed=seed,
                        k=k,
                        hidden=lat.z.shape[1],
                    )
                    out.append(r)
        df = pd.DataFrame(out)
        key.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(key, index=False)
        rows.append(df)
        print(f"  L {cond} seed {seed} done", flush=True)
    eps = pd.concat(rows, ignore_index=True)
    if cfg.get("_shard") is None:
        write_outputs(run_dir, eps, ["condition", "domain", "ablation"], {"experiment": cfg["experiment"]})
    return eps


# ----------------------------------------------------------------------- R
def _reuse_rows(lat_s, lat_t, tag: dict, rcfg: dict, seed: int) -> tuple[list[dict], list[dict]]:
    """Reuse score under the default settings and the sensitivity variants."""
    summary, phase_rows = [], []
    variants = [("default", 4, PHASES_FOR_REUSE)]
    for npts in rcfg.get("sensitivity_points", [2, 8]):
        variants.append((f"points{npts}", npts, PHASES_FOR_REUSE))
    variants.append(("contact_phases", 4, (1, 2, 3)))
    for name, npts, phases in variants:
        rr = reuse_score(
            lat_s,
            lat_t,
            phases=phases,
            n_points=npts,
            n_perm=rcfg.get("n_perm", 50),
            n_boot=rcfg.get("n_boot", 200) if name == "default" else 0,
            seed=seed,
        )
        summary.append(
            {
                **tag,
                "variant": name,
                "reuse": rr.score,
                "reuse_ci_low": rr.ci[0],
                "reuse_ci_high": rr.ci[1],
                "cka_minus_null": float(np.nanmean([v["cka_minus_null"] for v in rr.per_phase.values()])),
            }
        )
        if name == "default":
            for p, v in rr.per_phase.items():
                phase_rows.append({**tag, "phase": p, **v})
    return summary, phase_rows


def prepare_reuse_v2_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate (or find) every dataset experiment R needs; returns their paths.
    Called once before launching shards so parallel jobs never generate the same file."""
    d, rcfg = cfg["data"], cfg["reuse"]
    shift = rcfg["shift"]
    acfg = {**cfg, "data": {**d, "expert": rcfg.get("analysis_expert", {})}}
    an_seed = d.get("analysis_seed", 777)
    paths: dict[str, Any] = {
        "analysis_source": ensure_dataset(acfg, "analysis", "source", d["n_analysis"], an_seed, p_perturbed=0.0),
        "analysis_shifted": {
            s: ensure_dataset(acfg, "analysis", shift, d["n_analysis"], an_seed, severity=s, p_perturbed=0.0)
            for s in rcfg["severities"]
        },
        "finetune": {},
    }
    for ft in rcfg["finetune"]:
        fcfg = {**cfg, "data": {**d, "expert": ft.get("expert", {})}}
        for s in ft.get("severities", rcfg["severities"]):
            paths["finetune"][(ft["name"], s)] = ensure_dataset(
                fcfg,
                "finetune",
                ft.get("domain", shift),
                max(ft["budgets"]) + 8,
                d.get("finetune_seed", 555),
                severity=s,
            )
    return paths


def run_reuse_v2(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    _save_config(cfg, run_dir)
    d, ev, rcfg = cfg["data"], cfg["eval"], cfg["reuse"]
    shift = rcfg["shift"]
    ck_dir = Path(run_dir) / "checkpoints"
    paths = prepare_reuse_v2_data(cfg)
    src_an = load_dataset(paths["analysis_source"])
    tgt_an = {s: load_dataset(p) for s, p in paths["analysis_shifted"].items()}
    # --- reference representations (independent of the trained models)
    ref_rows, ref_phase = [], []
    ref_norm = Normalizer.fit(src_an.arrays["obs"])
    for kind in ("obs", "proprio", "progress"):
        ls = reference_latents(src_an, kind, ref_norm)
        for s, tgt in tgt_an.items():
            lt = reference_latents(tgt, kind, ref_norm)
            a, b = _reuse_rows(ls, lt, {"model": f"ref_{kind}", "seed": -1, "severity": s}, rcfg, 0)
            ref_rows += a
            ref_phase += b
    # --- per pretrained model
    points, reuse_rows, phase_rows, probe_rows, eps_all = [], [], [], [], []
    jobs = [(s, c) for s in cfg["seeds"] for c in cfg["conditions"]]
    probe_sevs = set(rcfg.get("probe_severities", rcfg["severities"]))
    stride = int(rcfg.get("probe_stride", 1))
    for seed, cond in _jobs(cfg, jobs):
        pre = _ckpt(cfg, cond, seed)
        model, mcfg, norm, ck = load_checkpoint(pre)
        cache = Path(run_dir) / "analysis" / f"{cond}_seed{seed}.json"
        if cache.exists():
            cached = load_json(cache)
        else:
            cached = {"reuse": [], "phase": [], "probe": []}
            ls = extract_latents(model, norm, src_an)
            keep_s = (ls.t % stride) == 0
            within = None
            if probe_sevs:
                within = grouped_cv_probe(ls.z[keep_s], ls.mode[keep_s], ls.episode[keep_s], seed=seed)
            # untrained network with the same architecture and seed (reference)
            torch.manual_seed(seed)
            layout = {k: slice(*v) for k, v in ck["layout"].items()}
            untrained, _ = build_model(layout, mcfg)
            lu = extract_latents(untrained.eval(), norm, src_an)
            for s, tgt in tgt_an.items():
                lt = extract_latents(model, norm, tgt)
                tag = {"model": cond, "seed": seed, "severity": s}
                a, b = _reuse_rows(ls, lt, tag, rcfg, seed)
                cached["reuse"] += a
                cached["phase"] += b
                a, b = _reuse_rows(
                    lu,
                    extract_latents(untrained.eval(), norm, tgt),
                    {"model": f"untrained_{cond}", "seed": seed, "severity": s},
                    rcfg,
                    seed,
                )
                cached["reuse"] += a
                if s in probe_sevs:
                    keep_t = (lt.t % stride) == 0
                    zs, ys, es = ls.z[keep_s], ls.mode[keep_s], ls.episode[keep_s]
                    zt, yt, et = lt.z[keep_t], lt.mode[keep_t], lt.episode[keep_t]
                    xp = cross_domain_probe(zs, ys, zt, yt, seed=seed)
                    hp = paired_heldout_probe(zs, ys, es, zt, yt, et, seed=seed)
                    cached["probe"].append(
                        {
                            **tag,
                            "within_source_cv": within.balanced_accuracy,
                            "cross_all_pairs": xp.balanced_accuracy,
                            "heldout_pairs_source": hp.source_bacc,
                            "heldout_pairs_shifted": hp.shifted_bacc,
                            "recall_source": hp.source_recall,
                            "recall_shifted": hp.shifted_recall,
                            "support_source": hp.support_source,
                            "support_shifted": hp.support_shifted,
                        }
                    )
            save_json(cached, cache)
        reuse_rows += cached["reuse"]
        phase_rows += cached["phase"]
        probe_rows += cached["probe"]
        # --- adaptation runs
        for ft in rcfg["finetune"]:
            if ft.get("conditions") and cond not in ft["conditions"]:
                continue
            for s in ft.get("severities", rcfg["severities"]):
                fcfg = {**cfg, "data": {**d, "expert": ft.get("expert", {})}}
                dom = ft.get("domain", shift)
                ft_path = ensure_dataset(
                    fcfg, "finetune", dom, max(ft["budgets"]) + 8, d.get("finetune_seed", 555), severity=s
                )
                base = f"{cond}_seed{seed}_{ft['name']}_sev{s:g}"
                ev_doms = tuple(ft.get("eval_domains", [dom]))
                lab = {"condition": cond, "seed": seed, "ft": ft["name"], "severity": s}
                zs = _eval_cached(
                    run_dir,
                    f"{base}_b0",
                    pre,
                    ev_doms,
                    ev["n_episodes"],
                    seed=ev.get("seed", 123),
                    severity=s,
                    labels={**lab, "budget": 0},
                )
                eps_all.append(zs)
                for budget in ft["budgets"]:
                    res = _train_cached(
                        [ft_path],
                        mcfg,
                        train_config(
                            cfg,
                            seed,
                            steps=ft.get("steps", 300),
                            max_episodes=budget,
                            val_fraction=0.0,
                            lr=ft.get("lr", 3e-4),
                            log_grad_norms=False,
                        ),
                        ck_dir,
                        init_checkpoint=pre,
                        tag=f"{base}_b{budget}",
                        verbose=False,
                    )
                    eps_all.append(
                        _eval_cached(
                            run_dir,
                            f"{base}_b{budget}",
                            res.checkpoint,
                            ev_doms,
                            ev["n_episodes"],
                            seed=ev.get("seed", 123),
                            severity=s,
                            labels={**lab, "budget": budget},
                        )
                    )
        print(f"  R {cond} seed {seed} done", flush=True)
    if cfg.get("_shard") is not None:
        return pd.DataFrame()
    eps = pd.concat(eps_all, ignore_index=True)
    eps.to_csv(Path(run_dir) / "episode_metrics.csv", index=False)
    for name, rows in (
        ("reuse_scores", reuse_rows + ref_rows),
        ("cka_by_phase", phase_rows + ref_phase),
        ("probes", probe_rows),
    ):
        pd.DataFrame(rows).to_csv(Path(run_dir) / f"{name}.csv", index=False)
    # adaptation curves and gain metrics
    g = (
        eps.groupby(["condition", "seed", "ft", "severity", "domain", "budget"])
        .agg(success=("success", "mean"), peak_force=("peak_force", "mean"))
        .reset_index()
    )
    g.to_csv(Path(run_dir) / "adaptation_curves.csv", index=False)
    thr = rcfg.get("threshold", 0.9)
    for (cond, seed, ftn, s, dom), c in g.groupby(["condition", "seed", "ft", "severity", "domain"]):
        c = c.sort_values("budget")
        b, y = c.budget.to_numpy(float), c.success.to_numpy(float)
        zs = y[0]
        gain = y - zs
        reached = b[y >= thr]
        points.append(
            {
                "model": cond,
                "seed": seed,
                "ft": ftn,
                "severity": s,
                "domain": dom,
                "zero_shot_success": zs,
                "final_success": y[-1],
                "max_gain": float(gain.max()),
                "final_gain": float(gain[-1]),
                "gain_auc": float(_trapz(gain, b) / (b.max() - b.min())) if b.max() > b.min() else np.nan,
                "demos_to_threshold": float(reached.min()) if len(reached) else np.inf,
                "threshold": thr,
            }
        )
    pd.DataFrame(points).to_csv(Path(run_dir) / "adaptation_points.csv", index=False)
    save_json({"experiment": cfg["experiment"], "shift": shift}, Path(run_dir) / "metrics.json")
    return eps
