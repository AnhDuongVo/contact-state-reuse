"""Tables and key numbers for the v0.2 experiments (E, T, L, G1, G2, R).

All intervals are crossed seed x episode bootstrap intervals (training runs
and evaluation episodes resampled independently; pairing of conditions on the
same seed and episode is kept). Every number in the revised white paper comes
from the files this script writes.

    python scripts/summarize_revision.py --out results/revision
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from contact_state_reuse.analysis.stats import spearman_with_ci
from contact_state_reuse.evaluation.metrics import paired_crossed, seed_episode_summary

DOMAINS = ("source", "shifted_friction", "tight_clearance", "sensor_noise", "occlusion")
METRICS = ("success", "peak_force", "peak_force_sum", "force_violation", "jam_or_failure")
CONTRAST_METRICS = ("success", "peak_force", "peak_force_sum", "force_violation")

E_CONTRASTS = {
    "factorial": [
        ("force input | BC", "bc_F", "bc"),
        ("force input | obs pred", "obs_F", "obs"),
        ("force input | force target", "F_ftgt", "ftgt"),
        ("force input | obs pred + force target", "obs_F_ftgt", "obs_ftgt"),
        ("force target | no force input", "ftgt", "bc"),
        ("force target | force input", "F_ftgt", "bc_F"),
        ("force target | obs pred, no force input", "obs_ftgt", "obs"),
        ("force target | obs pred, force input", "obs_F_ftgt", "obs_F"),
        ("obs pred | no force", "obs", "bc"),
        ("obs pred | force input", "obs_F", "bc_F"),
        ("obs pred | force target, no input", "obs_ftgt", "ftgt"),
        ("obs pred | force input + target", "obs_F_ftgt", "F_ftgt"),
        ("full vs BC with force", "obs_F_ftgt", "bc_F"),
        ("full vs BC without force", "obs_F_ftgt", "bc"),
    ],
    "failure_data": [
        ("all episodes | BC", "bc_F_alldata", "bc_F"),
        ("all episodes | obs pred", "obs_F_alldata", "obs_F"),
        ("all episodes | obs pred + force target", "obs_F_ftgt_alldata", "obs_F_ftgt"),
    ],
    "aux_controls": [
        ("reconstruction vs BC", "recon_F", "bc_F"),
        ("prediction vs reconstruction", "obs_F_ftgt", "recon_F"),
        ("stop-gradient vs BC", "obs_F_ftgt_stopgrad", "bc_F"),
        ("prediction vs stop-gradient", "obs_F_ftgt", "obs_F_ftgt_stopgrad"),
        ("separate predictive core vs BC", "obs_F_ftgt_sepcore", "bc_F"),
        ("shared vs separate predictive core", "obs_F_ftgt", "obs_F_ftgt_sepcore"),
        ("weight 0.1 vs 1", "obs_F_ftgt_w0.1", "obs_F_ftgt"),
        ("weight 0.3 vs 1", "obs_F_ftgt_w0.3", "obs_F_ftgt"),
        ("weight 3 vs 1", "obs_F_ftgt_w3", "obs_F_ftgt"),
        ("weight 0.1 vs BC", "obs_F_ftgt_w0.1", "bc_F"),
    ],
    "fault_aware": [
        ("fault-aware vs standard | full", "obs_F_ftgt_aug", "obs_F_ftgt"),
        ("fault-aware vs standard | BC", "bc_F_aug", "bc_F"),
        ("fault-aware full vs fault-aware BC", "obs_F_ftgt_aug", "bc_F_aug"),
        ("fault-aware full vs BC without force", "obs_F_ftgt_aug", "bc"),
    ],
    "bc_diagnostics": [
        ("lr 3e-4 vs 1e-3 | BC force", "bc_F_lr3e-4", "bc_F"),
        ("lr 1e-4 vs 1e-3 | BC force", "bc_F_lr1e-4", "bc_F"),
        ("lr 3e-4 vs 1e-3 | BC", "bc_lr3e-4", "bc"),
        ("lr 3e-4 vs 1e-3 | full", "obs_F_ftgt_lr3e-4", "obs_F_ftgt"),
        ("force input | BC, lr 3e-4", "bc_F_lr3e-4", "bc_lr3e-4"),
        ("force input | obs + force targets, lr 3e-4", "obs_F_ftgt_lr3e-4", "obs_ftgt_lr3e-4"),
        ("full vs BC force | both lr 3e-4", "obs_F_ftgt_lr3e-4", "bc_F_lr3e-4"),
        ("full vs BC without force | both lr 3e-4", "obs_F_ftgt_lr3e-4", "bc_lr3e-4"),
        ("full (lr 1e-3) vs tuned BC force (lr 3e-4)", "obs_F_ftgt", "bc_F_lr3e-4"),
        ("3000 vs 1500 updates | BC force", "bc_F_long", "bc_F"),
        ("no previous action | BC force", "bc_F_noprev", "bc_F"),
        ("no previous action | BC", "bc_noprev", "bc"),
        ("force input | BC without previous action", "bc_F_noprev", "bc_noprev"),
        ("no previous action | full", "obs_F_ftgt_noprev", "obs_F_ftgt"),
        ("full vs BC | both without previous action", "obs_F_ftgt_noprev", "bc_F_noprev"),
        ("log force vs raw | BC", "bc_F_log", "bc_F"),
        ("contact flag vs raw | BC", "bc_F_binary", "bc_F"),
        ("log force vs no force | BC", "bc_F_log", "bc"),
        ("contact flag vs no force | BC", "bc_F_binary", "bc"),
        ("no DART noise | BC force", "bc_F_dart0", "bc_F"),
        ("no DART noise | BC", "bc_dart0", "bc"),
        ("force input | BC without DART noise", "bc_F_dart0", "bc_dart0"),
        ("2x capacity | BC force", "bc_F_wide", "bc_F"),
        ("2x capacity BC force vs BC", "bc_F_wide", "bc"),
        ("full vs 2x capacity BC force", "obs_F_ftgt", "bc_F_wide"),
    ],
}
G_CONTRASTS = [
    ("force input | BC", "bc_F", "bc"),
    ("full vs BC with force", "obs_F_ftgt", "bc_F"),
    ("full vs obs pred without force", "obs_F_ftgt", "obs"),
    ("obs pred | no force", "obs", "bc"),
    ("force target, no force input vs BC", "obs_ftgt", "bc"),
    ("force input | obs + force targets", "obs_F_ftgt", "obs_ftgt"),
    ("force input | BC, lr 3e-4", "bc_F_lr3e-4", "bc_lr3e-4"),
    ("full vs BC with force | both lr 3e-4", "obs_F_ftgt_lr3e-4", "bc_F_lr3e-4"),
    ("full (lr 1e-3) vs BC with force (lr 3e-4)", "obs_F_ftgt", "bc_F_lr3e-4"),
]


def _run(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if (p / "episode_metrics.csv").exists() or (p / "adaptation_points.csv").exists() else None


def cond_table(eps: pd.DataFrame, domains: tuple[str, ...], metrics=METRICS, group="condition") -> pd.DataFrame:
    rows = []
    for c, g in eps.groupby(group):
        for dom in (*domains, "pooled"):
            sub = g if dom == "pooled" else g[g.domain == dom]
            if sub.empty:
                continue
            cols = ("domain", "episode") if dom == "pooled" else ("episode",)
            for m in metrics:
                if m not in sub or sub[m].isna().all():
                    continue
                rows.append({group: c, "domain": dom, **seed_episode_summary(sub, m, episode_cols=cols)})
    return pd.DataFrame(rows)


def contrasts(eps: pd.DataFrame, items, domains, family: str, col="condition", metrics=CONTRAST_METRICS):
    rows = []
    have = set(eps[col].unique())
    for name, a, b in items:
        if a not in have or b not in have:
            continue
        for dom in (*domains, "pooled"):
            sub = eps if dom == "pooled" else eps[eps.domain == dom]
            cols = ("domain", "episode") if dom == "pooled" else ("episode",)
            for m in metrics:
                if m in sub and not sub[m].isna().all():
                    rows.append(
                        {
                            "family": family,
                            "contrast": name,
                            "domain": dom,
                            **paired_crossed(sub, col, a, b, m, episode_cols=cols),
                        }
                    )
    return rows


def fmt(r, d=2) -> str:
    return f"{r['mean']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"


def fmt_diff(r, d=2) -> str:
    star = " *" if (r["ci_low"] > 0 or r["ci_high"] < 0) else ""
    return f"{r['diff']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]{star}"


# --------------------------------------------------------------------- E
def summarize_E(run: Path, out: Path, key: dict) -> list[str]:
    eps = pd.read_csv(run / "episode_metrics.csv")
    base = eps[eps.corruption == "none"]
    tab = cond_table(base, DOMAINS)
    tab.to_csv(out / "E_conditions.csv", index=False)
    rows = []
    for fam, items in E_CONTRASTS.items():
        rows += contrasts(base, items, DOMAINS, fam)
    con = pd.DataFrame(rows)
    con.to_csv(out / "E_contrasts.csv", index=False)
    # fault-aware: extra sensor corruptions on the source domain
    extra = eps[eps.corruption != "none"]
    fault_rows = []
    if len(extra):
        for cor, g in extra.groupby("corruption"):
            for name, a, b in E_CONTRASTS["fault_aware"][:3]:
                for m in ("success", "peak_force"):
                    if {a, b} <= set(g.condition):
                        fault_rows.append(
                            {"corruption": cor, "contrast": name, **paired_crossed(g, "condition", a, b, m)}
                        )
        ext = []
        for (c, cor), g in extra.groupby(["condition", "corruption"]):
            for m in ("success", "peak_force"):
                ext.append({"condition": c, "corruption": cor, **seed_episode_summary(g, m)})
        pd.DataFrame(ext).to_csv(out / "E_fault_extra_conditions.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(out / "E_fault_extra_contrasts.csv", index=False)
    # behaviour metrics (policy-chosen compliance, approach speed, timing)
    beh = []
    for (c, dom), g in base.groupby(["condition", "domain"]):
        for m in (
            "compliance_pre_contact",
            "compliance_in_contact",
            "approach_speed",
            "first_contact_step",
            "steps",
            "time_to_success",
            "impulse",
        ):
            if m in g and not g[m].isna().all():
                beh.append({"condition": c, "domain": dom, **seed_episode_summary(g, m)})
    pd.DataFrame(beh).to_csv(out / "E_behaviour.csv", index=False)
    # first-contact anticipation (pre-contact steps only) vs simple baselines
    au = (
        base.groupby(["condition", "domain", "seed"])[["first_contact_auroc", "auroc_current_force", "auroc_distance"]]
        .first()
        .reset_index()
    )
    au = au.dropna(subset=["first_contact_auroc"])
    au.groupby(["condition", "domain"])[["first_contact_auroc", "auroc_current_force", "auroc_distance"]].agg(
        ["mean", "std"]
    ).round(3).to_csv(out / "E_auroc.csv")
    # training: final losses and gradient norms per term (shared trunk)
    hist = json.loads((run / "training_histories.json").read_text())
    tr = []
    for tag, h in hist.items():
        cond, seed = tag.rsplit("_seed", 1)
        last = h[-1]
        g_mean = pd.DataFrame(h).filter(like="grad_").mean()
        tr.append(
            {
                "condition": cond,
                "seed": int(seed),
                "val_action": last.get("val_action"),
                "train_action": last.get("train_action"),
                "val_obs": last.get("val_obs"),
                "val_force": last.get("val_force"),
                **{f"{k}_mean": float(v) for k, v in g_mean.items()},
                **{f"{k}_final": last.get(k) for k in g_mean.index},
            }
        )
    trd = pd.DataFrame(tr).sort_values(["condition", "seed"])
    trd.to_csv(out / "E_training.csv", index=False)

    # key numbers
    def cell(c, dom, m):
        r = tab[(tab.condition == c) & (tab.domain == dom) & (tab.metric == m)]
        return None if r.empty else r.iloc[0].to_dict()

    key["E"] = {
        "cells": {
            c: {
                dom: {m: cell(c, dom, m) for m in ("success", "peak_force", "peak_force_sum")}
                for dom in (*DOMAINS, "pooled")
            }
            for c in base.condition.unique()
        },
        "contrasts": con.to_dict("records"),
    }
    # markdown
    lines = [
        "## E: factorial (fixed trunk, successful demonstrations only, 5 seeds)",
        "",
        "Pooled over the five domains; crossed seed x episode 95 % bootstrap intervals; per-seed means.",
        "",
        "| condition | success | per-seed success | peak force [N] | summed-contact peak [N] |",
        "|---|---|---|---|---|",
    ]
    for c in E_order(base.condition.unique()):
        s, f, fs = cell(c, "pooled", "success"), cell(c, "pooled", "peak_force"), cell(c, "pooled", "peak_force_sum")
        if s:
            lines.append(
                f"| {c} | {fmt(s)} | {', '.join(f'{v:.2f}' for v in s['per_seed'])} | {fmt(f, 1)} | "
                f"{fmt(fs, 1) if fs else 'n/a'} |"
            )
    lines += [
        "",
        "### Contrasts (pooled over domains)",
        "",
        "| family | contrast | success diff | peak force diff [N] | seeds favouring a (success) |",
        "|---|---|---|---|---|",
    ]
    pooled = con[con.domain == "pooled"]
    for (fam, name), g in pooled.groupby(["family", "contrast"], sort=False):
        s, f = g[g.metric == "success"].iloc[0], g[g.metric == "peak_force"].iloc[0]
        lines.append(f"| {fam} | {name} | {fmt_diff(s)} | {fmt_diff(f, 1)} | {s.seeds_a_better}/{s.n_seeds} |")
    return lines


def E_order(conds) -> list[str]:
    order = [c for fam in E_CONTRASTS.values() for _, a, b in fam for c in (b, a)]
    seen, out = set(), []
    for c in order + sorted(conds):
        if c in conds and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------- T
def summarize_T(run: Path, out: Path, key: dict) -> list[str]:
    eps = pd.read_csv(run / "episode_metrics.csv")
    eps["compliance_override"] = eps.compliance_override.astype(str)
    pol = eps[eps.compliance_override == "policy"].copy()
    pol["window_reached"] = (pol.masked_steps > 0).astype(float)
    tab = []
    for (c, dom, cor), g in pol.groupby(["condition", "domain", "corruption"]):
        for m in ("success", "peak_force", "force_violation", "masked_steps", "masked_force", "window_reached"):
            tab.append({"condition": c, "domain": dom, "corruption": cor, **seed_episode_summary(g, m)})
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "T_results.csv", index=False)
    pairs = [(c, "none") for c in pol.corruption.unique() if c != "none"]
    pairs += [("first_contact_window", c) for c in ("onset+6", "onset+12", "onset+24", "random_window")]
    pairs += [("hold_onset", "first_contact_window"), ("onset_only", "zero")]
    rows = []
    for c, g in pol.groupby("condition"):
        for dom in ("source", "tight_clearance"):
            sub = g[g.domain == dom]
            for a, b in pairs:
                for m in ("success", "peak_force", "force_violation"):
                    if {a, b} <= set(sub.corruption):
                        rows.append({"condition": c, "domain": dom, **paired_crossed(sub, "corruption", a, b, m)})
    con = pd.DataFrame(rows)
    con.to_csv(out / "T_contrasts.csv", index=False)
    comp = eps[eps.corruption == "none"]
    crow = []
    for (c, dom), g in comp.groupby(["condition", "domain"]):
        for ov in g.compliance_override.unique():
            gg = g[g.compliance_override == ov]
            for m in ("success", "peak_force", "compliance_in_contact"):
                crow.append({"condition": c, "domain": dom, "compliance": ov, **seed_episode_summary(gg, m)})
    pd.DataFrame(crow).to_csv(out / "T_compliance.csv", index=False)
    ccon = []
    for ov in [o for o in comp.compliance_override.unique() if o != "policy"]:
        for dom in ("source", "tight_clearance"):
            sub = comp[(comp.domain == dom) & (comp.compliance_override == ov)]
            for a, b in (("obs_F_ftgt", "bc_F"), ("obs_F_ftgt", "bc"), ("obs_F_ftgt", "obs_F")):
                for m in ("success", "peak_force"):
                    if {a, b} <= set(sub.condition):
                        ccon.append({"compliance": ov, "domain": dom, **paired_crossed(sub, "condition", a, b, m)})
    pd.DataFrame(ccon).to_csv(out / "T_compliance_contrasts.csv", index=False)
    key["T"] = {"contrasts": con.to_dict("records"), "compliance_contrasts": ccon}
    lines = [
        "## T: when is force used (evaluation-time interventions, 5 seeds)",
        "",
        "| model | domain | intervention | success | peak force [N] | masked steps | masked force [N] | "
        "window reached |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in tab[tab.metric == "success"].iterrows():
        g = tab[(tab.condition == r.condition) & (tab.domain == r.domain) & (tab.corruption == r.corruption)]

        def v(m, d=2, g=g):
            x = g[g.metric == m]
            return fmt(x.iloc[0], d) if len(x) else "n/a"

        lines.append(
            f"| {r.condition} | {r.domain} | {r.corruption} | {v('success')} | {v('peak_force', 1)} | "
            f"{g[g.metric == 'masked_steps'].iloc[0]['mean']:.1f} | "
            f"{g[g.metric == 'masked_force'].iloc[0]['mean']:.1f} | "
            f"{g[g.metric == 'window_reached'].iloc[0]['mean']:.2f} |"
        )
    return lines


# --------------------------------------------------------------------- L
def summarize_L(run: Path, out: Path, key: dict) -> list[str]:
    eps = pd.read_csv(run / "episode_metrics.csv")
    # average the random draws per (seed, episode) so every ablation has one value per cell
    eps = (
        eps.groupby(["condition", "seed", "domain", "ablation", "episode"])[["success", "peak_force", "jam_or_failure"]]
        .mean()
        .reset_index()
    )
    tab, rows = [], []
    for (c, dom, ab), g in eps.groupby(["condition", "domain", "ablation"]):
        for m in ("success", "peak_force"):
            tab.append({"condition": c, "domain": dom, "ablation": ab, **seed_episode_summary(g, m)})
    for (c, dom), g in eps.groupby(["condition", "domain"]):
        for a, b in (
            ("contact", "none"),
            ("movement", "none"),
            ("random", "none"),
            ("contact", "random"),
            ("contact", "movement"),
        ):
            for m in ("success", "peak_force"):
                rows.append({"condition": c, "domain": dom, **paired_crossed(g, "ablation", a, b, m)})
    pd.DataFrame(tab).to_csv(out / "L_results.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "L_contrasts.csv", index=False)
    key["L"] = {"contrasts": rows}
    lines = [
        "## L: latent ablation (25 % of dimensions clamped, 5 seeds)",
        "",
        "| model | domain | contrast | success diff | peak force diff [N] |",
        "|---|---|---|---|---|",
    ]
    rr = pd.DataFrame(rows)
    for (c, dom, a, b), g in rr.groupby(["condition", "domain", "a", "b"], sort=False):
        s, f = g[g.metric == "success"].iloc[0], g[g.metric == "peak_force"].iloc[0]
        lines.append(f"| {c} | {dom} | {a} - {b} | {fmt_diff(s)} | {fmt_diff(f, 1)} |")
    return lines


# --------------------------------------------------------------------- G
def summarize_G(runs: dict[str, Path], out: Path, key: dict, scripted: Path | None) -> list[str]:
    lines = []
    key["G"] = {}
    for name, run in runs.items():
        eps = pd.read_csv(run / "episode_metrics.csv")
        eps = eps[eps.corruption == "none"]
        doms = tuple(eps.domain.unique())
        tab = cond_table(eps, doms, ("success", "peak_force", "jam_or_failure"))
        tab.to_csv(out / f"{name}_conditions.csv", index=False)
        con = pd.DataFrame(contrasts(eps, G_CONTRASTS, doms, name, metrics=("success", "peak_force")))
        con.to_csv(out / f"{name}_contrasts.csv", index=False)
        key["G"][name] = {"conditions": tab.to_dict("records"), "contrasts": con.to_dict("records")}
        lines += [
            f"## {name}",
            "",
            "| condition | domain | success | per-seed | peak force [N] |",
            "|---|---|---|---|---|",
        ]
        for _, r in tab[tab.metric == "success"].iterrows():
            f = tab[(tab.condition == r.condition) & (tab.domain == r.domain) & (tab.metric == "peak_force")].iloc[0]
            lines.append(
                f"| {r.condition} | {r.domain} | {fmt(r)} | {', '.join(f'{v:.2f}' for v in r.per_seed)} | {fmt(f, 1)} |"
            )
        lines += ["", "| contrast | domain | success diff | peak force diff [N] |", "|---|---|---|---|"]
        for (cn, dom), g in con.groupby(["contrast", "domain"], sort=False):
            s, f = g[g.metric == "success"].iloc[0], g[g.metric == "peak_force"].iloc[0]
            lines.append(f"| {cn} | {dom} | {fmt_diff(s)} | {fmt_diff(f, 1)} |")
        lines.append("")
    if scripted is not None and scripted.exists():
        sc = pd.read_csv(scripted)
        key["G"]["scripted"] = sc.to_dict("records")
        lines += ["## Scripted controllers (100 episodes per domain)", "", sc.round(3).to_markdown(index=False), ""]
    return lines


# --------------------------------------------------------------------- R
def summarize_R(run: Path, out: Path, key: dict) -> list[str]:
    pts = pd.read_csv(run / "adaptation_points.csv")
    pts.to_csv(out / "R_adaptation_points.csv", index=False)
    curves = pd.read_csv(run / "adaptation_curves.csv")
    curves.to_csv(out / "R_adaptation_curves.csv", index=False)
    agg = (
        pts.groupby(["model", "ft", "severity", "domain"])
        .agg(
            zero_shot=("zero_shot_success", "mean"),
            zero_shot_sd=("zero_shot_success", "std"),
            final=("final_success", "mean"),
            final_gain=("final_gain", "mean"),
            final_gain_sd=("final_gain", "std"),
            gain_auc=("gain_auc", "mean"),
            gain_auc_sd=("gain_auc", "std"),
            demos_to_thr_median=("demos_to_threshold", "median"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out / "R_adaptation_summary.csv", index=False)
    reuse = pd.read_csv(run / "reuse_scores.csv")
    reuse.to_csv(out / "R_reuse_scores.csv", index=False)
    ragg = (
        reuse.groupby(["model", "severity", "variant"])
        .agg(
            reuse=("reuse", "mean"),
            reuse_sd=("reuse", "std"),
            cka_minus_null=("cka_minus_null", "mean"),
            n=("reuse", "count"),
        )
        .reset_index()
    )
    ragg.to_csv(out / "R_reuse_summary.csv", index=False)
    cka = pd.read_csv(run / "cka_by_phase.csv")
    cka.to_csv(out / "R_cka_by_phase.csv", index=False)
    probes = pd.read_csv(run / "probes.csv")
    probes.to_csv(out / "R_probes.csv", index=False)
    pagg = (
        probes.groupby(["model", "severity"])[
            ["within_source_cv", "cross_all_pairs", "heldout_pairs_source", "heldout_pairs_shifted"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    pagg.to_csv(out / "R_probes_summary.csv")
    # correlations at the level of trained models (n = 15), and per severity
    s_pts = pts[(pts.ft == "search")]
    d_reuse = reuse[(reuse.variant == "default") & reuse.model.isin(s_pts.model.unique())]
    m = (
        d_reuse.groupby(["model", "seed"])
        .reuse.mean()
        .rename("reuse")
        .reset_index()
        .merge(s_pts.groupby(["model", "seed"]).gain_auc.mean().rename("gain_auc").reset_index(), on=["model", "seed"])
    )
    m = m.merge(
        s_pts.groupby(["model", "seed"]).zero_shot_success.mean().rename("zero_shot").reset_index(),
        on=["model", "seed"],
    )
    corr = {
        "model_level_reuse_vs_gain_auc": spearman_with_ci(m.reuse, m.gain_auc),
        "model_level_reuse_vs_zero_shot": spearman_with_ci(m.reuse, m.zero_shot),
        "per_severity": {},
        "within_family": {},
    }
    for sev, g in s_pts.groupby("severity"):
        mm = g.merge(d_reuse[d_reuse.severity == sev][["model", "seed", "reuse"]], on=["model", "seed"])
        corr["per_severity"][str(sev)] = spearman_with_ci(mm.reuse, mm.gain_auc)
    for fam, g in m.groupby("model"):
        corr["within_family"][fam] = spearman_with_ci(g.reuse, g.gain_auc)
    m.to_csv(out / "R_model_level_points.csv", index=False)
    (out / "R_correlations.json").write_text(json.dumps(corr, indent=1))
    key["R"] = {
        "adaptation": agg.to_dict("records"),
        "reuse": ragg.to_dict("records"),
        "correlations": corr,
        "probes": probes.groupby(["model", "severity"])[
            ["within_source_cv", "cross_all_pairs", "heldout_pairs_source", "heldout_pairs_shifted"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
    }
    lines = [
        "## R: adaptation (gain over zero-shot) and representation stability (exploratory)",
        "",
        agg.round(3).to_markdown(index=False),
        "",
        ragg[ragg.variant == "default"].round(3).to_markdown(index=False),
        "",
        "```",
        json.dumps(corr, indent=1),
        "```",
        "",
    ]
    return lines


# ---------------------------------------------------------- v1 follow-up
def v1_reuse_by_severity(v1_dir: Path, out: Path, key: dict) -> list[str]:
    """Why was v1 reuse highest at intermediate severity? Per-phase values and support."""
    p = v1_dir / "runs" / "D_reuse" / "cka_by_phase.csv"
    if not p.exists():
        return []
    cka = pd.read_csv(p)
    t = (
        cka.groupby(["model", "severity", "phase"])
        .agg(
            cka=("cka", "mean"),
            null=("null_mean", "mean"),
            reuse=("reuse", "mean"),
            reuse_sd=("reuse", "std"),
            n_episodes=("n_episodes", "mean"),
        )
        .reset_index()
    )
    t.to_csv(out / "v1_reuse_by_severity_phase.csv", index=False)
    key["v1_reuse_by_severity"] = t.to_dict("records")
    return ["## v1 follow-up: reuse by severity and phase", "", t.round(3).to_markdown(index=False), ""]


def v1_reference_reuse(data_dir: Path, out: Path) -> None:
    """Reuse score of raw observations and proprioception on the v0.1 tight-fit analysis sets."""
    from contact_state_reuse.analysis.latents import reference_latents
    from contact_state_reuse.analysis.reuse import reuse_score
    from contact_state_reuse.data.dataset import Normalizer, load_dataset

    src_p = data_dir / "analysis_source_sev1_n60_s777_p0_d0.1.npz"
    if not src_p.exists():
        return
    src = load_dataset(src_p)
    norm = Normalizer.fit(src.arrays["obs"])
    rows = []
    for kind in ("obs", "proprio"):
        ls = reference_latents(src, kind, norm)
        for sev in ("0.33", "0.67", "1"):
            p = data_dir / f"analysis_tight_clearance_sev{sev}_n60_s777_p0_d0.1.npz"
            if not p.exists():
                continue
            r = reuse_score(ls, reference_latents(load_dataset(p), kind, norm), n_perm=50, n_boot=0, seed=0)
            for ph, v in r.per_phase.items():
                rows.append({"reference": kind, "severity": float(sev), "phase": ph, **v})
            rows.append({"reference": kind, "severity": float(sev), "phase": "mean", "reuse": r.score})
    pd.DataFrame(rows).to_csv(out / "v1_reference_reuse.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="medium")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--out", default="results/revision")
    p.add_argument("--v1", default="results/medium")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {
        e: _run(f"{a.outputs}/{e}_{a.scale}/run1")
        for e in ("E_factorial", "T_timing", "L_ablation", "G_nonpriv", "G_search", "R_reuse")
    }
    key: dict = {"scale": a.scale, "runs": {k: str(v) for k, v in runs.items()}}
    lines = [
        f"# Revision results ({a.scale} scale)",
        "",
        "Generated by `scripts/summarize_revision.py`. Intervals: crossed bootstrap over training seeds and "
        "evaluation episodes (95 %). `*` = interval excludes 0.",
        "",
    ]
    if runs["E_factorial"]:
        lines += summarize_E(runs["E_factorial"], out, key) + [""]
    if runs["T_timing"]:
        lines += summarize_T(runs["T_timing"], out, key) + [""]
    if runs["L_ablation"]:
        lines += summarize_L(runs["L_ablation"], out, key) + [""]
    g = {k: v for k, v in runs.items() if k.startswith("G_") and v}
    if g:
        lines += summarize_G(g, out, key, out / "scripted_baselines.csv")
    if runs["R_reuse"]:
        lines += summarize_R(runs["R_reuse"], out, key)
    lines += v1_reuse_by_severity(Path(a.v1), out, key)
    v1_reference_reuse(Path("data/experiments"), out)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "key_numbers.json").write_text(json.dumps(key, indent=1, default=float))
    # keep a copy of the run outputs without checkpoints
    for name, run in runs.items():
        if run is not None:
            dst = out / "runs" / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(run, dst, ignore=shutil.ignore_patterns("checkpoints", "*.pt", "evals"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
