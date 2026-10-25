"""Collect the outputs of experiments A-D into one set of tables and hypothesis checks.

Every number in the README results table and in the white paper is produced
here, from the CSV/JSON files the experiment runs wrote.

    python scripts/summarize_results.py --scale medium --out results/medium
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from contact_state_reuse.evaluation.metrics import paired_difference, seed_level
from contact_state_reuse.utils.io import save_json

EXPS = ("A_modality", "B_objective", "C_force_timing", "D_reuse")
DOMAINS = ("source", "shifted_friction", "tight_clearance", "sensor_noise", "occlusion")


def latest_run(root: Path, exp: str, scale: str) -> Path:
    runs = sorted(p for p in (root / f"{exp}_{scale}").glob("*") if (p / "episode_metrics.csv").exists())
    if not runs:
        raise FileNotFoundError(f"no finished run for {exp} at scale {scale} under {root}")
    return runs[-1]


def fmt(m: float, lo: float, hi: float, d: int = 2) -> str:
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def cell(summary: pd.DataFrame, cond: str, domain: str, metric: str, d: int = 2, corruption: str = "none") -> str:
    s = summary[(summary.condition == cond) & (summary.domain == domain) & (summary.metric == metric)]
    if "corruption" in s:
        s = s[s.corruption == corruption]
    if s.empty:
        return "n/a"
    r = s.iloc[0]
    return fmt(r["mean"], r.ci_low, r.ci_high, d)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="medium")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    root = Path(a.outputs)
    out = Path(a.out or f"results/{a.scale}")
    out.mkdir(parents=True, exist_ok=True)
    runs = {e: latest_run(root, e, a.scale) for e in EXPS}
    # keep a copy of every run (without checkpoints) next to the summary
    for e, r in runs.items():
        dst = out / "runs" / e
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(r, dst, ignore=shutil.ignore_patterns("checkpoints"))
    res: dict = {"scale": a.scale, "runs": {e: str(r) for e, r in runs.items()}}

    # ---------------- A and B: success / force by condition and domain
    epsA = pd.read_csv(runs["A_modality"] / "episode_metrics.csv")
    epsB = pd.read_csv(runs["B_objective"] / "episode_metrics.csv")
    sumA = pd.read_csv(runs["A_modality"] / "results.csv")
    sumB = pd.read_csv(runs["B_objective"] / "results.csv")
    eps_d = pd.read_csv(runs["D_reuse"] / "reuse_points.csv")

    rows = []
    for name, src, cond in (
        ("Reactive, extero + proprio", sumA, "reactive_EP"),
        ("Reactive, + force", sumA, "reactive_EPF"),
        ("Sequence BC, extero + proprio", sumA, "sequence_EP"),
        ("Sequence BC, + force", sumA, "sequence_EPF"),
        ("World model, no force (input or target)", sumA, "world_model_EP"),
        ("World model, next-obs prediction only", sumB, "wm_obs"),
        ("World model, self-supervised (obs + force)", sumA, "world_model_EPF"),
        ("World model, simulator-supervised aux", sumB, "wm_obs_force_contactsup"),
    ):
        rows.append(
            {
                "condition": name,
                "key": cond,
                "source_success": cell(src, cond, "source", "success"),
                "tight_success": cell(src, cond, "tight_clearance", "success"),
                "friction_success": cell(src, cond, "shifted_friction", "success"),
                "source_peak": cell(src, cond, "source", "peak_force", 1),
                "tight_peak": cell(src, cond, "tight_clearance", "peak_force", 1),
            }
        )
    dmap = {"sequence_EPF": "sequence_bc", "world_model_EPF": "wm_selfsup", "wm_obs_force_contactsup": "wm_contactsup"}
    for r in rows:
        dk = dmap.get(r["key"])
        sub = eps_d[(eps_d.model == dk) & (eps_d.severity == eps_d.severity.max())] if dk else pd.DataFrame()
        if len(sub):
            r["probe"] = f"{sub.probe_source_bacc.mean():.2f} +/- {sub.probe_source_bacc.std(ddof=1):.2f}"
            r["reuse"] = f"{sub.reuse.mean():.2f} +/- {sub.reuse.std(ddof=1):.2f}"
        else:
            r["probe"] = r["reuse"] = "n/a"
    main_table = pd.DataFrame(rows)
    main_table.to_csv(out / "main_table.csv", index=False)

    # full per-domain success tables
    pd.concat([sumA.assign(experiment="A"), sumB.assign(experiment="B")]).to_csv(out / "AB_summary.csv", index=False)
    seedA = seed_level(epsA, ["condition", "domain"], ["success", "peak_force", "force_violation", "jam_or_failure"])
    seedB = seed_level(epsB, ["condition", "domain"], ["success", "peak_force", "force_violation", "jam_or_failure"])
    pd.concat([seedA.assign(experiment="A"), seedB.assign(experiment="B")]).to_csv(
        out / "AB_seed_level.csv", index=False
    )

    # ---------------- H1: force vs no force, paired, per model kind and domain
    h1 = []
    for a_, b_ in (
        ("reactive_EPF", "reactive_EP"),
        ("sequence_EPF", "sequence_EP"),
        ("world_model_EPF", "world_model_EP"),
    ):
        for dom in DOMAINS:
            sub = epsA[epsA.domain == dom]
            for m in ("success", "peak_force", "force_violation"):
                h1.append({"domain": dom, **paired_difference(sub, "condition", a_, b_, m)})
    h1 = pd.DataFrame(h1)
    h1.to_csv(out / "H1_force_vs_noforce.csv", index=False)

    # ---------------- H2: probes and objectives
    h2_rows = []
    for m, g in eps_d.groupby("model"):
        for sev, gg in g.groupby("severity"):
            h2_rows.append(
                {
                    "model": m,
                    "severity": sev,
                    "probe_source_mean": gg.probe_source_bacc.mean(),
                    "probe_source_sd": gg.probe_source_bacc.std(ddof=1),
                    "probe_cross_mean": gg.probe_cross_domain_bacc.mean(),
                    "probe_cross_sd": gg.probe_cross_domain_bacc.std(ddof=1),
                    "reuse_mean": gg.reuse.mean(),
                    "reuse_sd": gg.reuse.std(ddof=1),
                    "zero_shot_success": gg.zero_shot_success.mean(),
                    "adaptation_auc": gg.adaptation_auc.mean(),
                    "final_success": gg.final_success.mean(),
                    "peak_force_change": gg.peak_force_change.mean(),
                    "n_seeds": len(gg),
                }
            )
    h23 = pd.DataFrame(h2_rows)
    h23.to_csv(out / "H2_H3_probe_reuse.csv", index=False)
    auroc = epsB.groupby(["condition", "domain"]).first_contact_auroc.agg(["mean", "std"]).reset_index()
    auroc.to_csv(out / "first_contact_auroc.csv", index=False)
    h2b = []
    for a_, b_ in (("wm_obs", "sequence_bc"), ("wm_obs_force", "wm_obs"), ("wm_obs_force_contactsup", "wm_obs_force")):
        for dom in DOMAINS:
            for m in ("success", "peak_force"):
                h2b.append({"domain": dom, **paired_difference(epsB[epsB.domain == dom], "condition", a_, b_, m)})
    pd.DataFrame(h2b).to_csv(out / "B_objective_paired.csv", index=False)

    # ---------------- H4
    res["H4"] = json.loads((runs["D_reuse"] / "reuse_correlations.json").read_text())

    # ---------------- H5
    epsC = pd.read_csv(runs["C_force_timing"] / "episode_metrics.csv")
    h5 = []
    for cond in epsC.condition.unique():
        for dom in epsC.domain.unique():
            sub = epsC[(epsC.condition == cond) & (epsC.domain == dom)]
            for m in ("success", "peak_force", "force_violation", "jam_or_failure"):
                for b_ in ("random_window", "none"):
                    h5.append(
                        {
                            "condition": cond,
                            "domain": dom,
                            **paired_difference(sub, "corruption", "first_contact_window", b_, m),
                        }
                    )
                h5.append({"condition": cond, "domain": dom, **paired_difference(sub, "corruption", "zero", "none", m)})
    h5 = pd.DataFrame(h5)
    h5.to_csv(out / "H5_force_timing.csv", index=False)
    overlap = epsC[epsC.corruption == "random_window"].window_overlapped_first_contact
    res["C_random_window_overlap_rate"] = float(pd.to_numeric(overlap, errors="coerce").mean())

    # ---------------- markdown
    lines = [
        f"# Results summary ({a.scale} scale)",
        "",
        "Generated by `scripts/summarize_results.py`. Means with 95 % bootstrap intervals over all evaluation "
        "episodes pooled across 3 seeds; probe and reuse are mean +/- s.d. over seeds (strongest shift).",
        "",
        "| Condition | Source success | Tight-clearance success | Friction-shift success | Peak force source [N] | "
        "Peak force tight [N] | Contact-mode probe | Reuse score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in main_table.iterrows():
        lines.append(
            f"| {r.condition} | {r.source_success} | {r.tight_success} | {r.friction_success} | "
            f"{r.source_peak} | {r.tight_peak} | {r.probe} | {r.reuse} |"
        )
    lines += [
        "",
        "## H1: with minus without force (paired)",
        "",
        "| model | domain | metric | diff | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for _, r in h1.iterrows():
        star = " *" if (r.ci_low > 0 or r.ci_high < 0) else ""
        lines.append(
            f"| {r.a.split('_')[0]} | {r.domain} | {r.metric} | {r['diff']:+.3f}{star} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |"
        )
    lines += [
        "",
        "## H5: force zeroed at first contact vs other conditions (paired)",
        "",
        "| model | domain | comparison | metric | diff | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in h5.iterrows():
        star = " *" if (r.ci_low > 0 or r.ci_high < 0) else ""
        lines.append(
            f"| {r.condition} | {r.domain} | {r.a} - {r.b} | {r.metric} | {r['diff']:+.3f}{star} | "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] |"
        )
    lines += [
        "",
        "## H2/H3: probes and reuse (experiment D)",
        "",
        h23.round(3).to_markdown(index=False),
        "",
        "## H4: correlations",
        "",
        "```",
        json.dumps(res["H4"], indent=1),
        "```",
        "",
        "`*` = interval excludes 0.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    save_json(res, out / "summary.json")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
