"""Figures for the white paper, drawn from the summarised results.

MUJOCO_GL=osmesa python scripts/make_paper_figures.py --results results/medium --out docs/whitepaper/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contact_state_reuse.analysis.plotting import INK2, MARKERS, MODE_COLORS, PALETTE, SURFACE  # noqa: E402
from contact_state_reuse.envs import MODE_NAMES, ContactInsertionEnv  # noqa: E402
from contact_state_reuse.envs.expert import ScriptedExpert  # noqa: E402
from contact_state_reuse.evaluation.metrics import bootstrap_ci  # noqa: E402

DOM_LABEL = {
    "source": "source",
    "shifted_friction": "friction",
    "tight_clearance": "tight\nclearance",
    "sensor_noise": "sensor\nnoise",
    "occlusion": "occlusion",
}
DOMS = list(DOM_LABEL)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("wrote", path)


def env_figure(out: Path) -> None:
    """Rendered snapshots of one expert episode plus its force trace with phase labels."""
    env = ContactInsertionEnv(render_mode="rgb_array")
    env.reset(seed=11)
    ex = ScriptedExpert()
    ex.reset(env, np.random.default_rng(3), "nominal")
    r = mujoco.Renderer(env.model, 300, 300)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ent = env.entrance()
    cam.lookat[:] = [ent[0], 0, ent[1] + 0.012]
    cam.distance, cam.azimuth, cam.elevation = 0.22, 90, -10
    frames, modes, forces, depth = [], [], [], []
    done, info = False, {"contact_mode": 0}
    while not done:
        r.update_scene(env.data, camera=cam)
        frames.append((r.render().copy(), info["contact_mode"]))
        _, _, term, trunc, info = env.step(ex.act(env))
        modes.append(info["contact_mode"])
        forces.append(info["peak_force"])
        depth.append(info["insertion_depth"])
        done = term or trunc
    r.update_scene(env.data, camera=cam)
    frames.append((r.render().copy(), info["contact_mode"]))
    picks = []
    for m in (0, 1, 3, 4):
        idx = [i for i, (_, mm) in enumerate(frames) if mm == m]
        if idx:
            picks.append(idx[len(idx) // 2] if m in (0, 3) else idx[0])
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, len(picks), height_ratios=[1.25, 1])
    for j, i in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frames[i][0][30:270, 30:270])
        ax.set_axis_off()
        ax.set_title(f"t = {i * 0.05:.2f} s, {MODE_NAMES[frames[i][1]].replace('_', ' ')}", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(1, len(forces) + 1) * 0.05
    for k in range(len(t)):
        ax.axvspan(t[k] - 0.05, t[k], color=MODE_COLORS[modes[k]], alpha=0.3, lw=0)
    ax.plot(t, forces, color="#0b0b0b", lw=1.6, label="contact force [N]")
    ax2 = ax.twinx()
    ax2.plot(t, 1e3 * np.asarray(depth), color=PALETTE[1], ls="--", lw=1.4, label="insertion depth [mm]")
    ax2.set_ylabel("depth [mm]")
    ax2.grid(False)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in MODE_COLORS]
    ax.legend(
        handles, MODE_NAMES, ncol=6, fontsize=7.5, loc="upper left", title="simulator phase label", title_fontsize=8
    )
    fig.tight_layout()
    save(fig, out / "fig_env.png")


def _ci_table(eps: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    rows = []
    for key, g in eps.groupby(by):
        m, lo, hi = bootstrap_ci(g[metric].to_numpy(float))
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def modality_figure(epsA: pd.DataFrame, out: Path) -> None:
    kinds = [("reactive", "Reactive"), ("sequence", "Sequence BC"), ("world_model", "Predictive world model")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
    for col, (k, title) in enumerate(kinds):
        for row, (metric, ylab) in enumerate((("success", "success rate"), ("peak_force", "peak contact force [N]"))):
            ax = axes[row, col]
            for j, (suffix, lab) in enumerate((("EP", "extero + proprio"), ("EPF", "+ force"))):
                t = (
                    _ci_table(epsA[epsA.condition == f"{k}_{suffix}"], ["domain"], metric)
                    .set_index("domain")
                    .reindex(DOMS)
                )
                x = np.arange(len(DOMS)) + (j - 0.5) * 0.28
                ax.errorbar(
                    x,
                    t["mean"],
                    yerr=[t["mean"] - t.lo, t.hi - t["mean"]],
                    fmt=MARKERS[j],
                    ms=7,
                    color=PALETTE[[1, 0][j]],
                    mfc=SURFACE if j == 0 else PALETTE[0],
                    mew=2,
                    lw=1.8,
                    label=lab,
                )
            ax.set_xticks(range(len(DOMS)))
            ax.set_xticklabels([DOM_LABEL[d] for d in DOMS], fontsize=8)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Experiment A: with vs without force at matched capacity (mean, 95% CI; 3 seeds pooled)",
        fontsize=11,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, out / "fig_A_modality.png")


def forest(
    df: pd.DataFrame, out: Path, name: str, title: str, label_col: str, metrics=("success", "peak_force")
) -> None:
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.2 * len(metrics), 0.32 * len(df) / len(metrics) + 1.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        d = df[df.metric == m].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        sig = (d.ci_low > 0) | (d.ci_high < 0)
        ax.axvline(0, color=INK2, lw=1)
        for yi, (_, r), s in zip(y, d.iterrows(), sig):
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color=PALETTE[0] if s else "#9a9994", lw=2)
            ax.plot(r["diff"], yi, "o", color=PALETTE[0] if s else "#9a9994", ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels(d[label_col], fontsize=8)
        ax.set_title(
            {
                "success": "difference in success rate",
                "peak_force": "difference in peak force [N]",
                "force_violation": "difference in P(peak > 10 N)",
            }.get(m, m),
            fontsize=10,
        )
    fig.suptitle(title, fontsize=11, color=INK2)
    fig.tight_layout()
    save(fig, out / name)


def objective_figure(epsB: pd.DataFrame, auroc: pd.DataFrame, out: Path) -> None:
    conds = [
        ("sequence_bc", "sequence BC"),
        ("wm_obs", "WM: next obs"),
        ("wm_obs_force", "WM: obs + force"),
        ("wm_obs_force_contactsup", "WM: + sim-supervised contact"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2, 2, 1.2]})
    for ax, (metric, lab) in zip(axes[:2], (("success", "success rate"), ("peak_force", "peak contact force [N]"))):
        for j, (c, name) in enumerate(conds):
            t = _ci_table(epsB[epsB.condition == c], ["domain"], metric).set_index("domain").reindex(DOMS)
            x = np.arange(len(DOMS)) + (j - 1.5) * 0.18
            ax.errorbar(
                x,
                t["mean"],
                yerr=[t["mean"] - t.lo, t.hi - t["mean"]],
                fmt=MARKERS[j],
                ms=6,
                color=PALETTE[j],
                lw=1.6,
                label=name,
            )
        ax.set_xticks(range(len(DOMS)))
        ax.set_xticklabels([DOM_LABEL[d] for d in DOMS], fontsize=8)
        ax.set_ylabel(lab)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.4, -0.06))
    a = auroc[auroc.condition.isin([c for c, _ in conds[1:]])].groupby("condition")["mean"].mean()
    names = [n for c, n in conds[1:]]
    vals = [a.get(c, np.nan) for c, _ in conds[1:]]
    axes[2].bar(range(3), vals, color=[PALETTE[1], PALETTE[2], PALETTE[3]])
    for i, v in enumerate(vals):
        axes[2].text(
            i,
            (v if np.isfinite(v) else 0) + 0.02,
            f"{v:.2f}" if np.isfinite(v) else "n/a\n(no force head)",
            ha="center",
            fontsize=8,
        )
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(0.5, color=INK2, ls=":", lw=1)
    axes[2].set_title("first-contact prediction AUROC", fontsize=10)
    fig.suptitle("Experiment B: objective ablation at matched capacity (mean, 95% CI)", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out / "fig_B_objective.png")


def timing_figure(epsC: pd.DataFrame, out: Path) -> None:
    corr = [
        ("none", "intact"),
        ("random_window", "random 0.3 s window"),
        ("first_contact_window", "0.3 s at first contact"),
        ("zero", "zeroed whole episode"),
    ]
    conds = list(dict.fromkeys(epsC.condition))
    doms = list(dict.fromkeys(epsC.domain))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (metric, lab) in zip(
        axes,
        (("success", "success rate"), ("peak_force", "peak contact force [N]"), ("force_violation", "P(peak > 10 N)")),
    ):
        xt = []
        for ci, c in enumerate(conds):
            for di, d in enumerate(doms):
                base = (ci * len(doms) + di) * 1.2
                xt.append((base + 0.45, f"{c.replace('_EPF', '')}\n{d.replace('_', ' ')}"))
                for j, (cm, _) in enumerate(corr):
                    g = epsC[(epsC.condition == c) & (epsC.domain == d) & (epsC.corruption == cm)]
                    m, lo, hi = bootstrap_ci(g[metric].to_numpy(float))
                    ax.errorbar(
                        base + j * 0.3,
                        m,
                        yerr=[[m - lo], [hi - m]],
                        fmt=MARKERS[j],
                        color=PALETTE[j],
                        ms=6,
                        lw=1.6,
                        label=corr[j][1] if (ci == 0 and di == 0) else None,
                    )
        ax.set_xticks([x for x, _ in xt])
        ax.set_xticklabels([t for _, t in xt], fontsize=7.5)
        ax.set_ylabel(lab)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Experiment C: when is force needed? (mean, 95% CI; paired episodes)", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out / "fig_C_timing.png")


def reuse_figures(run: Path, out: Path, corr: dict) -> None:
    pts = pd.read_csv(run / "reuse_points.csv")
    curves = pd.read_csv(run / "adaptation_curves.csv")
    cka = pd.read_csv(run / "cka_by_phase.csv")
    models = list(dict.fromkeys(pts.model))
    names = {"sequence_bc": "sequence BC", "wm_selfsup": "WM self-supervised", "wm_contactsup": "WM sim-supervised aux"}
    # (a) probes and reuse vs severity
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for j, m in enumerate(models):
        g = pts[pts.model == m].groupby("severity")
        for ax, col in zip(axes, ("probe_source_bacc", "probe_cross_domain_bacc", "reuse")):
            s = g[col].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                s.severity + (j - 1) * 0.015,
                s["mean"],
                yerr=s["std"],
                fmt=MARKERS[j] + "-",
                color=PALETTE[j],
                ms=6,
                lw=1.6,
                capsize=3,
                label=names.get(m, m),
            )
    for ax, t in zip(
        axes, ("within-source probe (balanced acc.)", "train source, test shifted", "reuse score (null-normalised CKA)")
    ):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("tight-clearance severity")
    axes[0].axhline(1 / 6, color=INK2, ls=":", lw=1)
    axes[1].axhline(1 / 6, color=INK2, ls=":", lw=1)
    axes[2].axhline(0, color=INK2, ls=":", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Experiment D: contact-mode decoding and representation reuse (mean +/- s.d. over 3 seeds)",
        fontsize=11,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, out / "fig_D_probe_reuse.png")
    # (b) CKA by phase at the strongest shift
    sev = cka.severity.max()
    c = cka[cka.severity == sev].groupby(["model", "phase"])[["cka", "null_mean"]].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    phases = sorted(c.phase.unique())
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        r = c[c.model == m].set_index("phase").reindex(phases)
        x = np.arange(len(phases)) - 0.4 + w * (j + 0.5)
        ax.bar(
            x, r[("cka", "mean")], w * 0.9, yerr=r[("cka", "std")], color=PALETTE[j], label=names.get(m, m), capsize=2
        )
        ax.plot(
            x,
            r[("null_mean", "mean")],
            "_",
            color="#0b0b0b",
            ms=14,
            mew=2,
            label="permutation null" if j == 0 else None,
        )
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([MODE_NAMES[p].replace("_", " ") for p in phases])
    ax.set_ylabel("linear CKA, source vs shifted")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"phase-matched similarity at severity {sev:g} (mean +/- s.d. over seeds)", fontsize=10)
    save(fig, out / "fig_D_cka.png")
    # (c) adaptation curves and reuse vs adaptation
    sevs = sorted(curves.severity.unique())
    fig, axes = plt.subplots(1, len(sevs) + 1, figsize=(4.2 * (len(sevs) + 1), 3.5))
    for ax, s in zip(axes, sevs):
        for j, m in enumerate(models):
            g = curves[(curves.severity == s) & (curves.model == m)].groupby("budget").success.agg(["mean", "std"])
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=g["std"],
                fmt=MARKERS[j] + "-",
                color=PALETTE[j],
                ms=5,
                capsize=3,
                label=names.get(m, m),
            )
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks(sorted(curves.budget.unique()))
        ax.set_xticklabels([str(b) for b in sorted(curves.budget.unique())])
        ax.set_xlabel("fine-tuning demonstrations")
        ax.set_title(f"severity {s:g}", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("success rate (shifted domain)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    ax = axes[-1]
    for j, m in enumerate(models):
        g = pts[pts.model == m]
        ax.scatter(g.reuse, g.adaptation_auc, color=PALETTE[j], marker=MARKERS[j], s=45, label=names.get(m, m))
    r = corr["reuse_vs_adaptation_auc"]
    ax.set_xlabel("reuse score")
    ax.set_ylabel("adaptation efficiency")
    ax.set_title("reuse vs adaptation", fontsize=10)
    ax.text(
        0.02,
        0.02,
        f"Spearman rho = {r['rho']:.2f}\n95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}], n = {r['n']}",
        transform=ax.transAxes,
        fontsize=8,
        color=INK2,
    )
    fig.suptitle("Experiment D: adaptation to tight clearance (mean +/- s.d. over seeds)", fontsize=11, color=INK2)
    fig.tight_layout()
    save(fig, out / "fig_D_adaptation.png")


def ablation_figure(rep: Path, out: Path) -> None:
    """Latent feature ablation for each analysed model (success and peak force)."""
    models = [m for m in ("sequence_bc", "wm_selfsup") if (rep / m / "latent_ablation_results.csv").exists()]
    if not models:
        return
    names = {"sequence_bc": "sequence BC", "wm_selfsup": "WM self-supervised"}
    sets = [
        ("none", "no ablation"),
        ("contact", "contact-ranked"),
        ("movement", "movement-ranked"),
        ("random", "random"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (metric, lab) in zip(axes, (("success", "success rate"), ("peak_force", "peak contact force [N]"))):
        for i, m in enumerate(models):
            r = pd.read_csv(rep / m / "latent_ablation_results.csv")
            r = r[r.metric == metric]
            for j, (k, kl) in enumerate(sets):
                for di, dom in enumerate(("source", "tight_clearance")):
                    q = r[(r.ablation == k) & (r.domain == dom)]
                    if q.empty:
                        continue
                    q = q.iloc[0]
                    x = i * 2.6 + di * 1.2 + j * 0.22
                    ax.errorbar(
                        x,
                        q["mean"],
                        yerr=[[q["mean"] - q.ci_low], [q.ci_high - q["mean"]]],
                        fmt=MARKERS[j],
                        color=PALETTE[j],
                        ms=6,
                        lw=1.6,
                        label=kl if (i == 0 and di == 0) else None,
                    )
        ax.set_xticks([i * 2.6 + di * 1.2 + 0.33 for i in range(len(models)) for di in range(2)])
        ax.set_xticklabels([f"{names[m]}\n{d}" for m in models for d in ("source", "tight clearance")], fontsize=8)
        ax.set_ylabel(lab)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Latent feature ablation: 25% of latent dimensions clamped to their mean (seed 0; 40 episodes per cell)",
        fontsize=11,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out / "fig_ablation.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/medium")
    p.add_argument("--out", default="docs/whitepaper/figures")
    a = p.parse_args()
    res, out = Path(a.results), Path(a.out)
    runs = res / "runs"
    env_figure(out)
    epsA = pd.read_csv(runs / "A_modality" / "episode_metrics.csv")
    epsB = pd.read_csv(runs / "B_objective" / "episode_metrics.csv")
    epsC = pd.read_csv(runs / "C_force_timing" / "episode_metrics.csv")
    modality_figure(epsA, out)
    h1 = pd.read_csv(res / "H1_force_vs_noforce.csv")
    h1["label"] = (
        h1.a.str.replace("_EPF", "").str.replace("world_model", "world model") + ", " + h1.domain.str.replace("_", " ")
    )
    forest(h1, out, "fig_A_forest.png", "Experiment A: paired difference, with minus without force", "label")
    objective_figure(epsB, pd.read_csv(res / "first_contact_auroc.csv"), out)
    h5 = pd.read_csv(res / "H5_force_timing.csv")
    h5 = h5[(h5.a == "first_contact_window") & (h5.b == "random_window")].copy()
    h5["label"] = h5.condition.str.replace("_EPF", "") + ", " + h5.domain.str.replace("_", " ")
    timing_figure(epsC, out)
    forest(
        h5,
        out,
        "fig_C_forest.png",
        "Experiment C: force zeroed at first contact minus random window (paired)",
        "label",
        metrics=("success", "peak_force", "force_violation"),
    )
    reuse_figures(runs / "D_reuse", out, json.loads((res / "summary.json").read_text())["H4"])
    ablation_figure(res / "representation", out)
    rep = res / "representation" / "wm_selfsup" / "figures"
    for name in ("pca_source.png", "event_aligned_first_contact.png", "probe_confusion_cross_domain.png"):
        if (rep / name).exists():
            (out / f"fig_repr_{name}").write_bytes((rep / name).read_bytes())


if __name__ == "__main__":
    main()
