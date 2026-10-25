"""Figures for the revised white paper (v0.2), from results/revision.

python scripts/make_revision_figures.py --results results/revision --out docs/whitepaper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contact_state_reuse.analysis.plotting import INK, INK2, PALETTE, _save  # noqa: E402

BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, PURPLE, RED = PALETTE
DOM_LABEL = {
    "source": "source",
    "shifted_friction": "friction",
    "tight_clearance": "tight fit",
    "sensor_noise": "sensor fault",
    "occlusion": "occlusion",
    "pooled": "all five",
}


def _ci(ax, x, r, color, marker="o", label=None, ms=6):
    ax.errorbar(
        x,
        r["mean"],
        yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
        fmt=marker,
        color=color,
        ms=ms,
        capsize=2.5,
        lw=1.4,
        label=label,
        zorder=3,
    )


def _seeds(ax, x, per_seed, color):
    ps = np.asarray(per_seed, float)
    jit = np.linspace(-0.08, 0.08, len(ps))
    ax.scatter(x + jit, ps, s=10, color=color, alpha=0.45, lw=0, zorder=2)


def _cell(tab, cond, dom, metric):
    r = tab[(tab.condition == cond) & (tab.domain == dom) & (tab.metric == metric)]
    if r.empty:
        return None
    r = r.iloc[0].to_dict()
    if isinstance(r.get("per_seed"), str):
        r["per_seed"] = json.loads(r["per_seed"])
    return r


# ---------------------------------------------------------------- factorial
def fig_factorial(res: Path, out: Path) -> None:
    tab = pd.read_csv(res / "E_conditions.csv")
    groups = [
        ("BC only", "bc", "bc_F"),
        ("+ force\ntarget", "ftgt", "F_ftgt"),
        ("+ obs\nprediction", "obs", "obs_F"),
        ("+ obs and\nforce targets", "obs_ftgt", "obs_F_ftgt"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (metric, dom, ylab) in zip(
        axes,
        (
            ("success", "pooled", "success rate (all five domains)"),
            ("peak_force", "pooled", "peak contact force [N]"),
            ("success", "sensor_noise", "success rate (sensor fault)"),
        ),
    ):
        for i, (_lab, no_f, with_f) in enumerate(groups):
            for j, (cond, col, mk, name) in enumerate(
                ((no_f, BLUE, "o", "no force input"), (with_f, ORANGE, "s", "force input"))
            ):
                r = _cell(tab, cond, dom, metric)
                if r is None:
                    continue
                x = i + (j - 0.5) * 0.3
                _seeds(ax, x, r["per_seed"], col)
                _ci(ax, x, r, col, mk, label=name if i == 0 else None)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Fixed trunk, identical data: which training signal makes force useful? (5 seeds; dots = seeds)", fontsize=10.5
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out / "fig_factorial.png")


def fig_domains(res: Path, out: Path) -> None:
    tab = pd.read_csv(res / "E_conditions.csv")
    conds = [
        ("bc", "BC", BLUE, "o"),
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED, "P"),
        ("ftgt", "force target only", PURPLE, "v"),
        ("obs_ftgt", "obs + force pred, no force input", DGREEN, "X"),
        ("obs_F_ftgt", "obs + force pred, force input", GREEN, "^"),
    ]
    doms = ["source", "shifted_friction", "tight_clearance", "sensor_noise", "occlusion"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, metric, ylab in ((axes[0], "success", "success rate"), (axes[1], "peak_force", "peak contact force [N]")):
        for j, (c, lab, col, mk) in enumerate(conds):
            for i, d in enumerate(doms):
                r = _cell(tab, c, d, metric)
                if r is not None:
                    _ci(ax, i + (j - 2.5) * 0.12, r, col, mk, label=lab if i == 0 else None, ms=5)
        ax.set_xticks(range(len(doms)), [DOM_LABEL[d] for d in doms])
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out / "fig_domains.png")


CONTROL_GROUPS = [
    (
        "factorial",
        [
            "force input | BC",
            "force input | obs pred + force target",
            "force target | no force input",
            "obs pred | force input",
            "full vs BC with force",
        ],
    ),
    ("training data", ["all episodes | BC", "all episodes | obs pred + force target"]),
    (
        "auxiliary-loss controls",
        [
            "reconstruction vs BC",
            "prediction vs reconstruction",
            "separate predictive core vs BC",
            "shared vs separate predictive core",
            "stop-gradient vs BC",
            "weight 0.1 vs BC",
            "weight 3 vs 1",
        ],
    ),
    (
        "tuning the baseline",
        [
            "lr 3e-4 vs 1e-3 | BC",
            "lr 3e-4 vs 1e-3 | BC force",
            "force input | BC, lr 3e-4",
            "full vs BC force | both lr 3e-4",
            "full (lr 1e-3) vs tuned BC force (lr 3e-4)",
            "3000 vs 1500 updates | BC force",
            "2x capacity | BC force",
        ],
    ),
    (
        "BC inputs and data",
        [
            "no previous action | BC force",
            "log force vs raw | BC",
            "contact flag vs raw | BC",
            "force input | BC without DART noise",
            "fault-aware vs standard | BC",
        ],
    ),
]


def fig_controls(res: Path, out: Path) -> None:
    con = pd.read_csv(res / "E_contrasts.csv")
    con = con[con.domain == "pooled"]
    rows, ticks, seps, heads = [], [], [], []
    y = 0
    for title, names in CONTROL_GROUPS:
        heads.append((y - 0.6, title))
        for n in names:
            if (con.contrast == n).any():
                rows.append((y, n))
                ticks.append(n)
                y += 1
        seps.append(y - 0.5)
        y += 0.8
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.3 * y + 1.2), sharey=True)
    for ax, metric, xl in (
        (axes[0], "success", "difference in success rate"),
        (axes[1], "peak_force", "difference in peak force [N]"),
    ):
        for yy, n in rows:
            r = con[(con.contrast == n) & (con.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            sig = r.ci_low > 0 or r.ci_high < 0
            col = INK if sig else "#9a9993"
            ax.errorbar(
                r["diff"],
                yy,
                xerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]],
                fmt="o",
                color=col,
                ms=4.5,
                capsize=2,
                lw=1.2,
            )
        ax.axvline(0, color=INK2, lw=0.8)
        for sp in seps[:-1]:
            ax.axhline(sp + 0.4, color="#d8d7d2", lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    for yy, t in heads:
        axes[0].text(
            -0.02,
            yy,
            t,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=INK2,
        )
    axes[0].set_yticks([r[0] for r in rows], [r[1] for r in rows], fontsize=8)
    axes[0].set_ylim(y - 0.5, -1.2)
    fig.suptitle(
        "Paired contrasts (a minus b), pooled over the five domains; crossed seed x episode 95% intervals; "
        "black = interval excludes 0",
        fontsize=9.5,
    )
    fig.tight_layout()
    _save(fig, out / "fig_controls.png")


# ------------------------------------------------------------------- timing
def fig_timing(res: Path, out: Path) -> None:
    tab = pd.read_csv(res / "T_results.csv")
    con = pd.read_csv(res / "T_contrasts.csv")
    order = [
        "first_contact_window",
        "onset+6",
        "onset+12",
        "onset+24",
        "random_window",
        "hold_onset",
        "onset_only",
        "zero",
    ]
    labels = [
        "onset\n(0-0.3 s)",
        "+0.3 s",
        "+0.6 s",
        "+1.2 s",
        "random",
        "hold\nonset",
        "force only\nat onset",
        "no force",
    ]
    conds = [
        ("obs_F_ftgt", "obs + force prediction", GREEN, "^"),
        ("obs_F", "obs prediction", YELLOW, "D"),
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED, "P"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
    for col_i, dom in enumerate(("source", "tight_clearance")):
        for j, (c, lab, col, mk) in enumerate(conds):
            for i, cor in enumerate(order):
                r = con[
                    (con.condition == c)
                    & (con.domain == dom)
                    & (con.a == cor)
                    & (con.b == "none")
                    & (con.metric == "peak_force")
                ]
                if r.empty:
                    continue
                r = r.iloc[0]
                d = {"mean": r["diff"], "ci_low": r.ci_low, "ci_high": r.ci_high}
                _ci(axes[0, col_i], i + (j - 1.5) * 0.17, d, col, mk, label=lab if i == 0 else None, ms=5)
                t = tab[
                    (tab.condition == c)
                    & (tab.domain == dom)
                    & (tab.corruption == cor)
                    & (tab.metric == "masked_force")
                ]
                if len(t):
                    axes[1, col_i].bar(i + (j - 1.5) * 0.19, t.iloc[0]["mean"], width=0.18, color=col, alpha=0.8)
        axes[0, col_i].axhline(0, color=INK2, lw=0.8)
        axes[0, col_i].set_title(DOM_LABEL[dom])
        axes[0, col_i].set_ylabel("change in peak force [N]\n(paired, vs intact force)")
        axes[1, col_i].set_ylabel("force signal removed\n[N, summed per episode]")
        axes[1, col_i].set_yscale("symlog", linthresh=10)
        axes[1, col_i].set_xticks(range(len(order)), labels, fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Removing the force reading for 0.3 s at different times (5 seeds, 60 episodes each)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out / "fig_timing.png")


def fig_behaviour(res: Path, out: Path) -> None:
    beh = pd.read_csv(res / "E_behaviour.csv")
    comp = pd.read_csv(res / "T_compliance.csv") if (res / "T_compliance.csv").exists() else None
    conds = [
        ("bc", "BC", BLUE),
        ("bc_F", "BC + force", ORANGE),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED),
        ("obs_ftgt", "obs + force pred,\nno force input", DGREEN),
        ("obs_F", "obs pred + force", YELLOW),
        ("obs_F_ftgt", "obs + force pred", GREEN),
    ]
    mets = [
        ("compliance_in_contact", "commanded stiffness in contact\n(0 soft, 1 stiff)"),
        ("approach_speed", "peg speed before contact [m/s]"),
        ("steps", "episode length [steps]"),
    ]
    n = len(mets) + (1 if comp is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    for ax, (m, lab) in zip(axes, mets):
        for i, (c, _name, col) in enumerate(conds):
            r = beh[(beh.condition == c) & (beh.domain == "source") & (beh.metric == m)]
            if len(r):
                r = r.iloc[0].to_dict()
                if isinstance(r["per_seed"], str):
                    r["per_seed"] = json.loads(r["per_seed"])
                _seeds(ax, i, r["per_seed"], col)
                _ci(ax, i, r, col)
        ax.set_xticks(range(len(conds)), [c[1] for c in conds], rotation=35, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="x", visible=False)
    if comp is not None:
        ax = axes[-1]
        for k, (ov, mk, lab) in enumerate(
            (("policy", "o", "policy's own"), ("-0.5", "s", "fixed soft"), ("0.5", "^", "fixed stiff"))
        ):
            for i, (c, _name, col) in enumerate(conds):
                r = comp[
                    (comp.condition == c)
                    & (comp.domain == "source")
                    & (comp.compliance == ov)
                    & (comp.metric == "peak_force")
                ]
                if len(r):
                    _ci(
                        ax,
                        i + (k - 1) * 0.22,
                        r.iloc[0].to_dict(),
                        col,
                        mk,
                        ms=4.5,
                        label=lab if i == len(conds) - 1 else None,
                    )
        ax.set_xticks(range(len(conds)), [c[1] for c in conds], rotation=35, ha="right", fontsize=8)
        ax.set_title("peak force [N] with compliance\nfixed by the harness", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3)
        ax.grid(axis="x", visible=False)
    fig.suptitle("How the policies behave (source domain, 5 seeds)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, out / "fig_behaviour.png")


def fig_fault(res: Path, out: Path) -> None:
    tab = pd.read_csv(res / "E_conditions.csv")
    ext = pd.read_csv(res / "E_fault_extra_conditions.csv") if (res / "E_fault_extra_conditions.csv").exists() else None
    conds = [
        ("bc", "BC (no force)", BLUE, "o"),
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_aug", "BC + force, fault-aware", PINK, "P"),
        ("obs_ftgt", "obs + force pred, no force input", PURPLE, "v"),
        ("obs_F_ftgt", "obs + force pred", GREEN, "^"),
        ("obs_F_ftgt_aug", "obs + force pred, fault-aware", DGREEN, "X"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    xs = ["source", "sensor_noise", "delay", "bias", "noise"]
    xl = ["source", "sensor-fault\ndomain", "delay 4 steps\n(source)", "bias 3 N\n(source)", "noise 3 N\n(source)"]
    for j, (c, lab, col, mk) in enumerate(conds):
        for i, x in enumerate(xs):
            if x in ("source", "sensor_noise"):
                r = _cell(tab, c, x, "success")
            elif ext is not None:
                rr = ext[(ext.condition == c) & (ext.corruption == x) & (ext.metric == "success")]
                r = rr.iloc[0].to_dict() if len(rr) else None
            else:
                r = None
            if r is not None:
                _ci(ax, i + (j - 2.5) * 0.12, r, col, mk, label=lab if i == 0 else None, ms=5)
    ax.set_xticks(range(len(xs)), xl, fontsize=8.5)
    ax.set_ylabel("success rate")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Fault-aware training: synthetic sensor faults added to a copy of every training episode", fontsize=9.5
    )
    fig.tight_layout()
    _save(fig, out / "fig_fault.png")


# ------------------------------------------------------------ generalisation
def fig_generalisation(res: Path, out: Path) -> None:
    files = [("G_nonpriv", "second demonstrator (non-privileged)"), ("G_search", "second task (hole search)")]
    have = [(f, t) for f, t in files if (res / f"{f}_conditions.csv").exists()]
    if not have:
        return
    conds = [
        ("bc", "BC", BLUE, "o"),
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED, "P"),
        ("obs", "obs pred", YELLOW, "D"),
        ("obs_ftgt", "obs + force pred, no force input", PURPLE, "v"),
        ("obs_F_ftgt", "obs + force pred, force input", GREEN, "^"),
    ]
    sc = pd.read_csv(res / "scripted_baselines.csv") if (res / "scripted_baselines.csv").exists() else None
    fig, axes = plt.subplots(2, len(have), figsize=(5.6 * len(have), 5.6), squeeze=False)
    for k, (f, title) in enumerate(have):
        tab = pd.read_csv(res / f"{f}_conditions.csv")
        doms = [d for d in tab.domain.unique() if d != "pooled"]
        for row, metric, ylab in ((0, "success", "success rate"), (1, "peak_force", "peak contact force [N]")):
            ax = axes[row, k]
            for j, (c, lab, col, mk) in enumerate(conds):
                for i, d in enumerate(doms):
                    r = _cell(tab, c, d, metric)
                    if r is not None:
                        _ci(ax, i + (j - 2.5) * 0.12, r, col, mk, label=lab if i == 0 else None, ms=5)
            if sc is not None and metric == "success":
                ctrl = "non_privileged" if f == "G_nonpriv" else "search"
                for i, d in enumerate(doms):
                    v = sc[(sc.controller == ctrl) & (sc.domain == d)]
                    if len(v):
                        ax.plot(
                            [i - 0.35, i + 0.35],
                            [v.success.iloc[0]] * 2,
                            color=INK2,
                            lw=1.2,
                            ls="--",
                            label="scripted controller" if i == 0 else None,
                        )
            ax.set_xticks(range(len(doms)), [d.replace("search+", "+").replace("_", " ") for d in doms], fontsize=8)
            ax.set_ylabel(ylab)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(title, fontsize=10)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out / "fig_generalisation.png")


# --------------------------------------------------------------- adaptation
def fig_adaptation(res: Path, out: Path) -> None:
    if not (res / "R_adaptation_curves.csv").exists():
        return
    cur = pd.read_csv(res / "R_adaptation_curves.csv")
    models = [
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED, "P"),
        ("obs_F", "obs pred + force", YELLOW, "D"),
        ("obs_F_ftgt", "obs + force pred", GREEN, "^"),
    ]
    panels = [
        ("search", 0.5, "search", "large pose error, severity 0.5"),
        ("search", 0.75, "search", "severity 0.75"),
        ("search", 1.0, "search", "severity 1"),
        ("source_ctrl", 1.0, "source", "control: fine-tune on source"),
        ("tight_lr3e-4", 1.0, "tight_clearance", "tight fit, lr 3e-4"),
        ("tight_lr1e-4", 1.0, "tight_clearance", "tight fit, lr 1e-4"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, (ft, sev, dom, title) in zip(axes.flat, panels):
        sub = cur[(cur.ft == ft) & (cur.severity == sev) & (cur.domain == dom)]
        for c, lab, col, mk in models:
            g = sub[sub.condition == c].groupby("budget").success.agg(["mean", "std"]).reset_index()
            if g.empty:
                continue
            ax.errorbar(g.budget, g["mean"], yerr=g["std"], color=col, marker=mk, ms=4.5, capsize=2, lw=1.4, label=lab)
        if ft.startswith("tight"):
            for c, _lab, col, _mk in models:
                g = (
                    cur[(cur.ft == ft) & (cur.domain == "source") & (cur.condition == c)]
                    .groupby("budget")
                    .success.mean()
                )
                if len(g):
                    ax.plot(g.index, g.values, color=col, ls=":", lw=1.2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("fine-tuning demonstrations")
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("success rate")
    axes[1, 0].set_ylabel("success rate")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Fine-tuning (mean +/- s.d. over 5 seeds; dotted: success on the source domain after fine-tuning)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out / "fig_adaptation.png")


def fig_reuse(res: Path, out: Path) -> None:
    if not (res / "R_reuse_scores.csv").exists():
        return
    rs = pd.read_csv(res / "R_reuse_scores.csv")
    cka = pd.read_csv(res / "R_cka_by_phase.csv")
    d = rs[rs.variant == "default"].copy()
    d["family"] = d.model.str.replace("untrained_.*", "untrained", regex=True)
    fams = [
        ("ref_obs", "raw observations", INK2, "o"),
        ("ref_proprio", "proprioception", "#9a9993", "v"),
        ("ref_progress", "time / progress", PURPLE, "P"),
        ("untrained", "untrained network", PINK, "x"),
        ("bc_F", "BC + force", ORANGE, "s"),
        ("bc_F_lr3e-4", "BC + force, lr 3e-4", RED, "P"),
        ("obs_F", "obs pred + force", YELLOW, "D"),
        ("obs_F_ftgt", "obs + force pred", GREEN, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    sevs = sorted(d.severity.unique())
    for j, (f, lab, col, mk) in enumerate(fams):
        g = d[d.family == f].groupby("severity").reuse.agg(["mean", "std"]).reindex(sevs)
        ax.errorbar(
            np.arange(len(sevs)) + (j - 3.5) * 0.08,
            g["mean"],
            yerr=g["std"].fillna(0),
            fmt=mk,
            color=col,
            ms=5,
            capsize=2,
            label=lab,
        )
    ax.set_xticks(range(len(sevs)), [f"severity {s:g}" for s in sevs])
    ax.set_ylabel("reuse score (null-normalised CKA)")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    ax.set_title("Paired source vs shifted latents, compared with reference representations", fontsize=9.5)
    ax = axes[1]
    sub = cka[(cka.severity == max(sevs)) & cka.model.isin(["obs_F_ftgt", "bc_F", "ref_obs"])]
    phases = ["approach", "first contact", "alignment", "insertion"]
    for j, (m, lab, col) in enumerate(
        (
            ("ref_obs", "raw observations", INK2),
            ("bc_F", "BC + force", ORANGE),
            ("obs_F_ftgt", "obs + force pred", GREEN),
        )
    ):
        g = sub[sub.model == m].groupby("phase")[["cka", "null_q05", "null_q95", "null_mean"]].mean()
        x = g.index.to_numpy() + (j - 1) * 0.25
        ax.bar(x, g.cka, width=0.23, color=col, alpha=0.85, label=f"{lab}: CKA")
        ax.vlines(x, g.null_q05, g.null_q95, color=INK, lw=2.2)
        ax.scatter(x, g.null_mean, color="white", edgecolor=INK, s=14, zorder=3)
    ax.set_xticks(range(4), phases)
    ax.set_ylabel("linear CKA")
    ax.set_title("Severity 1: CKA (bars) and permutation null (5th to 95th percentile, mean)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out / "fig_reuse.png")


def fig_ablation(res: Path, out: Path) -> None:
    if not (res / "L_results.csv").exists():
        return
    tab = pd.read_csv(res / "L_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    abl = [
        ("none", "none", INK2, "o"),
        ("contact", "contact-ranked", RED, "s"),
        ("movement", "movement-ranked", BLUE, "D"),
        ("random", "random", "#9a9993", "^"),
    ]
    groups = [(c, d) for c in ("bc_F", "bc_F_lr3e-4", "obs_F_ftgt") for d in ("source", "tight_clearance")]
    names = {"bc_F": "BC + force", "bc_F_lr3e-4": "BC + force,\nlr 3e-4", "obs_F_ftgt": "obs + force pred"}
    for ax, metric, ylab in ((axes[0], "success", "success rate"), (axes[1], "peak_force", "peak force [N]")):
        for j, (a, lab, col, mk) in enumerate(abl):
            for i, (c, d) in enumerate(groups):
                r = tab[(tab.condition == c) & (tab.domain == d) & (tab.ablation == a) & (tab.metric == metric)]
                if len(r):
                    rr = r.iloc[0].to_dict()
                    if isinstance(rr["per_seed"], str):
                        rr["per_seed"] = json.loads(rr["per_seed"])
                    _ci(ax, i + (j - 1.5) * 0.17, rr, col, mk, label=lab if i == 0 else None, ms=5)
        ax.set_xticks(range(len(groups)), [f"{names[c]}\n{DOM_LABEL[d]}" for c, d in groups], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Clamping 25% of latent dimensions to their mean (5 seeds, 40 episodes per cell)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out / "fig_ablation_seeds.png")


def fig_force_input(res: Path, out: Path) -> None:
    p = res / "force_input_analysis.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    roll = [
        ("bc_F", "BC + force", ORANGE),
        ("bc_F_log", "BC + log force", PINK),
        ("bc_F_binary", "BC + contact flag", PURPLE),
        ("bc_F_aug", "BC + force, fault-aware", RED),
        ("obs_F", "obs pred + force", YELLOW),
        ("obs_F_ftgt", "obs + force pred", GREEN),
    ]
    sens = [
        ("bc_F", "BC + force", ORANGE),
        ("bc_F_noprev", "BC + force,\nno prev. action", BLUE),
        ("bc_F_aug", "BC + force,\nfault-aware", RED),
        ("obs_F", "obs pred + force", YELLOW),
        ("obs_F_ftgt", "obs + force pred", GREEN),
        ("obs_F_ftgt_aug", "obs + force pred,\nfault-aware", DGREEN),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    items = [("demonstrations", d["demonstrations"]["all"], INK2, None)]
    items += [
        (lab, d["rollouts"][c]["all"], col, d["rollouts"][c]["frac_above_demo_q99"])
        for c, lab, col in roll
        if c in d["rollouts"]
    ]
    q99 = d["demonstrations"]["all"]["q99"]
    for i, (_lab, qq, col, frac) in enumerate(items):
        ax.vlines(i, qq["q50"], qq["q99"], color=col, lw=3)
        ax.scatter([i, i, i], [qq["q50"], qq["q90"], qq["q99"]], color=col, s=[14, 24, 34], zorder=3)
        if frac is not None:
            ax.text(i, qq["q99"] * 1.25, f"{100 * frac:.1f}%", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(q99, color=INK2, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(items)), [it[0] for it in items], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("observed |F| per step [N]\n(median, 90th, 99th percentile)")
    ax.set_title(
        "Force readings in demonstrations vs own rollouts (source);\n"
        "% = steps above the demonstrations' 99th percentile (dotted)",
        fontsize=9,
    )
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    s = d["sensitivity"]
    have = [c for c in sens if c[0] in s]
    for i, (c, _lab, col) in enumerate(have):
        for k, (key, mk) in enumerate((("free_space", "o"), ("in_contact", "s"))):
            m, sd = s[c][key]
            ax.errorbar(
                i + (k - 0.5) * 0.25,
                m,
                yerr=0 if np.isnan(sd) else sd,
                fmt=mk,
                color=col,
                capsize=2,
                ms=5,
                label=("free space" if k == 0 else "in contact") if i == 0 else None,
                mfc=col if k else "white",
            )
    ax.set_xticks(range(len(have)), [h[1] for h in have], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("||d action / d force|| [1/N]")
    ax.set_title(
        "Sensitivity of the action to the current force reading\n(held-out demonstrations, mean +/- s.d. over 5 seeds)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out / "fig_force_input.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/revision")
    p.add_argument("--out", default="docs/whitepaper/figures")
    a = p.parse_args()
    res, out = Path(a.results), Path(a.out)
    for f in (
        fig_factorial,
        fig_domains,
        fig_controls,
        fig_timing,
        fig_behaviour,
        fig_fault,
        fig_generalisation,
        fig_adaptation,
        fig_reuse,
        fig_ablation,
        fig_force_input,
    ):
        try:
            f(res, out)
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
