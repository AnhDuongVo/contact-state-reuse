"""Figures. Every function takes data computed elsewhere and writes one PNG.

Colour follows the condition or label, never the rank, and identity is also
carried by markers or text labels so figures stay readable without colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from ..data.dataset import TrajectoryData  # noqa: E402
from ..envs.labels import MODE_NAMES  # noqa: E402
from .latents import LatentSet  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MODE_COLORS = ["#9ec5f4", "#eb6834", "#eda100", "#2a78d6", "#008300", "#e34948"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "legend.frameon": False,
        "figure.dpi": 120,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_episode_overview(data: TrajectoryData, path: str | Path, n_episodes: int = 3) -> Path:
    """Contact force and insertion depth over time with simulator phases shaded."""
    a = data.arrays
    eps = [e for e in range(data.n_episodes) if a["ep_success"][e]][:n_episodes] or [0]
    fig, axes = plt.subplots(len(eps), 1, figsize=(8, 2.2 * len(eps)), squeeze=False)
    for ax, e in zip(axes[:, 0], eps):
        r = data.episode_rows(e)
        t = a["t"][r] * data.meta["env_config"]["control_dt"]
        modes = a["contact_mode"][r]
        for k in range(len(t) - 1):
            ax.axvspan(t[k], t[k + 1], color=MODE_COLORS[modes[k]], alpha=0.18, lw=0)
        ax.plot(t, a["contact_force_norm"][r], color=INK, lw=1.6, label="true |F| [N]")
        ax.plot(
            t,
            np.hypot(*a["obs"][r][:, data.layout["force"]][:, :2].T),
            color=PALETTE[0],
            lw=1.0,
            alpha=0.8,
            label="observed |F| [N]",
        )
        ax2 = ax.twinx()
        ax2.plot(t, 1e3 * a["insertion_depth"][r], color=PALETTE[1], lw=1.4, ls="--", label="depth [mm]")
        ax2.set_ylabel("depth [mm]")
        ax2.grid(False)
        ax.set_ylabel("force [N]")
        ax.set_title(f"episode {e} ({a['ep_domain'][e]}, {a['ep_variant'][e]})", loc="left")
    axes[-1, 0].set_xlabel("time [s]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in MODE_COLORS]
    fig.legend(
        handles, MODE_NAMES, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), title="simulator phase label"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path)


def plot_condition_metric(
    summary: pd.DataFrame,
    metric: str,
    x: str,
    hue: str,
    path: str | Path,
    title: str,
    ylabel: str | None = None,
    x_order: list | None = None,
    hue_order: list | None = None,
) -> Path:
    """Dot + 95 % CI per (x, hue) for one metric from a tidy summary table."""
    s = summary[summary.metric == metric]
    xs = [v for v in (x_order or []) if v in set(s[x])] or list(dict.fromkeys(s[x]))
    hues = [v for v in (hue_order or []) if v in set(s[hue])] or list(dict.fromkeys(s[hue]))
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(xs) + 2), 3.4))
    width = 0.7 / max(len(hues), 1)
    for j, h in enumerate(hues):
        for i, xv in enumerate(xs):
            r = s[(s[x] == xv) & (s[hue] == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            pos = i - 0.35 + width * (j + 0.5)
            ax.plot([pos, pos], [r.ci_low, r.ci_high], color=PALETTE[j % 8], lw=2)
            ax.plot(pos, r["mean"], MARKERS[j % 8], color=PALETTE[j % 8], ms=7, label=h if i == 0 else None)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, path)


def plot_training_curves(histories: dict[str, list[dict]], path: str | Path) -> Path:
    keys = ["action", "obs", "force", "contact"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    for j, (name, hist) in enumerate(histories.items()):
        h = pd.DataFrame(hist)
        for ax, k in zip(axes, keys):
            col = f"val_{k}" if f"val_{k}" in h else f"train_{k}"
            if col in h:
                ax.plot(h.step, h[col], color=PALETTE[j % 8], marker=MARKERS[j % 8], ms=3, label=name)
    for ax, k in zip(axes, keys):
        ax.set_title(f"{k} loss (validation)")
        ax.set_xlabel("step")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, path: str | Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if cm[i, j] > 0.6 else INK,
            )
    ax.set_xticks(range(len(MODE_NAMES)))
    ax.set_xticklabels(MODE_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(MODE_NAMES)))
    ax.set_yticklabels(MODE_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("simulator label")
    ax.grid(False)
    ax.set_title(title)
    return _save(fig, path)


def plot_pca_trajectories(lat: LatentSet, path: str | Path, max_episodes: int = 40, title: str = "") -> Path:
    """PCA of latents (fit on all rows) with episode trajectories, coloured three ways.

    Descriptive only: a 2-D projection can make continuous variation look
    clustered and can hide structure; it is not evidence of discrete states.
    """
    pcs = PCA(2).fit_transform(lat.z)
    eps = np.unique(lat.episode)[:max_episodes]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    keep = np.isin(lat.episode, eps)
    for e in eps:
        m = lat.episode == e
        for ax in axes:
            ax.plot(pcs[m, 0], pcs[m, 1], color="#b9b8b2", lw=0.5, zorder=1)
    for k, name in enumerate(MODE_NAMES):
        m = keep & (lat.mode == k)
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=7, color=MODE_COLORS[k], label=name, zorder=2)
    sc = axes[1].scatter(pcs[keep, 0], pcs[keep, 1], s=7, c=lat.force_norm[keep], cmap="viridis", zorder=2)
    fig.colorbar(sc, ax=axes[1], label="true contact force [N]")
    for v, name, c in ((True, "success", PALETTE[0]), (False, "failure", PALETTE[7])):
        m = keep & (lat.success == v)
        axes[2].scatter(pcs[m, 0], pcs[m, 1], s=7, color=c, label=name, zorder=2)
    axes[0].legend(markerscale=2, fontsize=7.5, title="simulator label")
    axes[2].legend(markerscale=2)
    for ax, t in zip(axes, ("contact mode", "contact force", "episode outcome")):
        ax.set_title(t)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(
        (title + "  " if title else "") + "(descriptive projection, not evidence of discrete states)",
        fontsize=10,
        color=INK2,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_cka_by_phase(rows: pd.DataFrame, path: str | Path, title: str) -> Path:
    """rows: columns model, phase, cka, null_mean, null_sd."""
    models = list(dict.fromkeys(rows.model))
    phases = sorted(rows.phase.unique())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.7 / max(len(models), 1)
    for j, m in enumerate(models):
        r = rows[rows.model == m].set_index("phase").reindex(phases)
        xs = np.arange(len(phases)) - 0.35 + w * (j + 0.5)
        ax.bar(xs, r.cka, w * 0.9, color=PALETTE[j % 8], label=f"{m}: matched episodes")
        ax.errorbar(
            xs,
            r.null_mean,
            yerr=r.null_sd,
            fmt="_",
            color=INK,
            ms=14,
            lw=1,
            label="permutation null (mean +/- sd)" if j == 0 else None,
        )
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([MODE_NAMES[p] for p in phases])
    ax.set_ylabel("linear CKA (source vs shifted)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1, 0.5))
    return _save(fig, path)


def plot_xy_scatter(
    df: pd.DataFrame, x: str, y: str, hue: str, path: str | Path, title: str, annotation: str = ""
) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for j, (h, g) in enumerate(df.groupby(hue)):
        ax.scatter(g[x], g[y], color=PALETTE[j % 8], marker=MARKERS[j % 8], s=40, label=str(h))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    if annotation:
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(fontsize=8, title=hue)
    return _save(fig, path)


def plot_curves(
    df: pd.DataFrame, x: str, y: str, hue: str, path: str | Path, title: str, ylabel: str | None = None
) -> Path:
    """Mean over seeds with min-max band, one line per hue level."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for j, (h, g) in enumerate(df.groupby(hue)):
        s = g.groupby(x)[y].agg(["mean", "min", "max"]).reset_index()
        ax.fill_between(s[x], s["min"], s["max"], color=PALETTE[j % 8], alpha=0.12, lw=0)
        ax.plot(s[x], s["mean"], marker=MARKERS[j % 8], color=PALETTE[j % 8], label=str(h))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)
