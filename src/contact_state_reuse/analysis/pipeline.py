"""End-to-end representation analysis for one checkpoint and a paired pair of datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.dataset import load_dataset
from ..evaluation.metrics import summarize
from ..evaluation.rollout import PolicyAgent, run_episodes
from ..models.factory import load_checkpoint
from ..utils.io import save_json
from .ablation import clamp_fn, dimension_sets
from .latents import align, extract_latents
from .plotting import PALETTE, _save, plot_cka_by_phase, plot_condition_metric, plot_confusion, plot_pca_trajectories
from .probes import cross_domain_probe, grouped_cv_probe
from .reuse import reuse_score


def plot_event_aligned(aligned: dict[str, np.ndarray], before: int, path: str | Path, title: str) -> Path:
    """Mean distance of z_t from the last pre-contact latent, aligned to first contact."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    t = np.arange(-before, next(iter(aligned.values())).shape[1] - before)
    for j, (name, arr) in enumerate(aligned.items()):
        ref = arr[:, before - 1 : before]
        d = np.linalg.norm(arr - ref, axis=-1)
        m, s = np.nanmean(d, 0), np.nanstd(d, 0)
        ax.fill_between(t, m - s, m + s, color=PALETTE[j], alpha=0.15, lw=0)
        ax.plot(t, m, color=PALETTE[j], marker="o", ms=3, label=f"{name} (n={len(arr)})")
    ax.axvline(0, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("steps relative to first contact")
    ax.set_ylabel("||z_t - z_(first contact - 1)||")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def analyze_representations(
    checkpoint: str | Path,
    source: str | Path,
    shifted: str | Path,
    out_dir: str | Path,
    ablation_episodes: int = 30,
    ablation_fraction: float = 0.25,
    seed: int = 0,
    eval_seed: int = 123,
) -> dict[str, Any]:
    """Probes, reuse/CKA, PCA, event alignment and latent feature ablation.

    ``source`` and ``shifted`` must be datasets generated with the same seed
    (paired initial conditions) for the reuse score to be meaningful.
    """
    out = Path(out_dir)
    figs = out / "figures"
    model, mcfg, norm, _ = load_checkpoint(checkpoint)
    src, tgt = load_dataset(source), load_dataset(shifted)
    shift_name = str(tgt.arrays["ep_domain"][0])
    ls, lt = extract_latents(model, norm, src), extract_latents(model, norm, tgt)
    res: dict[str, Any] = {"checkpoint": str(checkpoint), "model": mcfg.to_dict(), "shift": shift_name}

    p_src = grouped_cv_probe(ls.z, ls.mode, ls.episode, seed=seed)
    p_x = cross_domain_probe(ls.z, ls.mode, lt.z, lt.mode, seed=seed)
    res["probe_source"] = {
        "balanced_accuracy": p_src.balanced_accuracy,
        "fold_scores": p_src.fold_scores,
        "chance": p_src.chance,
    }
    res["probe_cross_domain"] = {"balanced_accuracy": p_x.balanced_accuracy, "chance": p_x.chance}
    plot_confusion(p_src.confusion, figs / "probe_confusion_source.png", "source probe (episode-grouped CV)")
    plot_confusion(p_x.confusion, figs / "probe_confusion_cross_domain.png", f"train source, test {shift_name}")

    rr = reuse_score(ls, lt, seed=seed)
    res["reuse"] = {"score": rr.score, "ci95": list(rr.ci), "per_phase": rr.per_phase}
    if rr.per_phase:
        rows = pd.DataFrame([{"model": mcfg.kind, "phase": p, **v} for p, v in rr.per_phase.items()])
        plot_cka_by_phase(rows, figs / "cka_by_phase.png", f"source vs {shift_name}")

    plot_pca_trajectories(ls, figs / "pca_source.png", title="source")
    plot_pca_trajectories(lt, figs / "pca_shifted.png", title=shift_name)
    before = 5
    al = {name: align(lat, "first_contact", before, 15)[0] for name, lat in (("source", ls), (shift_name, lt))}
    plot_event_aligned(al, before, figs / "event_aligned_first_contact.png", "latent change around first contact")

    if mcfg.kind != "reactive" and ablation_episodes > 0:
        acts = src.arrays["expert_action"]
        k = max(1, int(round(ablation_fraction * ls.z.shape[1])))
        sets = dimension_sets(ls.z, ls.mode, acts, k, n_random=3, seed=seed)
        means = ls.z.mean(0)
        rows = []
        for dom in ("source", shift_name):
            for name, dims in [("none", np.array([], int)), *sets.items()]:
                fn = clamp_fn(dims, means) if len(dims) else None
                agent = PolicyAgent(model, norm, latent_fn=fn)
                for r in run_episodes(agent, ablation_episodes, domain=dom, seed=eval_seed):
                    r["ablation"] = "random" if name.startswith("random") else name
                    r["ablation_draw"] = name
                    rows.append(r)
        abl = pd.DataFrame(rows)
        abl.to_csv(out / "latent_ablation_episodes.csv", index=False)
        s = summarize(abl, ["ablation", "domain"], ["success", "peak_force", "force_violation", "jam_or_failure"])
        s.to_csv(out / "latent_ablation_results.csv", index=False)
        res["latent_ablation"] = {"k": k, "dims": {n: d.tolist() for n, d in sets.items()}}
        for m in ("success", "peak_force"):
            plot_condition_metric(
                s,
                m,
                "domain",
                "ablation",
                figs / f"latent_ablation_{m}.png",
                f"latent feature ablation ({k} of {ls.z.shape[1]} dims clamped to mean)",
                m,
                hue_order=["none", "contact", "movement", "random"],
            )
    save_json(res, out / "representation_analysis.json")
    return res
