"""Re-draw the standard figures of a finished run from its CSV files.

python scripts/make_figures.py outputs/A_modality_quick/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from contact_state_reuse.analysis.plotting import plot_condition_metric, plot_curves, plot_xy_scatter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    a = p.parse_args()
    run = Path(a.run_dir)
    if not (run / "results.csv").exists():
        raise SystemExit(f"{run} has no results.csv; is it a finished run directory?")
    figs = run / "figures"
    s = pd.read_csv(run / "results.csv")
    if {"condition", "domain"} <= set(s.columns):
        for corr in s.get("corruption", pd.Series(["none"])).unique():
            sub = s[s.corruption == corr] if "corruption" in s else s
            for m in ("success", "peak_force", "force_violation", "jam_or_failure"):
                plot_condition_metric(
                    sub, m, "domain", "condition", figs / f"{m}_{corr}.png", f"{m} ({corr}); mean, 95% bootstrap CI", m
                )
    if (run / "adaptation_curves.csv").exists():
        c = pd.read_csv(run / "adaptation_curves.csv")
        for sev in sorted(c.severity.unique()):
            plot_curves(
                c[c.severity == sev],
                "budget",
                "success",
                "model",
                figs / f"adaptation_curve_sev{sev:g}.png",
                f"adaptation (severity {sev:g})",
                "success rate",
            )
    if (run / "reuse_points.csv").exists():
        pts = pd.read_csv(run / "reuse_points.csv")
        corr = json.loads((run / "reuse_correlations.json").read_text())["reuse_vs_adaptation_auc"]
        plot_xy_scatter(
            pts,
            "reuse",
            "adaptation_auc",
            "model",
            figs / "reuse_vs_adaptation.png",
            "reuse score vs adaptation efficiency",
            f"Spearman rho={corr['rho']:.2f} [{corr['ci_low']:.2f}, {corr['ci_high']:.2f}], n={corr['n']}",
        )
    print(f"figures written to {figs}")


if __name__ == "__main__":
    main()
