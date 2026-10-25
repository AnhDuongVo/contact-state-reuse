"""Representation analysis for one checkpoint on a paired (source, shifted) dataset pair.

The two datasets must be generated with the same seed so episodes are paired:
    python scripts/generate_dataset.py --domain source --episodes 60 --seed 777 --p-perturbed 0 --name an_source
    python scripts/generate_dataset.py --domain tight_clearance --episodes 60 --seed 777 --p-perturbed 0 --name an_tight
    python scripts/analyze_representations.py --checkpoint outputs/train/world_model_seed0.pt \
        --source data/an_source.npz --shifted data/an_tight.npz
"""

from __future__ import annotations

import argparse
import json

from contact_state_reuse.analysis.pipeline import analyze_representations
from contact_state_reuse.utils import make_run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--shifted", required=True)
    p.add_argument("--ablation-episodes", type=int, default=30)
    p.add_argument("--ablation-fraction", type=float, default=0.25)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run = make_run_dir(a.out, "representation_analysis")
    res = analyze_representations(a.checkpoint, a.source, a.shifted, run, a.ablation_episodes, a.ablation_fraction)
    print(json.dumps({k: res[k] for k in ("probe_source", "probe_cross_domain")}, indent=1))
    print(f"reuse score {res['reuse']['score']:.3f}  CI {res['reuse']['ci95']}")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
