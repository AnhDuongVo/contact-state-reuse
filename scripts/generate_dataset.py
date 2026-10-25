"""Generate a trajectory dataset with the scripted oracle.

Examples:
    python scripts/generate_dataset.py --domain source --episodes 200 --seed 0
    python scripts/generate_dataset.py --domain tight_clearance --severity 0.5 --episodes 50
    python scripts/generate_dataset.py --domain mixed --episodes 300 --out data/mixture
"""

from __future__ import annotations

import argparse

from contact_state_reuse.data import DatasetConfig, generate_dataset
from contact_state_reuse.envs import DOMAINS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="source", choices=DOMAINS)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--severity", type=float, default=1.0, help="shift severity in [0, 1]")
    p.add_argument("--p-perturbed", type=float, default=0.2, help="fraction of perturbed (often failing) demos")
    p.add_argument("--dart-sigma", type=float, default=0.1, help="std of action noise during collection")
    p.add_argument("--name", default=None, help="file stem (default: <domain>_n<episodes>_s<seed>)")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    name = a.name or f"{a.domain}_n{a.episodes}_s{a.seed}"
    generate_dataset(
        DatasetConfig(
            name=name,
            domain=a.domain,
            severity=a.severity,
            n_episodes=a.episodes,
            seed=a.seed,
            p_perturbed=a.p_perturbed,
            dart_sigma=a.dart_sigma,
            out_dir=a.out,
        )
    )


if __name__ == "__main__":
    main()
