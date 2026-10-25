"""Train one model.

Examples:
    python scripts/train.py --data data/source_n200_s0.npz --model configs/models/sequence.yaml
    python scripts/train.py --data data/source_n200_s0.npz --model configs/models/world_model.yaml \
        --train configs/train/quick.yaml --out outputs/train --tag wm
    # data mixture: pass several datasets
    python scripts/train.py --data data/source.npz data/mixed.npz --model configs/models/world_model.yaml
"""

from __future__ import annotations

import argparse

from contact_state_reuse.models import ModelConfig
from contact_state_reuse.training import TrainConfig, train_model
from contact_state_reuse.utils import from_dict, load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="one or more dataset .npz files")
    p.add_argument("--model", required=True, help="model YAML (configs/models/*.yaml)")
    p.add_argument("--train", default="configs/train/quick.yaml", help="training YAML")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/train")
    p.add_argument("--tag", default=None)
    p.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    a = p.parse_args()
    mc = load_config(a.model)
    mc = {k: v for k, v in mc.items() if not k.startswith("_")}
    if "modalities" in mc:
        mc["modalities"] = tuple(mc["modalities"])
    tc = {k: v for k, v in load_config(a.train).items() if not k.startswith("_")}
    tc["seed"] = a.seed
    mcfg, tcfg = from_dict(ModelConfig, mc), from_dict(TrainConfig, tc)
    tag = a.tag or f"{mcfg.kind}_seed{a.seed}"
    res = train_model(a.data, mcfg, tcfg, a.out, init_checkpoint=a.init, tag=tag)
    print(f"saved {res.checkpoint} ({res.n_params} parameters, hidden={res.model_config.hidden})")


if __name__ == "__main__":
    main()
