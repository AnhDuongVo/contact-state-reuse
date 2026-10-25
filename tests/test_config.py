from pathlib import Path

import pytest
import yaml

from contact_state_reuse.envs import EnvConfig
from contact_state_reuse.models import ModelConfig
from contact_state_reuse.utils import from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_env_yaml_matches_defaults():
    cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text())
    defaults = EnvConfig().to_dict()
    for k, v in cfg.items():
        assert list(defaults[k]) == v if isinstance(v, list) else defaults[k] == v, k


@pytest.mark.parametrize("name", ["A_modality", "B_objective", "C_force_timing", "D_reuse"])
@pytest.mark.parametrize("scale", ["quick", "full"])
def test_experiment_configs_parse(name, scale):
    cfg = load_config(ROOT / "configs" / "experiments" / f"{name}.yaml", scale)
    for cond in cfg["conditions"]:
        merged = {**cfg.get("model_defaults", {}), **cond["model"]}
        if "modalities" in merged:
            merged["modalities"] = tuple(merged["modalities"])
        from_dict(ModelConfig, merged)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        from_dict(ModelConfig, {"kind": "sequence", "hiden": 3})
