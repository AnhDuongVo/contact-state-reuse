"""Small shared utilities (seeding, I/O, configuration)."""

from .config import deep_merge, from_dict, load_config
from .io import get_device, git_version, load_json, make_run_dir, save_json, timestamp
from .seeding import derive_seed, make_rng, seed_everything

__all__ = [
    "deep_merge",
    "derive_seed",
    "from_dict",
    "get_device",
    "git_version",
    "load_config",
    "load_json",
    "make_rng",
    "make_run_dir",
    "save_json",
    "seed_everything",
    "timestamp",
]
