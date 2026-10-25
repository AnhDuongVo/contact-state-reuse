"""YAML configuration loading.

Configs are plain YAML files. An experiment file may contain ``quick``,
``medium`` and ``full`` sections that are deep-merged over its base keys, so
one file documents every scale. Values are converted to typed dataclasses at the point
of use (``EnvConfig``, ``ModelConfig``, ``TrainConfig`` ...), which validates
field names and fails loudly on typos.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")
SCALES = ("quick", "medium", "full")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, scale: str | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally applying its ``quick`` or ``full`` section.

    Args:
        path: YAML file.
        scale: ``None``, ``"quick"``, ``"medium"`` or ``"full"``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    sections = {k: cfg.pop(k) for k in SCALES if k in cfg}
    if scale is not None:
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
        cfg = deep_merge(cfg, sections.get(scale, {}))
    cfg["_scale"] = scale or "base"
    cfg["_source"] = str(path)
    return cfg


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build dataclass ``cls`` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} fields: {sorted(unknown)}; valid: {sorted(names)}")
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            if isinstance(val, list) and isinstance(f.default, tuple):
                val = tuple(val)
            kwargs[f.name] = val
    return cls(**kwargs)
