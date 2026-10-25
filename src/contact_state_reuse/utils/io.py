"""File-system helpers: run directories, JSON, git version, device selection."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return the compute device.

    ``auto`` selects CUDA when available, then Apple MPS, otherwise CPU.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def git_version(root: str | Path | None = None) -> str:
    """Short git commit hash of the working tree, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(root: str | Path, experiment: str, stamp: str | None = None) -> Path:
    """Create ``<root>/<experiment>/<timestamp>/`` with ``figures`` and ``checkpoints``."""
    run = Path(root) / experiment / (stamp or timestamp())
    (run / "figures").mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2, allow_nan=True)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
