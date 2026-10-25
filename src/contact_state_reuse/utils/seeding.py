"""Seeding helpers.

Every random draw in this package goes through a ``numpy.random.Generator`` or a
``torch.Generator`` created from an explicit seed. There is no reliance on
global NumPy state. ``seed_everything`` additionally seeds Python, NumPy and
PyTorch globals for third-party code (for example scikit-learn defaults and
dropout) and turns on deterministic PyTorch kernels where available.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, enable ``torch.use_deterministic_algorithms``
            (warn-only, so unsupported ops degrade to a warning rather than an
            error) and disable cuDNN autotuning.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - seeds third-party code using the global RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_seed(base: int, *keys: int | str) -> int:
    """Derive a reproducible child seed from a base seed and a sequence of keys.

    Uses NumPy's ``SeedSequence`` so that child seeds are statistically
    independent and stable across platforms.
    """
    entropy = [int(base)]
    for k in keys:
        if isinstance(k, str):
            entropy.append(int.from_bytes(k.encode("utf-8")[:8].ljust(8, b"\0"), "little"))
        else:
            entropy.append(int(k))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy generator from an explicit seed."""
    return np.random.default_rng(seed)
