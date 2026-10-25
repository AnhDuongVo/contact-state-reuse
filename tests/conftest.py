"""Shared fixtures: a tiny dataset generated once per test session."""

from __future__ import annotations

import pytest
import torch

from contact_state_reuse.data import DatasetConfig, generate_dataset

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate_dataset(DatasetConfig(name="src", domain="source", n_episodes=12, seed=3, out_dir=str(d)), verbose=False)
    generate_dataset(
        DatasetConfig(name="tight", domain="tight_clearance", n_episodes=12, seed=3, p_perturbed=0.2, out_dir=str(d)),
        verbose=False,
    )
    return d


@pytest.fixture(scope="session")
def tiny_dataset_path(tiny_data_dir):
    return tiny_data_dir / "src.npz"
