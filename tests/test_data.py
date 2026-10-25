import numpy as np
import pytest

from contact_state_reuse.data import (
    SchemaError,
    build_episode_tensors,
    episode_split,
    fit_normalizer,
    load_dataset,
    validate,
)


def test_roundtrip_and_schema(tiny_dataset_path):
    d = load_dataset(tiny_dataset_path)
    a = d.arrays
    assert d.n_episodes == 12
    assert a["obs"].shape[1] == 14
    assert d.meta["dataset"]["seed"] == 3
    assert "code_version" in d.meta and "env_config" in d.meta
    assert set(np.unique(a["contact_mode"])) <= set(range(6))
    validate(a)


def test_validation_catches_nan(tiny_dataset_path):
    a = dict(load_dataset(tiny_dataset_path).arrays)
    a["obs"] = a["obs"].copy()
    a["obs"][3, 0] = np.nan
    with pytest.raises(SchemaError):
        validate(a)


def test_validation_catches_bad_label(tiny_dataset_path):
    a = dict(load_dataset(tiny_dataset_path).arrays)
    a["contact_mode"] = a["contact_mode"].copy()
    a["contact_mode"][0] = 9
    with pytest.raises(SchemaError):
        validate(a)


def test_episode_split_no_leakage():
    for seed in range(5):
        tr, va = episode_split(50, 0.2, seed)
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == 50 and len(va) == 10


def test_paired_seeds_across_domains(tiny_data_dir):
    s, t = load_dataset(tiny_data_dir / "src.npz"), load_dataset(tiny_data_dir / "tight.npz")
    ps, pt = s.meta["episode_params"][0], t.meta["episode_params"][0]
    assert ps["slot_x"] == pt["slot_x"] and ps["friction"] == pt["friction"]
    assert pt["clearance"] < ps["clearance"]


def test_episode_tensors(tiny_dataset_path):
    d = load_dataset(tiny_dataset_path)
    eps = np.arange(4)
    T = build_episode_tensors(d, eps, fit_normalizer(d, eps))
    lengths = d.arrays["ep_length"][eps] + 1
    assert T.valid.sum().item() == lengths.sum()
    assert T.obs.shape[:2] == (4, lengths.max())
    # previous action at t is the executed action at t-1
    r = d.episode_rows(0)
    np.testing.assert_allclose(T.prev_action[0, 1].numpy(), d.arrays["action"][r.start])
    # no next-observation target on the final row
    assert T.next_mask[0, lengths[0] - 1] == 0
