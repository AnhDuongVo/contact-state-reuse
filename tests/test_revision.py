"""Tests for the features added in v0.2 (revision)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from contact_state_reuse.analysis.latents import extract_latents, reference_latents
from contact_state_reuse.analysis.probes import paired_heldout_probe
from contact_state_reuse.analysis.reuse import reuse_score
from contact_state_reuse.data import load_dataset
from contact_state_reuse.data.dataset import Normalizer, build_episode_tensors, transform_force
from contact_state_reuse.envs import ContactInsertionEnv
from contact_state_reuse.envs.domains import CORE_DOMAINS, EpisodeParams, domain_ranges, sample_params
from contact_state_reuse.envs.expert import ExpertConfig
from contact_state_reuse.envs.insertion_env import EnvConfig, obs_layout
from contact_state_reuse.evaluation.metrics import crossed_bootstrap, paired_crossed, seed_episode_summary
from contact_state_reuse.evaluation.rollout import ForceCorruption, _CorruptionState, run_episodes
from contact_state_reuse.models import ModelConfig, build_model
from contact_state_reuse.training.losses import compute_losses
from contact_state_reuse.training.trainer import TrainConfig, add_force_corruption, prepare_data


# ------------------------------------------------------------------ domains
def test_composite_domain_combines_shifts():
    r = domain_ranges("search+tight_clearance")
    assert r["search_offset"] == domain_ranges("search")["search_offset"]
    assert r["clearance"] == domain_ranges("tight_clearance")["clearance"]
    assert domain_ranges("source")["search_offset"] == (0.0, 0.0)
    with pytest.raises(ValueError):
        domain_ranges("search+nope")


def test_mixed_samples_only_core_domains():
    rng = np.random.default_rng(0)
    names = {sample_params(rng, "mixed")[1] for _ in range(200)}
    assert names <= set(CORE_DOMAINS)


def test_new_params_do_not_change_existing_random_streams():
    # fields added in v0.2 are fixed in the source domain, so no extra draws happen
    a = sample_params(np.random.default_rng(5), "tight_clearance")[0]
    assert a.chamfer == EpisodeParams().chamfer and a.search_offset == 0.0


# ------------------------------------------------------------------- expert
def test_non_privileged_expert_never_reads_privileged_state(monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("privileged state used")

    monkeypatch.setattr(ContactInsertionEnv, "privileged_state", fail)
    rows = run_episodes("expert", 3, seed=4, expert_cfg=ExpertConfig(privileged=False, aim_error=0.001))
    assert np.mean([r["success"] for r in rows]) >= 2 / 3


def test_search_expert_solves_search_task():
    rows = run_episodes(
        "expert", 3, domain="search", seed=2, expert_cfg=ExpertConfig(privileged=False, search=True, aim_error=0.001)
    )
    assert np.mean([r["success"] for r in rows]) >= 2 / 3


def test_search_offset_shifts_pose_estimate():
    env = ContactInsertionEnv(domain="search")
    obs, info = env.reset(seed=0)
    assert 0.005 <= info["params"]["search_offset"] <= 0.009
    assert abs(env.extero.offset[0]) >= 0.005 - 0.0025


# --------------------------------------------------------------- transforms
def test_force_transforms_and_normalizer_roundtrip():
    x = np.array([[0.0, 0.0, 3.0, -4.0, 0.02], [0.0, 0.0, 0.1, 0.2, 0.0]], np.float32)
    lg = transform_force(x, (2, 5), "log")
    assert np.allclose(lg[:, :2], 0) and np.isclose(lg[0, 2], np.log1p(3.0)) and lg[0, 3] < 0
    bn = transform_force(x, (2, 5), "binary")
    assert bn[0, 2] == 1.0 and bn[1, 2] == 0.0 and np.all(bn[:, 3:] == 0)
    n = Normalizer.fit(x, "log", (2, 5))
    n2 = Normalizer.from_state_dict(n.state_dict())
    assert np.allclose(n(x), n2(x)) and n2.force_transform == "log"


# --------------------------------------------------------------- corruptions
def _run(spec, contact_at=3, steps=20):
    cs = _CorruptionState(spec, np.random.default_rng(0), slice(0, 3))
    out = [cs.apply(np.full(3, 2.0, np.float32), t, t == contact_at)[0] for t in range(steps)]
    return np.asarray(out), cs


def test_onset_offset_window():
    out, cs = _run(ForceCorruption("onset_offset_window", window=4, offset=6))
    assert np.all(out[9:13] == 0) and np.all(out[:9] == 2) and np.all(out[13:] == 2)
    assert cs.masked_steps == 4 and np.isclose(cs.masked_force, 4 * np.hypot(2, 2))


def test_onset_only_and_hold():
    out, _ = _run(ForceCorruption("onset_only", window=4))
    assert np.all(out[3:7] == 2) and np.all(out[:3] == 0) and np.all(out[7:] == 0)
    cs = _CorruptionState(ForceCorruption("hold_onset", window=3), np.random.default_rng(0), slice(0, 3))
    vals = [cs.apply(np.full(3, float(t), np.float32), t, t == 5)[0] for t in range(10)]
    assert vals[5:8] == [4.0, 4.0, 4.0] and vals[8] == 8.0


def test_scale_and_unreached_window():
    out, _ = _run(ForceCorruption("scale", magnitude=0.5))
    assert np.all(out == 1.0)
    out, cs = _run(ForceCorruption("onset_offset_window", window=4, offset=30))
    assert cs.masked_steps == 0 and np.all(out == 2)


# -------------------------------------------------------------------- stats
def test_crossed_bootstrap_and_paired():
    rng = np.random.default_rng(0)
    m = rng.random((5, 100))
    mean, lo, hi = crossed_bootstrap(m)
    assert lo < mean < hi
    rows = []
    for s in range(4):
        for e in range(30):
            rows += [
                {"seed": s, "episode": e, "c": "a", "success": 1.0},
                {"seed": s, "episode": e, "c": "b", "success": 0.0},
            ]
    df = pd.DataFrame(rows)
    r = paired_crossed(df, "c", "a", "b", "success")
    assert r["diff"] == 1.0 and r["seeds_a_better"] == 4
    s = seed_episode_summary(df[df.c == "a"], "success")
    assert s["n_seeds"] == 4 and s["per_seed"] == [1.0] * 4


# -------------------------------------------------------------------- models
LAYOUT = obs_layout(EnvConfig())


def test_no_prev_action_ignores_prev_action():
    m, _ = build_model(LAYOUT, ModelConfig(kind="world_model", hidden=24, use_prev_action=False))
    x = torch.randn(2, 5, 14)
    a1 = m(x, torch.randn(2, 5, 4))["action"]
    a2 = m(x, torch.randn(2, 5, 4))["action"]
    assert torch.allclose(a1, a2)


@pytest.mark.parametrize("opt", [{"aux_core": True}, {"aux_stopgrad": True}, {"lambda_recon": 1.0}])
def test_aux_variants_step_matches_forward(opt):
    m, _ = build_model(LAYOUT, ModelConfig(kind="world_model", hidden=24, **opt))
    x, pa = torch.randn(2, 6, 14), torch.randn(2, 6, 4)
    full = m(x, pa)["action"]
    st, outs = m.init_state(2), []
    for t in range(6):
        o, st = m.step(x[:, t], pa[:, t], st)
        outs.append(o["action"])
    assert torch.allclose(torch.stack(outs, 1), full, atol=1e-5)


def test_stopgrad_blocks_trunk_gradient(tiny_dataset_path):
    d = load_dataset(tiny_dataset_path)
    cfg = ModelConfig(kind="world_model", hidden=24, lambda_obs=1.0, lambda_force=1.0, aux_stopgrad=True)
    m, _ = build_model(d.layout, cfg)
    t = build_episode_tensors(d, np.arange(3), Normalizer.fit(d.arrays["obs"]))
    losses = compute_losses(m(t.obs, t.prev_action), t, cfg)
    g = torch.autograd.grad(losses["obs"] + losses["force"], list(m.core.parameters()), allow_unused=True)
    assert all(x is None or torch.all(x == 0) for x in g)


# ------------------------------------------------------------------ training
def test_success_filter_and_force_augmentation(tiny_dataset_path):
    d = load_dataset(tiny_dataset_path)
    mcfg = ModelConfig(kind="world_model", hidden=24)
    train, _, norm, splits = prepare_data([d], TrainConfig(episode_filter="success", val_fraction=0.2), mcfg)
    assert all(d.arrays["ep_success"][e] for e in splits[0][0])
    aug = add_force_corruption(train, norm, d.layout["force"], 1.0, 0)
    n = len(train)
    assert len(aug) == 2 * n
    assert torch.equal(aug.future_force[n:], train.future_force)  # targets keep the original readings
    fs = d.layout["force"]
    assert not torch.allclose(aug.obs[n:, :, fs], train.obs[:, :, fs])
    assert torch.equal(aug.obs[n:, :, : fs.start], train.obs[:, :, : fs.start])


# ------------------------------------------------------------------ analysis
def test_reference_latents_and_paired_probe(tiny_data_dir):
    src, tgt = load_dataset(tiny_data_dir / "src.npz"), load_dataset(tiny_data_dir / "tight.npz")
    norm = Normalizer.fit(src.arrays["obs"])
    ls, lt = reference_latents(src, "obs", norm), reference_latents(tgt, "obs", norm)
    assert ls.z.shape == (len(src.arrays["t"]), 14)
    assert reference_latents(src, "progress").z.shape[1] == 2
    r = paired_heldout_probe(ls.z, ls.mode, ls.episode, lt.z, lt.mode, lt.episode, n_splits=3)
    assert 0 <= r.source_bacc <= 1 and 0 <= r.shifted_bacc <= 1
    assert set(r.support_source) == set(r.support_shifted)


def test_reuse_jackknife_interval_contains_estimate(tiny_data_dir):
    src, tgt = load_dataset(tiny_data_dir / "src.npz"), load_dataset(tiny_data_dir / "tight.npz")
    m, _ = build_model(src.layout, ModelConfig(kind="world_model", hidden=24))
    norm = Normalizer.fit(src.arrays["obs"])
    rr = reuse_score(extract_latents(m, norm, src), extract_latents(m, norm, tgt), n_perm=5, n_boot=1, min_episodes=3)
    assert rr.ci[0] <= rr.score <= rr.ci[1]
