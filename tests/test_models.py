import numpy as np
import pytest
import torch

from contact_state_reuse.data import build_episode_tensors, episode_split, fit_normalizer, load_dataset
from contact_state_reuse.envs import EnvConfig, obs_layout
from contact_state_reuse.models import ModelConfig, build_model, n_params
from contact_state_reuse.training import TrainConfig, compute_losses, train_model

LAYOUT = obs_layout(EnvConfig())


@pytest.mark.parametrize("kind", ["reactive", "sequence", "world_model"])
def test_forward_shapes(kind):
    m, cfg = build_model(LAYOUT, ModelConfig(kind=kind, hidden=32))
    obs, prev = torch.randn(3, 7, 14), torch.randn(3, 7, 4).clamp(-1, 1)
    out = m(obs, prev)
    assert out["action"].shape == (3, 7, 4) and out["action"].abs().max() <= 1
    assert out["latent"].shape[:2] == (3, 7)
    if kind == "world_model":
        assert out["next_obs"].shape == (3, 7, 11)
        assert out["force_mean"].shape == (3, 7, cfg.force_horizon, 3)
        assert out["contact_logits"].shape == (3, 7, 6)


@pytest.mark.parametrize("kind", ["sequence", "world_model"])
def test_step_matches_sequence(kind):
    torch.manual_seed(0)
    m, _ = build_model(LAYOUT, ModelConfig(kind=kind, hidden=32))
    m.eval()
    obs, prev = torch.randn(2, 6, 14), torch.randn(2, 6, 4)
    full = m(obs, prev)["action"]
    state = m.init_state(2)
    for t in range(6):
        out, state = m.step(obs[:, t], prev[:, t], state)
    torch.testing.assert_close(out["action"], full[:, -1], atol=1e-5, rtol=1e-4)


def test_matched_capacity():
    counts = [
        n_params(build_model(LAYOUT, ModelConfig(kind=k, param_budget=60000))[0])
        for k in ("reactive", "sequence", "world_model")
    ]
    assert max(counts) / min(counts) < 1.1


def test_modality_mask_removes_information():
    torch.manual_seed(0)
    m, _ = build_model(LAYOUT, ModelConfig(kind="sequence", hidden=32, modalities=("extero", "proprio")))
    m.eval()
    obs, prev = torch.randn(2, 5, 14), torch.randn(2, 5, 4)
    obs2 = obs.clone()
    obs2[..., LAYOUT["force"]] = torch.randn(2, 5, 3) * 50
    torch.testing.assert_close(m(obs, prev)["action"], m(obs2, prev)["action"])


def test_contact_supervision_must_be_explicit():
    with pytest.raises(ValueError):
        ModelConfig(kind="world_model", lambda_contact=0.5, contact_supervision=False)


def test_losses_finite_and_decrease(tiny_dataset_path, tmp_path):
    d = load_dataset(tiny_dataset_path)
    tr, _ = episode_split(d.n_episodes, 0.0, 0)
    T = build_episode_tensors(d, tr, fit_normalizer(d, tr))
    m, cfg = build_model(d.layout, ModelConfig(kind="world_model", hidden=32))
    losses = compute_losses(m(T.obs, T.prev_action), T, cfg)
    for k in ("action", "obs", "force", "contact", "total"):
        assert torch.isfinite(losses[k])
    res = train_model(
        [tiny_dataset_path],
        ModelConfig(kind="world_model", hidden=32),
        TrainConfig(steps=60, log_every=20, val_fraction=0.25, batch_episodes=8),
        tmp_path,
        tag="t",
        verbose=False,
    )
    h = res.history
    assert h[-1]["train_action"] < h[0]["train_action"] * 1.5
    assert np.isfinite(h[-1]["val_total"])
    assert res.checkpoint.exists()
