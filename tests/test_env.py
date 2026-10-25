import numpy as np
import pytest

from contact_state_reuse.envs import DOMAINS, ContactInsertionEnv, ContactMode, EnvConfig, obs_layout, sample_params
from contact_state_reuse.envs.expert import ScriptedExpert
from contact_state_reuse.envs.labels import label_step


def rollout(env, seed, policy, n=200):
    obs, info = env.reset(seed=seed)
    out = [obs]
    for _ in range(n):
        obs, r, term, trunc, info = env.step(policy(env))
        out.append(obs)
        if term or trunc:
            break
    return np.stack(out), info


def test_spaces_and_finite():
    env = ContactInsertionEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (14,) and obs.dtype == np.float32
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert isinstance(term, bool) and isinstance(trunc, bool)
    for k in ("contact_mode", "first_contact", "success", "jam", "insertion_depth", "contact_force_norm"):
        assert k in info


def test_labels_are_not_in_observation():
    env = ContactInsertionEnv()
    lay = obs_layout(env.cfg)
    assert set(lay) == {"extero", "proprio", "force"}
    assert lay["force"].stop == env.observation_space.shape[0]


def test_seed_reproducible():
    env = ContactInsertionEnv()
    a, _ = rollout(env, 5, lambda e: np.array([0.1, -0.5, 0.0, 0.0]), 40)
    b, _ = rollout(env, 5, lambda e: np.array([0.1, -0.5, 0.0, 0.0]), 40)
    np.testing.assert_array_equal(a, b)
    c, _ = rollout(env, 6, lambda e: np.array([0.1, -0.5, 0.0, 0.0]), 40)
    assert not np.array_equal(a[:5], c[:5])


def test_truncation():
    env = ContactInsertionEnv(EnvConfig(max_steps=10))
    env.reset(seed=0)
    for _ in range(10):
        _, _, term, trunc, _ = env.step(np.array([0.0, 1.0, 0.0, 0.0]))  # move up, never succeed
    assert trunc and not term


def test_expert_succeeds_and_terminates():
    env, ex = ContactInsertionEnv(), ScriptedExpert()
    wins = 0
    for s in range(4):
        env.reset(seed=s)
        ex.reset(env, np.random.default_rng(s), "nominal")
        done = False
        while not done:
            _, _, term, trunc, info = env.step(ex.act(env))
            done = term or trunc
        wins += info["success"]
        assert info["contact_mode"] in (ContactMode.SUCCESS, ContactMode.JAM_OR_FAILURE) or trunc
    assert wins >= 3


def test_large_aim_variant_fails():
    env, ex = ContactInsertionEnv(), ScriptedExpert()
    env.reset(seed=1)
    ex.reset(env, np.random.default_rng(1), "large_aim")
    done = False
    while not done:
        _, _, term, trunc, info = env.step(ex.act(env))
        done = term or trunc
    assert not info["success"]


def test_contact_force_sign_pressing_down():
    """Pressing the peg onto the slot's top face gives an upward force on the peg."""
    env = ContactInsertionEnv()
    env.reset(seed=0)
    import mujoco

    env.data.qpos[:3] = [env.params.slot_x + 0.03, 0.045, env.params.slot_angle]
    env.setpoint = env.data.qpos[:3].copy()
    mujoco.mj_forward(env.model, env.data)
    for _ in range(8):
        _, _, _, _, info = env.step(np.array([0.0, -1.0, 0.0, 0.0]))
    assert info["wrench_true"][1] > 1.0


@pytest.mark.parametrize("domain", DOMAINS)
def test_domains_sample(domain):
    p, concrete = sample_params(np.random.default_rng(0), domain)
    assert concrete in DOMAINS and concrete != "mixed"
    assert p.clearance > 0


def test_severity_interpolates():
    lo, _ = sample_params(np.random.default_rng(0), "tight_clearance", 0.0)
    hi, _ = sample_params(np.random.default_rng(0), "tight_clearance", 1.0)
    assert hi.clearance < lo.clearance


def test_label_rules():
    kw = dict(first_contact_window=4, depth=0.0, entry_depth=0.0035, in_contact=False)
    assert label_step(success=True, failed=False, steps_since_first_contact=0, **kw) == ContactMode.SUCCESS
    assert label_step(success=False, failed=True, steps_since_first_contact=0, **kw) == ContactMode.JAM_OR_FAILURE
    assert label_step(success=False, failed=False, steps_since_first_contact=None, **kw) == ContactMode.APPROACH
    assert label_step(success=False, failed=False, steps_since_first_contact=2, **kw) == ContactMode.FIRST_CONTACT
    kw.update(in_contact=True)
    assert label_step(success=False, failed=False, steps_since_first_contact=9, **kw) == ContactMode.ALIGNMENT
    kw.update(depth=0.01)
    assert label_step(success=False, failed=False, steps_since_first_contact=9, **kw) == ContactMode.INSERTION
