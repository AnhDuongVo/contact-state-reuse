import numpy as np

from contact_state_reuse.analysis import LatentSet, grouped_cv_probe, linear_cka, reuse_score
from contact_state_reuse.analysis.similarity import cka_blocks, cka_null


def test_cka_identity_and_invariances():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 10))
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert abs(linear_cka(x, x) - 1) < 1e-9
    assert abs(linear_cka(x, 3.0 * x @ q) - 1) < 1e-9  # rotation + isotropic scale
    assert linear_cka(x, rng.normal(size=(200, 10))) < 0.2  # independent data


def test_cka_null_below_matched():
    rng = np.random.default_rng(1)
    xb = rng.normal(size=(30, 4, 8))
    yb = xb + 0.1 * rng.normal(size=xb.shape)
    assert cka_blocks(xb, yb) > 0.9
    assert cka_null(xb, yb, n_perm=20).mean() < 0.5


def _latset(z, ep, mode):
    n = len(z)
    return LatentSet(
        z=z,
        episode=ep,
        t=np.zeros(n),
        mode=mode,
        force_norm=np.zeros(n),
        success=np.ones(n, bool),
        domain=np.array(["x"] * n),
        t_from_first_contact=np.zeros(n),
        t_to_end=np.zeros(n),
    )


def test_reuse_score_controlled():
    rng = np.random.default_rng(2)
    E, T = 30, 8
    ep = np.repeat(np.arange(E), T)
    mode = np.tile(np.repeat([0, 3], T // 2), E)
    base = rng.normal(size=(E * T, 6))
    src = _latset(base, ep, mode)
    same = reuse_score(src, _latset(base + 0.01 * rng.normal(size=base.shape), ep, mode), n_perm=20, n_boot=20)
    unrelated = reuse_score(src, _latset(rng.normal(size=base.shape), ep, mode), n_perm=20, n_boot=20)
    assert same.score > 0.9
    assert abs(unrelated.score) < 0.2


def test_grouped_probe_beats_chance_on_separable_data():
    rng = np.random.default_rng(3)
    ep = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 3, len(ep))
    z = np.eye(3)[y] * 3 + rng.normal(size=(len(ep), 3))
    res = grouped_cv_probe(z, y, ep)
    assert res.balanced_accuracy > 0.8 > res.chance
