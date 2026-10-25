"""Sensor models applied on top of noise-free simulator quantities.

ForceSensor
    Input: the planar contact wrench (Fx, Fz, Ty) acting on the peg, computed
    from MuJoCo contact forces (see ``ContactInsertionEnv.contact_wrench``).
    Output: delayed, biased, noisy reading with random dropout (zeros).

Exteroception
    A stand-in for a camera-based pose estimator. Input: true relative target
    pose (slot entrance minus peg tip, in world x/z, and slot angle minus peg
    angle). Output: the same plus a fixed per-episode calibration offset and
    per-frame noise. Near the slot the peg occludes the view: with probability
    ``occlusion`` a frame is dropped, the last estimate is held and the
    validity flag is 0.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .domains import EpisodeParams


class ForceSensor:
    """Corrupts the true planar wrench according to ``EpisodeParams``."""

    TORQUE_SCALE = 0.01  # torque noise/bias in N m per N of force noise/bias

    def __init__(self, params: EpisodeParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        scale = np.array([1.0, 1.0, self.TORQUE_SCALE])
        self.bias = direction * params.force_bias * scale
        self.noise_std = params.force_noise * scale
        self.delay = int(round(params.force_delay))
        self.buffer: deque[np.ndarray] = deque(maxlen=self.delay + 1)

    def __call__(self, wrench: np.ndarray) -> np.ndarray:
        self.buffer.append(np.asarray(wrench, dtype=np.float64).copy())
        delayed = self.buffer[0]
        reading = delayed + self.bias + self.rng.normal(size=3) * self.noise_std
        if self.p.force_dropout > 0 and self.rng.random() < self.p.force_dropout:
            reading = np.zeros(3)
        return reading


class Exteroception:
    """Noisy, occludable estimate of the slot pose relative to the peg."""

    OCCLUSION_RADIUS = 0.03  # m, horizontal distance at which the peg occludes the slot
    OCCLUSION_HEIGHT = 0.03  # m, above the slot entrance

    def __init__(self, params: EpisodeParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng
        d = rng.normal(size=2)
        d /= np.linalg.norm(d) + 1e-12
        self.offset = np.array([*(d * params.extero_bias), 0.0])
        self.last: np.ndarray | None = None

    def __call__(self, rel_true: np.ndarray) -> tuple[np.ndarray, float]:
        near = abs(rel_true[0]) < self.OCCLUSION_RADIUS and -rel_true[1] < self.OCCLUSION_HEIGHT
        if near and self.last is not None and self.rng.random() < self.p.occlusion:
            return self.last.copy(), 0.0
        noise = self.rng.normal(size=3) * np.array([self.p.extero_noise, self.p.extero_noise, 0.01])
        est = rel_true + self.offset + noise
        self.last = est
        return est.copy(), 1.0
