"""Scripted demonstrator (data-generation oracle).

The expert reads *privileged* simulator state through
``ContactInsertionEnv.privileged_state()``: the true slot pose and the
noise-free contact wrench. It is used only to generate demonstrations. Learned
models never see privileged state; they see the noisy observation vector.

Nominal strategy (in the slot frame):
1. approach a point 12 mm above the slot entrance, offset laterally by a
   sampled aim error (so first contact usually happens on the chamfer);
2. descend along the slot axis until contact;
3. comply laterally with the measured lateral contact force (admittance),
   regulate the axial force around 5 N, correct the angle, and use a soft
   compliance setting until the goal depth is reached.

Non-privileged mode (``privileged=False``, added in v0.2): the same state
machine, but it perceives only what a policy observes. The slot entrance is
estimated online from the noisy (biased, occludable) pose estimate, kept in
the world frame and held when a frame is invalid; contact and lateral forces
come from the force-sensor reading. It serves as a simple non-privileged
force-feedback controller (a baseline policy) and as a second demonstrator.

Search mode (``search=True``, second task): after touching down, the expert
first tries the normal insertion behaviour for ``search_wait`` steps. If the
tip has not dropped below the touch-down height by ``search_drop`` (it landed
on the top face), it presses with a light axial force and slides laterally,
first towards +x and reversing after ``search_travel``, until the tip drops
(it found the chamfer or the opening), then inserts normally. The search
direction is deterministic, so the behaviour is a function of the observation
history and can be imitated.

Perturbed variants (sampled with probability ``p_perturbed``) produce
meaningful failures and recoveries for the dataset:
* ``large_aim``   aim error beyond the chamfer, so the peg lands on the top face;
* ``stiff_push``  ignores force, stiff gains, constant downward push;
* ``angle_error`` large angle error and no angle correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .insertion_env import SLOT_DEPTH, ContactInsertionEnv, _rot

VARIANTS = ("nominal", "large_aim", "stiff_push", "angle_error")


@dataclass(frozen=True)
class ExpertConfig:
    aim_error: float = 0.0035  # max lateral aim error for nominal demos [m]
    large_aim: tuple[float, float] = (0.006, 0.009)
    angle_error: float = 0.02  # max angle error for nominal demos [rad]
    large_angle: tuple[float, float] = (0.06, 0.10)
    p_perturbed: float = 0.0  # probability of a perturbed variant
    descend_speed: float = 0.012  # m/s
    insert_speed: float = 0.02  # m/s at zero axial force
    target_force: float = 5.0  # N axial force while inserting
    admittance: float = 0.004  # m/s per N of lateral force
    contact_force: float = 1.0  # N, contact detection threshold
    compliance: float = -0.5  # compliance command while in contact
    privileged: bool = True  # False: perceive only the observation vector
    estimate_gain: float = 0.3  # EMA gain of the non-privileged entrance estimate
    search: bool = False  # second task: slide along the top face to find the opening
    search_speed: float = 0.02  # m/s lateral speed while searching
    search_travel: float = 0.012  # m before reversing direction
    search_force: float = 1.0  # N axial force while searching (below the jam detector threshold)
    search_drop: float = 0.0015  # m below touch-down height that ends the search
    search_wait: int = 10  # control steps of normal insertion before searching


class ScriptedExpert:
    """Privileged state-machine expert. Call ``reset(env, rng)`` after ``env.reset``."""

    def __init__(self, config: ExpertConfig | None = None) -> None:
        self.cfg = config or ExpertConfig()

    def reset(self, env: ContactInsertionEnv, rng: np.random.Generator, variant: str | None = None) -> str:
        c = self.cfg
        if variant is None:
            if rng.random() < c.p_perturbed:
                variant = VARIANTS[1 + int(rng.integers(len(VARIANTS) - 1))]
            else:
                variant = "nominal"
        if variant not in VARIANTS:
            raise ValueError(f"unknown expert variant {variant!r}")
        self.variant = variant
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if variant == "large_aim":
            self.aim = sign * rng.uniform(*c.large_aim)
        else:
            self.aim = rng.uniform(-c.aim_error, c.aim_error)
        if variant == "angle_error":
            self.ang = sign * rng.uniform(*c.large_angle)
        else:
            self.ang = rng.uniform(-c.angle_error, c.angle_error)
        self.mode = "approach"
        self.dt = env.cfg.control_dt
        self.scale = np.asarray(env.cfg.action_scale)
        self.ent_est: np.ndarray | None = None  # non-privileged entrance/angle estimate (world frame)
        self.touch: np.ndarray | None = None  # pose at search touch-down
        self.since_touch: int | None = None
        self.search_dir = 1.0
        return variant

    def _perceive(self, env: ContactInsertionEnv) -> dict:
        """State used by the controller: privileged simulator state, or an
        estimate built only from the observation vector."""
        if self.cfg.privileged:
            return env.privileged_state()
        obs, lay = env.last_obs, env.layout
        ext, pro, frc = obs[lay["extero"]], obs[lay["proprio"]], obs[lay["force"]]
        pose = pro[:3].astype(np.float64)
        if ext[3] > 0.5:  # valid frame: update the world-frame estimate
            meas = np.array([pose[0] + ext[0], pose[1] + ext[1], pose[2] + ext[2]])
            if self.ent_est is None:
                self.ent_est = meas
            else:
                self.ent_est = self.ent_est + self.cfg.estimate_gain * (meas - self.ent_est)
        if self.ent_est is None:
            self.ent_est = np.array([pose[0] + ext[0], pose[1] + ext[1], pose[2] + ext[2]])
        phi = float(self.ent_est[2])
        base = self.ent_est[:2] - _rot(phi) @ np.array([0.0, SLOT_DEPTH])
        lat, height = _rot(-phi) @ (pose[:2] - base)
        return {
            "pose": pose,
            "slot_angle": phi,
            "slot_base": base,
            "wrench": frc.astype(np.float64),
            "lateral": float(lat),
            "depth": float(SLOT_DEPTH - height),
        }

    def act(self, env: ContactInsertionEnv) -> np.ndarray:
        c, s = self.cfg, self._perceive(env)
        phi = s["slot_angle"]
        slot_base = s.get("slot_base", np.array([env.params.slot_x, 0.0]))
        to_world, to_slot = _rot(phi), _rot(-phi)
        pose = s["pose"]
        f_slot = to_slot @ s["wrench"][:2]  # (lateral, axial) force on the peg
        f_mag = float(np.hypot(*s["wrench"][:2]))
        target_angle = phi + self.ang
        v_slot = np.zeros(2)  # desired velocity in the slot frame [m/s]
        compliance = 0.0
        d_theta = 0.0
        if self.mode == "approach":
            goal_local = np.array([self.aim, SLOT_DEPTH + 0.012])
            goal = slot_base + to_world @ goal_local
            err_world = goal - pose[:2]
            v_world = np.clip(3.0 * err_world, -0.08, 0.08)
            v_slot = to_slot @ v_world
            d_theta = 0.5 * (target_angle - pose[2])
            if np.linalg.norm(err_world) < 0.0015:
                self.mode = "descend"
        if self.mode == "descend":
            v_slot = np.array([3.0 * (self.aim - s["lateral"]), -c.descend_speed])
            d_theta = 0.5 * (target_angle - pose[2])
            if f_mag > c.contact_force:
                self.mode = "insert"
                if c.search:
                    self.touch = pose.copy()
                    self.since_touch = 0
        if c.search and self.mode == "insert" and self.touch is not None and self.since_touch is not None:
            self.since_touch += 1
            if (to_slot @ (self.touch[:2] - pose[:2]))[1] > c.search_drop:
                self.since_touch = None  # dropped: keep inserting
            elif self.since_touch > c.search_wait:
                self.mode = "search"
                self.touch = pose.copy()
        if self.mode == "search":
            travel = (to_slot @ (pose[:2] - self.touch[:2]))[0]
            if self.search_dir > 0 and travel > c.search_travel:
                self.search_dir = -1.0
            lateral_v = self.search_dir * c.search_speed + c.admittance * f_slot[0]
            axial_v = -c.insert_speed * np.clip(1.0 - f_slot[1] / c.search_force, -1.0, 1.0)
            v_slot = np.array([lateral_v, axial_v])
            compliance = c.compliance
            d_theta = 0.5 * (phi - pose[2])
            dropped = (to_slot @ (self.touch[:2] - pose[:2]))[1]
            if dropped > c.search_drop:
                self.mode = "insert"
                self.since_touch = None
        if self.mode == "insert":
            if self.variant == "stiff_push":
                v_slot = np.array([0.0, -0.03])  # set-point stays at the aim offset
                compliance = 1.0
                d_theta = 0.0
            else:
                lateral_v = c.admittance * f_slot[0]
                # while still testing whether the peg drops (search mode), press lightly
                target = c.search_force if (c.search and self.since_touch is not None) else c.target_force
                axial_v = -c.insert_speed * np.clip(1.0 - f_slot[1] / target, -1.0, 1.0)
                v_slot = np.array([lateral_v, axial_v])
                compliance = c.compliance
                if self.variant == "angle_error":
                    d_theta = 0.0
                else:
                    d_theta = 0.5 * (phi - pose[2]) + 0.5 * s["wrench"][2]
            if s["depth"] > env.cfg.goal_depth + 0.002:
                v_slot[1] = 0.0
        v_world = to_world @ v_slot
        step = np.array([v_world[0] * self.dt, v_world[1] * self.dt, d_theta])
        action = np.concatenate([step / self.scale, [compliance]])
        return np.clip(action, -1.0, 1.0).astype(np.float32)
