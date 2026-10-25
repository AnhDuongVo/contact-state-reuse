"""Planar peg-in-slot insertion with MuJoCo.

Mechanism
---------
A rectangular peg (20 mm wide, 60 mm long) moves in the x-z plane with three
degrees of freedom: x and z translation of the peg tip and rotation about the
y axis through the tip. Each joint is driven by a MuJoCo actuator acting as a
PD controller around a set-point. The policy moves the set-point (bounded
deltas) and chooses a compliance level that scales the PD stiffness. The slot
is 40 mm deep, has a 3 mm 45-degree chamfer, and can be shifted and tilted.

Control: physics 500 Hz (2 ms), policy 20 Hz (25 physics steps per action).

Observation (flat ``float32`` vector, layout in ``OBS_LAYOUT``)
    extero   [4]  estimated (slot entrance - peg tip) in x, z [m], estimated
                  slot angle - peg angle [rad], estimate-valid flag
    proprio  [7]  joint positions (tip x, tip z, angle), joint velocities,
                  current compliance level in [0, 1]
                  (+3 joint efforts if ``EnvConfig.proprio_include_effort``)
    force    [3]  planar contact wrench on the peg (Fx, Fz [N], Ty [N m])
                  after the sensor model (noise, bias, delay, dropout)

What the force signal is
------------------------
The "force" channel is the sum of MuJoCo contact forces acting on the peg,
converted to the world frame, plus the torque of those forces about the peg
tip. It is therefore an idealised external-wrench measurement at the contact,
not a model of a wrist force/torque sensor (no tool inertia, no sensor
compliance) and not a model of motor-current sensing. MuJoCo contacts are soft
constraints, so magnitudes depend on solver parameters; they are meaningful
for comparing conditions within this simulator, not as absolute newtons on a
real robot.

Actions (``Box(-1, 1, (4,))``)
    [dx, dz, dtheta] scaled by ``EnvConfig.action_scale`` (metres / radians per
    control step) and a compliance command c; PD stiffness = kp * (0.25 + 0.75 *
    (c + 1) / 2).

Termination: success (depth >= goal), jam (no progress while pressing), or
damage (contact force above ``damage_force``). Truncation: ``max_steps``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .domains import EpisodeParams, sample_params
from .labels import label_step
from .sensors import Exteroception, ForceSensor

PEG_HALF_W = 0.010
PEG_HALF_T = 0.015
PEG_HALF_L = 0.030
SLOT_HALF_OUTER = 0.05
SLOT_DEPTH = 0.040
CHAMFER = 0.003


@dataclass(frozen=True)
class EnvConfig:
    """Fixed (non-randomised) task settings."""

    physics_dt: float = 0.002
    control_dt: float = 0.05
    max_steps: int = 160
    action_scale: tuple[float, float, float] = (0.004, 0.004, 0.03)
    windup: tuple[float, float, float] = (0.012, 0.012, 0.15)
    kp_lin: float = 2000.0
    kp_rot: float = 6.0
    damping_ratio: float = 0.8
    goal_depth: float = 0.025
    safe_force: float = 10.0  # force-limit for the violation metric [N]
    damage_force: float = 40.0  # terminate as failure above this [N]
    jam_window: int = 30  # steps (1.5 s)
    jam_progress: float = 0.0005  # m of depth gain required within the window
    jam_min_force: float = 2.0  # N mean force during the window
    first_contact_window: int = 4  # steps labelled FIRST_CONTACT after onset
    contact_threshold: float = 0.5  # N, true force that counts as contact
    proprio_include_effort: bool = False
    w_progress: float = 10.0
    w_success: float = 10.0
    w_force: float = 0.5
    w_jam: float = 5.0
    w_time: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def obs_layout(cfg: EnvConfig) -> dict[str, slice]:
    n_pro = 10 if cfg.proprio_include_effort else 7
    return {"extero": slice(0, 4), "proprio": slice(4, 4 + n_pro), "force": slice(4 + n_pro, 7 + n_pro)}


def _rot(phi: float) -> np.ndarray:
    """Local-to-world map on (x, z) for a body rotated by ``phi`` about +y."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, s], [-s, c]])


class ContactInsertionEnv(gym.Env):
    """Gymnasium environment for planar peg-in-slot insertion (see module docstring)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: EnvConfig | None = None,
        domain: str = "source",
        severity: float = 1.0,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (240, 320),
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.domain = domain
        self.severity = severity
        self.render_mode = render_mode
        self.render_size = render_size
        self.layout = obs_layout(self.cfg)
        n_obs = self.layout["force"].stop
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.n_sub = int(round(self.cfg.control_dt / self.cfg.physics_dt))
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer = None
        self.params: EpisodeParams | None = None

    # ------------------------------------------------------------ building
    def _xml(self, p: EpisodeParams) -> str:
        a = PEG_HALF_W + p.clearance / 2
        c, W, H, T = CHAMFER, SLOT_HALF_OUTER, SLOT_DEPTH, 0.03
        timeconst = 0.006 / np.sqrt(max(p.contact_stiffness, 1e-3))
        lo = H - c
        geoms = []

        def box(name: str, x: float, z: float, hx: float, hz: float, ry: float = 0.0) -> None:
            geoms.append(
                f'<geom name="{name}" type="box" pos="{x:.6f} 0 {z:.6f}" size="{hx:.6f} {T} {hz:.6f}" '
                f'euler="0 {ry:.6f} 0" friction="{p.friction} 0.005 0.0001" rgba="0.55 0.62 0.72 1"/>'
            )

        for s in (1, -1):
            box(f"wall{s}", s * (a + W) / 2, lo / 2, (W - a) / 2, lo / 2)
            box(f"top{s}", s * (a + c + W) / 2, lo + c / 2, (W - a - c) / 2, c / 2)
            box(f"cham{s}", s * (a + c), lo, c / np.sqrt(2), c / np.sqrt(2), np.pi / 4)
        box("floor", 0.0, -0.005, W, 0.005)
        m = p.peg_mass
        return f"""
<mujoco model="contact_insertion">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{self.cfg.physics_dt}" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default><geom solref="{timeconst:.5f} 1" solimp="0.95 0.99 0.0005 0.5 2" condim="3"/></default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.98 0.98 0.97" rgb2="0.82 0.85 0.9" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 0.5" dir="0 0.6 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="front" pos="{p.slot_x:.4f} -0.32 0.07" xyaxes="1 0 0 0 0 1"/>
    <body name="slot" pos="{p.slot_x:.6f} 0 0" euler="0 {p.slot_angle:.6f} 0">
      {"".join(geoms)}
    </body>
    <body name="peg" pos="0 0 0" gravcomp="1">
      <joint name="jx" type="slide" axis="1 0 0" armature="0.05"/>
      <joint name="jz" type="slide" axis="0 0 1" armature="0.05"/>
      <joint name="jr" type="hinge" axis="0 1 0" armature="0.001"/>
      <geom name="peg" type="box" pos="0 0 {PEG_HALF_L}" size="{PEG_HALF_W} {PEG_HALF_T} {PEG_HALF_L}"
            mass="{m:.4f}" friction="{p.friction} 0.005 0.0001" rgba="0.85 0.35 0.2 1"/>
      <site name="tip" pos="0 0 0" size="0.002"/>
    </body>
  </worldbody>
  <actuator>
    <general name="ax" joint="jx" gaintype="fixed" biastype="affine" gainprm="1" biasprm="0 -1 -1" forcerange="-80 80"/>
    <general name="az" joint="jz" gaintype="fixed" biastype="affine" gainprm="1" biasprm="0 -1 -1" forcerange="-80 80"/>
    <general name="ar" joint="jr" gaintype="fixed" biastype="affine" gainprm="1" biasprm="0 -1 -1" forcerange="-2 2"/>
  </actuator>
</mujoco>"""

    def _set_gains(self, compliance01: float) -> None:
        """Scale PD stiffness with the compliance command (0 = soft, 1 = stiff)."""
        assert self.model is not None
        s = 0.25 + 0.75 * compliance01
        m_lin = self.params.peg_mass + 0.05
        i_rot = self.params.peg_mass * (2 * PEG_HALF_L) ** 2 / 3 + 0.001
        zeta = self.cfg.damping_ratio
        for i, (kp, inertia) in enumerate(
            ((self.cfg.kp_lin, m_lin), (self.cfg.kp_lin, m_lin), (self.cfg.kp_rot, i_rot))
        ):
            k = kp * s
            kv = 2 * zeta * np.sqrt(k * inertia)
            self.model.actuator_gainprm[i, 0] = k
            self.model.actuator_biasprm[i, 1] = -k
            self.model.actuator_biasprm[i, 2] = -kv
        self.compliance01 = compliance01

    # ----------------------------------------------------------- geometry
    def entrance(self) -> np.ndarray:
        """World (x, z) of the slot entrance centre."""
        phi = self.params.slot_angle
        local = np.array([0.0, SLOT_DEPTH])
        return np.array([self.params.slot_x, 0.0]) + _rot(phi) @ local

    def peg_pose(self) -> np.ndarray:
        q = self.data.qpos[:3]
        return np.array([q[0], q[1], q[2]])

    def slot_frame_tip(self) -> np.ndarray:
        """Peg tip in the slot frame: (lateral offset, height above slot floor)."""
        tip = self.peg_pose()[:2]
        rel = tip - np.array([self.params.slot_x, 0.0])
        return _rot(-self.params.slot_angle) @ rel

    def insertion_depth(self) -> float:
        return float(SLOT_DEPTH - self.slot_frame_tip()[1])

    def contact_wrench(self) -> tuple[np.ndarray, bool]:
        """True planar wrench (Fx, Fz, Ty about the tip) exerted on the peg by
        the slot, summed over all peg contacts, and a contact flag."""
        m, d = self.model, self.data
        total = np.zeros(3)
        buf = np.zeros(6)
        tip = np.array([d.qpos[0], 0.0, d.qpos[1]])
        touching = False
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 != self._peg_geom and con.geom2 != self._peg_geom:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            frame = con.frame.reshape(3, 3)
            f_world = frame.T @ buf[:3]
            # mj_contactForce returns the force of geom1 on geom2 along the
            # contact normal (pointing from geom1 to geom2).
            f_on_peg = f_world if con.geom2 == self._peg_geom else -f_world
            r = con.pos - tip
            total[0] += f_on_peg[0]
            total[1] += f_on_peg[2]
            total[2] += r[2] * f_on_peg[0] - r[0] * f_on_peg[2]
            touching = True
        return total, touching

    # ---------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        if "params" in options:
            self.params, self.concrete_domain = options["params"], options.get("domain", self.domain)
        else:
            self.params, self.concrete_domain = sample_params(
                rng, options.get("domain", self.domain), options.get("severity", self.severity)
            )
        self.model = mujoco.MjModel.from_xml_string(self._xml(self.params))
        self.data = mujoco.MjData(self.model)
        self._peg_geom = self.model.geom("peg").id
        self._renderer = None
        ent = self.entrance()
        start = np.array(
            [
                ent[0] + rng.uniform(-0.03, 0.03),
                ent[1] + rng.uniform(0.03, 0.06),
                rng.uniform(-0.08, 0.08),
            ]
        )
        self.data.qpos[:3] = start
        self.setpoint = start.copy()
        self.data.ctrl[:] = self.setpoint
        self._set_gains(0.5)
        mujoco.mj_forward(self.model, self.data)
        self.force_sensor = ForceSensor(self.params, rng)
        self.extero = Exteroception(self.params, rng)
        self.t = 0
        self.first_contact_step: int | None = None
        self.success = False
        self.jammed = False
        self.damaged = False
        self.peak_force_episode = 0.0
        self.depth_hist: deque[float] = deque(maxlen=self.cfg.jam_window)
        self.contact_hist: deque[bool] = deque(maxlen=self.cfg.jam_window)
        self.force_hist: deque[float] = deque(maxlen=self.cfg.jam_window)
        self.last_action = np.zeros(4, dtype=np.float32)
        wrench, touching = self.contact_wrench()
        self._potential = self._progress_potential()
        obs = self._observe(wrench)
        info = self._info(wrench, touching, peak=float(np.linalg.norm(wrench[:2])), first=False)
        info["params"] = self.params.to_dict()
        info["domain"] = self.concrete_domain
        return obs, info

    def step(self, action: np.ndarray):
        if self.data is None:
            raise RuntimeError("call reset() before step()")
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {a.shape}")
        q = self.data.qpos[:3]
        self.setpoint = self.setpoint + a[:3] * np.asarray(self.cfg.action_scale)
        win = np.asarray(self.cfg.windup)
        self.setpoint = np.clip(self.setpoint, q - win, q + win)
        self._set_gains((a[3] + 1.0) / 2.0)
        self.data.ctrl[:] = self.setpoint
        peak, touching_any = 0.0, False
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            if self.data.ncon:
                w, touching = self.contact_wrench()
                peak = max(peak, float(np.hypot(w[0], w[1])))
                touching_any |= touching
        self.t += 1
        self.last_action = a.astype(np.float32)
        wrench, touching = self.contact_wrench()
        f_norm = float(np.hypot(wrench[0], wrench[1]))
        in_contact = f_norm > self.cfg.contact_threshold or peak > self.cfg.contact_threshold
        first = False
        if in_contact and self.first_contact_step is None:
            self.first_contact_step = self.t
            first = True
        self.peak_force_episode = max(self.peak_force_episode, peak)
        depth = self.insertion_depth()
        lateral = abs(self.slot_frame_tip()[0])
        self.depth_hist.append(depth)
        self.contact_hist.append(in_contact)
        self.force_hist.append(f_norm)
        self.success = depth >= self.cfg.goal_depth and lateral < self.params.clearance / 2 + 1e-4
        if not self.success and len(self.depth_hist) == self.cfg.jam_window:
            no_progress = max(self.depth_hist) - self.depth_hist[0] < self.cfg.jam_progress
            pressing = all(self.contact_hist) and np.mean(self.force_hist) > self.cfg.jam_min_force
            self.jammed = bool(no_progress and pressing)
        self.damaged = peak > self.cfg.damage_force
        terminated = bool(self.success or self.jammed or self.damaged)
        truncated = bool(not terminated and self.t >= self.cfg.max_steps)
        reward = self._reward(peak)
        obs = self._observe(wrench)
        info = self._info(wrench, in_contact, peak, first)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ helpers
    def _progress_potential(self) -> float:
        lat, height = self.slot_frame_tip()
        depth = SLOT_DEPTH - height
        if depth <= 0:
            return -float(np.hypot(lat, height - SLOT_DEPTH))
        return float(depth - abs(lat))

    def _reward(self, peak: float) -> float:
        c = self.cfg
        pot = self._progress_potential()
        r = c.w_progress * (pot - self._potential)
        self._potential = pot
        r += c.w_success * float(self.success)
        r -= c.w_force * max(0.0, peak - c.safe_force) / c.safe_force
        r -= c.w_jam * float(self.jammed or self.damaged)
        r -= c.w_time
        return float(r)

    def _observe(self, wrench: np.ndarray) -> np.ndarray:
        pose = self.peg_pose()
        ent = self.entrance()
        rel_true = np.array([ent[0] - pose[0], ent[1] - pose[1], self.params.slot_angle - pose[2]])
        est, valid = self.extero(rel_true)
        parts = [est, [valid], pose, self.data.qvel[:3], [self.compliance01]]
        if self.cfg.proprio_include_effort:
            parts.append(self.data.actuator_force[:3])
        parts.append(self.force_sensor(wrench))
        return np.concatenate(parts).astype(np.float32)

    def _info(self, wrench: np.ndarray, in_contact: bool, peak: float, first: bool) -> dict[str, Any]:
        depth = self.insertion_depth()
        since = None if self.first_contact_step is None else self.t - self.first_contact_step
        mode = label_step(
            success=self.success,
            failed=self.jammed or self.damaged,
            steps_since_first_contact=since,
            first_contact_window=self.cfg.first_contact_window,
            depth=depth,
            entry_depth=CHAMFER + 0.0005,
            in_contact=in_contact,
        )
        return {
            "contact_mode": int(mode),
            "first_contact": bool(first),
            "success": bool(self.success),
            "jam": bool(self.jammed),
            "damage": bool(self.damaged),
            "insertion_depth": float(depth),
            "lateral_error": float(self.slot_frame_tip()[0]),
            "contact_force_norm": float(np.hypot(wrench[0], wrench[1])),
            "peak_force": float(peak),
            "wrench_true": wrench.astype(np.float32),
            "in_contact": bool(in_contact),
            "t": int(self.t),
        }

    def privileged_state(self) -> dict[str, Any]:
        """Ground-truth state for the scripted expert (data-generation oracle only)."""
        wrench, touching = self.contact_wrench()
        lat, height = self.slot_frame_tip()
        return {
            "entrance": self.entrance(),
            "pose": self.peg_pose(),
            "slot_angle": self.params.slot_angle,
            "wrench": wrench,
            "touching": touching,
            "lateral": float(lat),
            "depth": float(SLOT_DEPTH - height),
            "clearance": self.params.clearance,
        }

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise ValueError("render() requires render_mode='rgb_array'")
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, *self.render_size)
            except Exception as exc:  # noqa: BLE001 - surface a helpful message
                raise RuntimeError(
                    "MuJoCo offscreen rendering is unavailable. Set MUJOCO_GL=egl (GPU) or "
                    "MUJOCO_GL=osmesa (CPU, needs libosmesa) before importing mujoco."
                ) from exc
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
