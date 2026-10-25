"""Named physical domains and per-episode randomisation.

A *domain* is a set of ranges for the physical and sensing parameters below.
Each episode draws one ``EpisodeParams`` from its domain using the episode's
seeded generator. Shifted domains are defined as an interpolation, controlled
by ``severity`` in [0, 1], between the source ranges and the fully shifted
ranges, so the same code produces both the named shifts and graded shift
levels for the representation-reuse experiment.

Units: metres, radians, newtons, kilograms, control steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

Range = tuple[float, float]


@dataclass(frozen=True)
class EpisodeParams:
    """Everything that varies between episodes and is hidden from the policy."""

    # geometry
    slot_x: float = 0.0  # lateral slot position [m]
    slot_angle: float = 0.0  # slot tilt about the y axis [rad]
    clearance: float = 0.0009  # slot width minus peg width [m]
    # physics
    friction: float = 0.35  # sliding friction coefficient (peg and slot)
    peg_mass: float = 0.25  # [kg]
    contact_stiffness: float = 1.0  # multiplier on contact stiffness (1 = default)
    # exteroception ("camera") model
    extero_noise: float = 0.0005  # per-frame noise of the relative target estimate [m]
    extero_bias: float = 0.0  # fixed per-episode calibration offset magnitude [m]
    occlusion: float = 0.0  # probability that a frame near the slot is dropped
    # force-signal model
    force_noise: float = 0.2  # white noise std [N] (torque: 1 % of this, in N m)
    force_bias: float = 0.0  # fixed per-episode bias magnitude [N]
    force_delay: float = 0.0  # delay in control steps (rounded)
    force_dropout: float = 0.0  # per-step probability that the reading is zeroed
    # added in v0.2 (appended so that existing domains draw identical random streams)
    chamfer: float = 0.003  # 45-degree chamfer size at the slot entrance [m]
    search_offset: float = 0.0  # extra lateral offset of the pose estimate, random sign [m]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


SOURCE: dict[str, Range] = {
    "slot_x": (-0.02, 0.02),
    "slot_angle": (-0.035, 0.035),
    "clearance": (0.0006, 0.0012),
    "friction": (0.2, 0.5),
    "peg_mass": (0.15, 0.35),
    "contact_stiffness": (0.7, 1.0),
    "extero_noise": (0.0003, 0.0008),
    "extero_bias": (0.0, 0.0025),
    "occlusion": (0.0, 0.2),
    "force_noise": (0.1, 0.3),
    "force_bias": (0.0, 0.2),
    "force_delay": (0.0, 0.0),
    "force_dropout": (0.0, 0.0),
    "chamfer": (0.003, 0.003),
    "search_offset": (0.0, 0.0),
}

# Fully shifted ranges (severity = 1). Only the listed keys move.
SHIFTS: dict[str, dict[str, Range]] = {
    "source": {},
    "shifted_friction": {"friction": (0.75, 1.0)},
    "tight_clearance": {"clearance": (0.00025, 0.00045)},
    "sensor_noise": {
        "force_noise": (1.0, 2.0),
        "force_bias": (1.0, 2.0),
        "force_delay": (2.0, 2.0),
        "force_dropout": (0.1, 0.1),
    },
    "occlusion": {"occlusion": (0.7, 0.9), "extero_noise": (0.0015, 0.0025)},
    # v0.2 additions (not part of the original five evaluation domains)
    # small_chamfer: much smaller entry chamfer, a geometry change used for the
    #   adaptation test because it lowers zero-shot success of the trained models
    "small_chamfer": {"chamfer": (0.0008, 0.0012)},
    # search: second task. The pose estimate is offset laterally by 5 to 9 mm, so
    #   the peg lands on the top face and has to find the opening by sliding.
    "search": {"search_offset": (0.005, 0.009)},
}

# the original evaluation domains (v0.1); "mixed" samples only from these
CORE_DOMAINS: tuple[str, ...] = ("source", "shifted_friction", "tight_clearance", "sensor_noise", "occlusion")
DOMAINS: tuple[str, ...] = (*SHIFTS.keys(), "mixed")


def domain_ranges(name: str, severity: float = 1.0) -> dict[str, Range]:
    """Parameter ranges of a named domain at a given shift severity.

    Composite domains join named shifts with ``+`` (for example
    ``search+tight_clearance``); ``severity`` applies to every component.
    """
    parts = name.split("+")
    for part in parts:
        if part not in SHIFTS:
            raise ValueError(f"unknown domain {part!r}; choose from {DOMAINS} (or join them with '+')")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    out = dict(SOURCE)
    for part in parts:
        for key, (hi_lo, hi_hi) in SHIFTS[part].items():
            lo, hi = SOURCE[key]
            out[key] = (lo + severity * (hi_lo - lo), hi + severity * (hi_hi - hi))
    return out


def sample_params(rng: np.random.Generator, domain: str = "source", severity: float = 1.0) -> tuple[EpisodeParams, str]:
    """Draw episode parameters. Returns the parameters and the concrete domain
    used (for ``mixed`` this is one of the named domains, chosen uniformly)."""
    concrete = domain
    if domain == "mixed":
        names = list(CORE_DOMAINS)
        concrete = names[int(rng.integers(len(names)))]
    ranges = domain_ranges(concrete, severity)
    vals = {}
    for f in fields(EpisodeParams):
        lo, hi = ranges[f.name]
        vals[f.name] = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    return EpisodeParams(**vals), concrete
