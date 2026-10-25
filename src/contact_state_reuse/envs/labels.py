"""Simulator-defined contact-mode labels.

These labels are *measurement* labels computed from privileged simulator
state for analysis (and, only in explicitly marked configurations, as an
auxiliary training target). They are never part of a policy observation. They
are a convenient discretisation of a continuous process; they are not a claim
that real contact phases are discretely separable.

Rules (evaluated every control step, in this order):

* ``SUCCESS``         insertion depth >= goal depth.
* ``JAM_OR_FAILURE``  the episode was terminated by jam detection or by
                      exceeding the damage force.
* ``FIRST_CONTACT``   within ``first_contact_window`` steps after the first
                      control step with peg/slot contact in the episode.
* ``INSERTION``       peg tip below the slot entrance by more than the chamfer
                      (the peg is inside the bore).
* ``ALIGNMENT``       the peg has touched the slot before and is in contact now
                      (on the chamfer or the top face).
* ``APPROACH``        everything else (free space before or between contacts).
"""

from __future__ import annotations

from enum import IntEnum


class ContactMode(IntEnum):
    APPROACH = 0
    FIRST_CONTACT = 1
    ALIGNMENT = 2
    INSERTION = 3
    SUCCESS = 4
    JAM_OR_FAILURE = 5


MODE_NAMES: tuple[str, ...] = tuple(m.name.lower() for m in ContactMode)
N_MODES: int = len(ContactMode)


def label_step(
    *,
    success: bool,
    failed: bool,
    steps_since_first_contact: int | None,
    first_contact_window: int,
    depth: float,
    entry_depth: float,
    in_contact: bool,
) -> ContactMode:
    """Assign the contact mode for one control step (see module docstring)."""
    if success:
        return ContactMode.SUCCESS
    if failed:
        return ContactMode.JAM_OR_FAILURE
    if steps_since_first_contact is not None and steps_since_first_contact < first_contact_window:
        return ContactMode.FIRST_CONTACT
    if depth > entry_depth:
        return ContactMode.INSERTION
    if steps_since_first_contact is not None and in_contact:
        return ContactMode.ALIGNMENT
    return ContactMode.APPROACH
