"""Constraint vocabulary (closed set).

This is the *only* language the VLM may emit (Stage 4) and the only one the solver
must understand. Adding a type is a contract change: update docs/contracts.md.

Implicit hard rules are always enforced by the solver and are not listed here:
inside the floor polygon, no footprint overlap, footprints clear of obstacles,
other assets' clearance zones and door swings.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat


class _Base(BaseModel):
    id: str
    hard: bool = False
    weight: PositiveFloat = 1.0  # soft constraints only
    origin: Literal["rule_library", "user", "vlm"] = "user"
    rationale: str | None = None


class Faces(_Base):
    """a's front points at b's centre, within ``max_angle_deg``."""

    type: Literal["faces"] = "faces"
    a: str
    b: str
    max_angle_deg: PositiveFloat = 20.0


class AgainstWall(_Base):
    """a's back (local -Y side) is within ``max_gap`` of a wall (any wall, or ``wall_id``)."""

    type: Literal["against_wall"] = "against_wall"
    a: str
    wall_id: str | None = None
    max_gap: NonNegativeFloat = 0.05


class Near(_Base):
    """Footprint-to-footprint distance between a and b is at most ``max_distance``."""

    type: Literal["near"] = "near"
    a: str
    b: str
    max_distance: NonNegativeFloat


class DistanceAtLeast(_Base):
    """Footprint-to-footprint distance between a and b is at least ``min_distance``."""

    type: Literal["distance_at_least"] = "distance_at_least"
    a: str
    b: str
    min_distance: NonNegativeFloat


class Parallel(_Base):
    """a and b orientation axes are parallel (yaw difference modulo 180 deg)."""

    type: Literal["parallel"] = "parallel"
    a: str
    b: str
    max_angle_deg: PositiveFloat = 5.0


class KeepClear(_Base):
    """No footprint within ``distance`` in front of an opening (door, window, passage).
    For windows, only assets taller than the sill count."""

    type: Literal["keep_clear"] = "keep_clear"
    opening_id: str
    distance: PositiveFloat = 0.6


class WalkwayWidth(_Base):
    """Every door/passage and every asset's access zone stay mutually reachable
    through free floor at least ``min_width`` wide."""

    type: Literal["walkway_width"] = "walkway_width"
    min_width: PositiveFloat = 0.9


class Fixed(_Base):
    """a keeps its initial pose (requires ``LayoutProblem.initial`` to contain a)."""

    type: Literal["fixed"] = "fixed"
    a: str


Constraint = Annotated[
    Faces | AgainstWall | Near | DistanceAtLeast | Parallel | KeepClear | WalkwayWidth | Fixed,
    Field(discriminator="type"),
]


def referenced_assets(c: Constraint) -> list[str]:
    return [v for k in ("a", "b") if (v := getattr(c, k, None)) is not None]
