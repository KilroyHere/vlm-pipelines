"""Solver input (LayoutProblem) and output (SolveResult)."""

from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat, model_validator

from .asset import Asset
from .constraints import Constraint, Fixed, referenced_assets
from .geometry import Pose2D
from .room import Room

SCHEMA_VERSION = "0.1.0"


class LayoutProblem(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    room: Room
    assets: list[Asset]
    constraints: list[Constraint] = []
    initial: dict[str, Pose2D] = Field(
        default_factory=dict, description="Current/scanned poses, keyed by asset id"
    )

    @model_validator(mode="after")
    def _check_refs(self) -> LayoutProblem:
        ids = [a.id for a in self.assets]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate asset ids")
        known = set(ids)
        openings = {o.id for o in self.room.openings}
        walls = {w.id for w in self.room.walls}
        for c in self.constraints:
            for ref in referenced_assets(c):
                if ref not in known:
                    raise ValueError(f"constraint {c.id!r} references unknown asset {ref!r}")
            if getattr(c, "opening_id", None) not in (None, *openings):
                raise ValueError(f"constraint {c.id!r} references unknown opening")
            if getattr(c, "wall_id", None) not in (None, *walls):
                raise ValueError(f"constraint {c.id!r} references unknown wall")
            if isinstance(c, Fixed) and c.a not in self.initial:
                raise ValueError(f"fixed constraint {c.id!r} needs an initial pose for {c.a!r}")
        for key in self.initial:
            if key not in known:
                raise ValueError(f"initial pose for unknown asset {key!r}")
        return self


class Placement(BaseModel):
    asset_id: str
    pose: Pose2D


class ConstraintResult(BaseModel):
    constraint_id: str  # implicit rules use "implicit:<name>" ids
    hard: bool
    satisfied: bool
    violation: NonNegativeFloat = 0.0  # 0 when satisfied; units are constraint-specific
    detail: str | None = None


class Layout(BaseModel):
    id: str
    placements: list[Placement]
    cost: float  # weighted soft violation; lower is better
    feasible: bool  # all hard constraints satisfied
    results: list[ConstraintResult] = []


class SolveStats(BaseModel):
    seed: int
    seconds: NonNegativeFloat
    evaluations: int


class SolveResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    problem_id: str
    layouts: list[Layout]  # best first, mutually diverse
    stats: SolveStats
    notes: list[str] = []  # e.g. which constraints block feasibility
