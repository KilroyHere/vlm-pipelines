"""Convert LLM output items into contract constraints, dropping invalid ones with reasons.

Every item is checked on its own (references exist, values are sane meters/degrees, no
duplicates) so one bad item never costs the rest. The surviving set is then validated
as ``LayoutProblem.constraints`` together with the problem's existing constraints, the
same check the solver's input goes through.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from roomlayout.schema import (
    AgainstWall,
    Constraint,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    LayoutProblem,
    Near,
    Parallel,
    WalkwayWidth,
)

from .output import (
    AgainstWallItem,
    DistanceAtLeastItem,
    FacesItem,
    FixedItem,
    KeepClearItem,
    LLMItem,
    LLMOutput,
    NearItem,
    ParallelItem,
    WalkwayWidthItem,
)

_CONSTRAINT = TypeAdapter(Constraint)

# Plausibility bounds. Distances are also capped by the room's bounding-box diagonal:
# a larger value is almost always a unit slip (feet or centimeters read as meters).
MAX_FACES_ANGLE_DEG = 90.0
MAX_PARALLEL_ANGLE_DEG = 45.0
MAX_WALL_GAP_M = 0.5
MAX_KEEP_CLEAR_M = 3.0
WALKWAY_RANGE_M = (0.4, 2.0)

_SYMMETRIC = {"near", "distance_at_least", "parallel"}


class Dropped(BaseModel):
    """An output item that was not turned into a constraint."""

    item: dict[str, Any]
    reason: str
    code: str  # unknown_asset | unknown_opening | unknown_wall | same_asset | bad_value |
    # no_current_position | duplicate | already_present | contract
    repairable: bool


class Accepted(BaseModel):
    constraint: Constraint
    restatement: str


def _room_extent(problem: LayoutProblem) -> float:
    xs = [p[0] for p in problem.room.floor.points]
    ys = [p[1] for p in problem.room.floor.points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _describe(item: LLMItem) -> str:
    d = item.model_dump(exclude={"source_text", "restatement", "hard", "type"})
    args = ", ".join(f"{k}={v!r}" for k, v in d.items())
    return f"{item.type}({args})"


def _key(c: Constraint) -> tuple:
    a = getattr(c, "a", None)
    b = getattr(c, "b", None)
    if c.type in _SYMMETRIC and a is not None and b is not None:
        a, b = sorted((a, b))
    return (c.type, a, b, getattr(c, "wall_id", None), getattr(c, "opening_id", None))


def _same(c1: Constraint, c2: Constraint) -> bool:
    """Same relation, same numbers, same strength (ids/origin/rationale ignored)."""
    skip = {"id", "origin", "rationale", "a", "b"}
    if _key(c1) != _key(c2):
        return False
    d1 = c1.model_dump(exclude=skip)
    d2 = c2.model_dump(exclude=skip)
    return all(
        (isinstance(v, float) and math.isclose(v, d2[k], abs_tol=1e-6)) or v == d2[k]
        for k, v in d1.items()
    )


def _base_id(item: LLMItem) -> str:
    parts = ["vlm", item.type]
    for attr in ("a", "b", "wall_id", "opening_id"):
        v = getattr(item, attr, None)
        if v:
            parts.append(str(v))
    return "_".join(parts)


class _Checker:
    def __init__(self, problem: LayoutProblem):
        self.problem = problem
        self.assets = {a.id for a in problem.assets}
        self.openings = {o.id for o in problem.room.openings}
        self.walls = {w.id for w in problem.room.walls}
        self.extent = _room_extent(problem)

    def check(self, item: LLMItem) -> tuple[str, str, bool] | None:
        """(code, reason, repairable) if the item is invalid, else None."""
        for attr in ("a", "b"):
            ref = getattr(item, attr, None)
            if ref is not None and ref not in self.assets:
                return ("unknown_asset", f"unknown asset id {ref!r}", True)
        if isinstance(item, KeepClearItem) and item.opening_id not in self.openings:
            return ("unknown_opening", f"unknown opening id {item.opening_id!r}", True)
        if isinstance(item, AgainstWallItem) and item.wall_id is not None:
            if item.wall_id not in self.walls:
                return ("unknown_wall", f"unknown wall id {item.wall_id!r}", True)
        b = getattr(item, "b", None)
        if b is not None and b == item.a:  # type: ignore[union-attr]
            return ("same_asset", f"relates {b!r} to itself", True)
        bad = self._bad_value(item)
        if bad:
            return ("bad_value", bad, True)
        if isinstance(item, FixedItem) and item.a not in self.problem.initial:
            return (
                "no_current_position",
                f"{item.a!r} has no recorded current position, so it can't be kept in place",
                False,
            )
        return None

    def _bad_value(self, item: LLMItem) -> str | None:
        def out(name: str, v: float | None, lo: float, hi: float, lo_open: bool) -> str | None:
            if v is None:
                return None
            if not math.isfinite(v) or (v <= lo if lo_open else v < lo) or v > hi:
                op = ">" if lo_open else ">="
                return f"{name}={v:g} outside plausible range ({op} {lo:g}, <= {hi:.2f})"
            return None

        e = self.extent
        match item:
            case FacesItem():
                return out("max_angle_deg", item.max_angle_deg, 0, MAX_FACES_ANGLE_DEG, True)
            case ParallelItem():
                return out("max_angle_deg", item.max_angle_deg, 0, MAX_PARALLEL_ANGLE_DEG, True)
            case AgainstWallItem():
                return out("max_gap", item.max_gap, 0, MAX_WALL_GAP_M, False)
            case NearItem():
                return out("max_distance (m)", item.max_distance, 0, e, False)
            case DistanceAtLeastItem():
                return out("min_distance (m)", item.min_distance, 0, e, True)
            case KeepClearItem():
                return out("distance (m)", item.distance, 0, min(MAX_KEEP_CLEAR_M, e), True)
            case WalkwayWidthItem():
                lo, hi = WALKWAY_RANGE_M
                return out("min_width (m)", item.min_width, lo, hi, False)
        return None


def to_constraint(item: LLMItem, cid: str) -> Constraint:
    """Map one LLM item onto the contract (origin="vlm", rationale = the user's words)."""
    common: dict[str, Any] = {
        "id": cid,
        "hard": item.hard,
        "origin": "vlm",
        "rationale": item.source_text,
    }

    def opt(**kw: float | str | None) -> dict[str, Any]:
        return {k: v for k, v in kw.items() if v is not None}

    match item:
        case FacesItem():
            return Faces(**common, a=item.a, b=item.b, **opt(max_angle_deg=item.max_angle_deg))
        case AgainstWallItem():
            return AgainstWall(
                **common, a=item.a, wall_id=item.wall_id, **opt(max_gap=item.max_gap)
            )
        case NearItem():
            return Near(**common, a=item.a, b=item.b, max_distance=item.max_distance)
        case DistanceAtLeastItem():
            return DistanceAtLeast(**common, a=item.a, b=item.b, min_distance=item.min_distance)
        case ParallelItem():
            return Parallel(**common, a=item.a, b=item.b, **opt(max_angle_deg=item.max_angle_deg))
        case KeepClearItem():
            return KeepClear(**common, opening_id=item.opening_id, **opt(distance=item.distance))
        case WalkwayWidthItem():
            return WalkwayWidth(**common, **opt(min_width=item.min_width))
        case FixedItem():
            return Fixed(**common, a=item.a)
    raise TypeError(f"unhandled item type {type(item).__name__}")


def convert(
    output: LLMOutput, problem: LayoutProblem
) -> tuple[list[Accepted], list[Dropped], list[str]]:
    """Validate and convert. Returns (accepted, dropped, repair_errors).

    ``repair_errors`` are one-line messages for the repairable drops, phrased for the
    model (they are sent back in the optional repair round).
    """
    checker = _Checker(problem)
    taken = {c.id for c in problem.constraints}
    accepted: list[Accepted] = []
    dropped: list[Dropped] = []
    repair_errors: list[str] = []

    def drop(i: int, item: LLMItem, code: str, reason: str, repairable: bool) -> None:
        dropped.append(
            Dropped(item=item.model_dump(), reason=reason, code=code, repairable=repairable)
        )
        if repairable:
            repair_errors.append(f"constraints[{i}] {_describe(item)}: {reason}")

    for i, item in enumerate(output.constraints):
        problem_found = checker.check(item)
        if problem_found:
            drop(i, item, *problem_found)
            continue
        base = _base_id(item)
        cid, n = base, 2
        while cid in taken:
            cid, n = f"{base}_{n}", n + 1
        try:
            c = to_constraint(item, cid)
        except ValidationError as e:  # contract field limits (e.g. PositiveFloat)
            drop(i, item, "contract", _first_error(e), True)
            continue
        if any(_same(c, prev.constraint) for prev in accepted):
            drop(i, item, "duplicate", "duplicate of an earlier item", False)
            continue
        if any(_same(c, old) for old in problem.constraints):
            drop(i, item, "already_present", "already in the problem's constraints", False)
            continue
        taken.add(cid)
        accepted.append(Accepted(constraint=c, restatement=item.restatement))

    accepted = _contract_check(problem, accepted, dropped)
    return accepted, dropped, repair_errors


def _first_error(e: ValidationError) -> str:
    err = e.errors()[0]
    loc = ".".join(str(x) for x in err["loc"])
    return f"{loc}: {err['msg']}" if loc else err["msg"]


def _with(problem: LayoutProblem, extra: list[Constraint]) -> None:
    data = problem.model_dump()
    data["constraints"] = [c.model_dump() for c in [*problem.constraints, *extra]]
    LayoutProblem.model_validate(data)


def _contract_check(
    problem: LayoutProblem, accepted: list[Accepted], dropped: list[Dropped]
) -> list[Accepted]:
    """The result must validate as ``LayoutProblem.constraints``; drop any item that doesn't."""
    try:
        _with(problem, [a.constraint for a in accepted])
        return accepted
    except ValidationError:
        pass
    kept: list[Accepted] = []
    for acc in accepted:
        try:
            _with(problem, [*(k.constraint for k in kept), acc.constraint])
            kept.append(acc)
        except ValidationError as e:
            dropped.append(
                Dropped(
                    item=acc.constraint.model_dump(),
                    reason=f"fails LayoutProblem validation: {_first_error(e)}",
                    code="contract",
                    repairable=False,
                )
            )
    return kept


def conflicts(problem: LayoutProblem, new: list[Constraint]) -> list[str]:
    """Plain-English warnings for requests that cannot all hold (not dropped: the user
    said them; the app should ask which one they meant)."""
    every = [*problem.constraints, *new]
    new_ids = {c.id for c in new}
    out: list[str] = []
    for c in every:
        if not isinstance(c, Near):
            continue
        for d in every:
            if (
                isinstance(d, DistanceAtLeast)
                and {c.a, c.b} == {d.a, d.b}
                and d.min_distance > c.max_distance
                and (c.id in new_ids or d.id in new_ids)
            ):
                out.append(
                    f"{c.id} wants {c.a} and {c.b} within {c.max_distance:g} m, but {d.id} "
                    f"wants them at least {d.min_distance:g} m apart"
                )
    walls: dict[str, list[AgainstWall]] = {}
    for c in every:
        if isinstance(c, AgainstWall) and c.hard and c.wall_id is not None:
            walls.setdefault(c.a, []).append(c)
    for a, cs in walls.items():
        if len({c.wall_id for c in cs}) > 1 and any(c.id in new_ids for c in cs):
            ids = ", ".join(c.id for c in cs)
            out.append(f"{a} is required against different walls ({ids})")
    return out
