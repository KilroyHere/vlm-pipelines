"""Why each layout looks the way it does, in plain words, with the numbers measured.

    from roomlayout.explain import explain_layout
    for line in explain_layout(problem, layout):
        line.kind   # "why" (a rule it meets), "bent" (a soft wish it misses), "broken" (hard)
        line.text   # "The gray L-shaped sectional faces the TV (4° off), 2.62 m away."

Measurements come from the solver's own geometry (``roomlayout.solver``): edge-to-edge
gaps between footprints, the facing angle, which wall a back is against, and the widest
walkway the layout keeps. The user's own requests come first, then the rule library's,
then the implicit rules (only when broken).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from roomlayout.constraints.vlm.summary import wall_names
from roomlayout.schema import (
    AgainstWall,
    Constraint,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    Layout,
    LayoutProblem,
    Near,
    Parallel,
    WalkwayWidth,
)
from roomlayout.solver import rules
from roomlayout.solver.geometry import Posed
from roomlayout.solver.model import Model

Kind = Literal["why", "bent", "broken"]
WALKWAY_WIDTHS = (1.2, 1.0, 0.9, 0.8, 0.7, 0.6)  # widest first; reported to 0.1 m


@dataclass
class Line:
    kind: Kind
    text: str
    constraint_id: str | None = None
    origin: str = "rule_library"  # user | vlm | rule_library | implicit
    assets: tuple[str, ...] = ()  # ids the line is about (the viewer highlights them)


def _name(problem: LayoutProblem, aid: str) -> str:
    a = next((x for x in problem.assets if x.id == aid), None)
    return f"the {a.label}" if a is not None and a.label else aid


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _wall_name(problem: LayoutProblem, wid: str) -> str:
    w = next((x for x in problem.room.walls if x.id == wid), None)
    if w is not None and getattr(w, "label", None):
        return f"the {w.label}"
    # Same compass names the request parser shows the model ("the south wall").
    return f"the {wall_names(problem).get(wid, wid)}"


def _nearest_wall(model: Model, g: Posed) -> str | None:
    (b0x, b0y), (b1x, b1y) = g.back
    best = None
    for w in model.room.walls:
        d = max(w.segment_distance(b0x, b0y), w.segment_distance(b1x, b1y))
        if best is None or d < best[0]:
            best = (d, w.id)
    return best[1] if best else None


def _back_gap(model: Model, g: Posed, wid: str | None) -> float | None:
    w = next((x for x in model.room.walls if x.id == wid), None)
    if w is None:
        return None
    (b0x, b0y), (b1x, b1y) = g.back
    return max(w.segment_distance(b0x, b0y), w.segment_distance(b1x, b1y))


def _widest_walkway(model: Model, geoms: list[Posed], skip: frozenset[str]) -> float | None:
    for w in WALKWAY_WIDTHS:
        if rules.walkway(model.room, model.templates, geoms, w, skip=skip).violation <= 0:
            return w
    return None


def _measure(problem: LayoutProblem, model: Model, G: list[Posed], c: Constraint) -> str:
    """One sentence describing what the layout does for constraint ``c``."""
    n = lambda aid: _name(problem, aid)  # noqa: E731
    g = lambda aid: G[model.index[aid]]  # noqa: E731
    if isinstance(c, Faces):
        ga, gb = g(c.a), g(c.b)
        ang = math.degrees(rules.faces_angle(ga, gb))
        centre = math.hypot(gb.x - ga.x, gb.y - ga.y)
        return f"{_cap(n(c.a))} faces {n(c.b)} ({ang:.0f}° off), {centre:.2f} m away."
    if isinstance(c, Near):
        gap = rules.footprint_distance(g(c.a), g(c.b))
        return f"{_cap(n(c.a))} is {gap:.2f} m from {n(c.b)} (at most {c.max_distance:g} m)."
    if isinstance(c, DistanceAtLeast):
        gap = rules.footprint_distance(g(c.a), g(c.b))
        return f"{_cap(n(c.a))} and {n(c.b)} are {gap:.2f} m apart (at least {c.min_distance:g} m)."
    if isinstance(c, AgainstWall):
        ga = g(c.a)
        wid = c.wall_id or _nearest_wall(model, ga)
        wall = _wall_name(problem, wid) if wid else "a wall"
        gap = _back_gap(model, ga, wid)
        if gap is not None and gap > c.max_gap + 0.01:
            return f"{_cap(n(c.a))} is {gap:.2f} m from {wall}."
        return f"{_cap(n(c.a))} stands against {wall}."
    if isinstance(c, Parallel):
        ang = math.degrees(rules.parallel_angle(g(c.a), g(c.b)))
        return f"{_cap(n(c.a))} lines up with {n(c.b)} ({ang:.0f}° off)."
    if isinstance(c, KeepClear):
        o = next((x for x in problem.room.openings if x.id == c.opening_id), None)
        kind = o.type.value.replace("opening", "passage") if o is not None else "opening"
        where = f" in {_wall_name(problem, o.wall_id)}" if o is not None else ""
        return f"Nothing blocks the {kind}{where} ({c.distance:g} m kept clear)."
    if isinstance(c, Fixed):
        return f"{_cap(n(c.a))} stays where it is."
    if isinstance(c, WalkwayWidth):
        skip = next((wt.skip for wt in model.walkways if wt.cid == c.id), frozenset())
        w = _widest_walkway(model, G, skip)
        if w is None:
            return "Some doors or pieces can't be reached along a free path."
        return f"Every door and piece is reachable along paths at least {w:.1f} m wide."
    return c.rationale or c.id


def _assets_of(c: Constraint) -> tuple[str, ...]:
    return tuple(x for x in (getattr(c, "a", None), getattr(c, "b", None)) if x)


def explain_layout(problem: LayoutProblem, layout: Layout) -> list[Line]:
    """Plain-English lines for ``layout``: user requests first, then the rule library;
    implicit rules (no overlap, inside the room, ...) only when broken."""
    model = Model(problem)
    poses = {p.asset_id: (p.pose.x, p.pose.y, p.pose.yaw) for p in layout.placements}
    for aid, p in problem.initial.items():
        poses.setdefault(aid, (p.x, p.y, p.yaw))
    G = [model.geom(t.idx, poses[t.id]) for t in model.templates]
    by_id = {r.constraint_id: r for r in layout.results}
    order = {"user": 0, "vlm": 0, "rule_library": 1}
    lines: list[Line] = []
    walkway_done = False
    for c in sorted(problem.constraints, key=lambda c: order.get(c.origin, 2)):
        r = by_id.get(c.id)
        if r is None:
            continue
        if isinstance(c, WalkwayWidth):
            if walkway_done and r.satisfied:
                continue  # one walkway line is enough when all walkway rules hold
            walkway_done = True
        text = _measure(problem, model, G, c)
        if r.satisfied:
            kind: Kind = "why"
        else:
            kind = "broken" if c.hard else "bent"
            wish = c.rationale or c.id
            text = f"{text} Wanted: {wish}."
        lines.append(Line(kind, text, c.id, c.origin, _assets_of(c)))
    for r in layout.results:
        if r.constraint_id.startswith("implicit:") and not r.satisfied:
            rule = r.constraint_id.removeprefix("implicit:").replace("_", " ")
            lines.append(
                Line(
                    "broken",
                    f"Breaks '{rule}': {r.detail or ''}".strip(),
                    r.constraint_id,
                    "implicit",
                )  # fmt: skip
            )
    rank = {"broken": 0, "bent": 1, "why": 2}
    return sorted(lines, key=lambda x: (rank[x.kind], order.get(x.origin, 2)))
