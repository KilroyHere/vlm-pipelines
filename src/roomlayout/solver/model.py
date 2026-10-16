"""Compile a LayoutProblem into cost terms; incremental cost state; per-constraint report.

Cost = W_HARD * (sum of hard violations) + sum(weight * soft violation).

Terms are either *unary* (depend on one asset's pose), *pair* (two assets) or the
global walkway check. A move only re-evaluates the unary and pair terms of the
assets it touched. The walkway check is lazy: it is evaluated exactly only for
candidates whose cheap hard terms are all zero; otherwise the last exact value is
carried over (such candidates are infeasible anyway and already pay W_HARD).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from roomlayout.schema import (
    AgainstWall,
    ConstraintResult,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    LayoutProblem,
    Near,
    OpeningType,
    Parallel,
    WalkwayWidth,
)

from . import rules
from .geometry import AssetTemplate, Posed, RoomGeom, bboxes_overlap

W_HARD = 100.0
GROUP_NEAR_MAX = 0.3  # m; hard `near` at most this tight makes a rigid group for moves
IMPLICIT_RULES = (
    "inside_room",
    "no_overlap",
    "clear_of_obstacles",
    "clearance_zones",
    "door_swing",
    "wall_mount",
)
Pose = tuple[float, float, float]


@dataclass
class Term:
    cid: str
    hard: bool
    weight: float
    a: int
    b: int  # -1 for unary terms
    fn: Callable[..., float]
    label: str
    unit: str
    local: bool = False  # zero whenever the two assets' footprint+zone boxes are disjoint


@dataclass
class WalkwayTerm:
    cid: str
    hard: bool
    weight: float
    min_width: float


def implicit_id(name: str) -> str:
    return f"implicit:{name}"


class Model:
    """Everything about a problem that does not change during search."""

    def __init__(self, problem: LayoutProblem):
        self.problem = problem
        self.room = RoomGeom(problem.room)
        self.templates = [AssetTemplate(i, a) for i, a in enumerate(problem.assets)]
        self.n = len(self.templates)
        self.index = {t.id: i for i, t in enumerate(self.templates)}
        self.notes: list[str] = []
        self.initial: list[Pose | None] = [None] * self.n
        for aid, p in problem.initial.items():
            self.initial[self.index[aid]] = (p.x, p.y, p.yaw)

        # pinned = never moved by the search (movable=False or a hard `fixed`)
        self.pinned = [False] * self.n
        for t in self.templates:
            if not t.asset.movable:
                if self.initial[t.idx] is None:
                    self.notes.append(
                        f"asset {t.id!r} has movable=false but no initial pose; "
                        "treating it as movable"
                    )
                else:
                    self.pinned[t.idx] = True
        for c in problem.constraints:
            if isinstance(c, Fixed) and c.hard:
                self.pinned[self.index[c.a]] = True
        self.movable = [i for i in range(self.n) if not self.pinned[i]]

        self.unary: list[list[Term]] = [[] for _ in range(self.n)]
        self.pair: dict[tuple[int, int], list[Term]] = {}
        self.walkways: list[WalkwayTerm] = []
        self.report_order: list[tuple[str, bool]] = [(implicit_id(r), True) for r in IMPLICIT_RULES]
        # search hints
        self.wall_walls: list[list[int] | None] = [None] * self.n  # walls to snap to
        self.wall_bound = [False] * self.n  # must stand against a wall (hard)
        self.wall_soft = [False] * self.n  # prefers a wall (soft against_wall)
        self.partners: list[list[int]] = [[] for _ in range(self.n)]
        self.faces_targets: list[list[int]] = [[] for _ in range(self.n)]
        # followers[i]: smaller movable assets tied to i by a hard `near` of <= 0.3 m
        # (desk -> chair, bed -> nightstands); they move rigidly with i in group moves
        self.followers: list[list[int]] = [[] for _ in range(self.n)]
        self._build_implicit()
        self._build_constraints()
        for c in problem.constraints:
            if isinstance(c, Near) and c.hard and c.max_distance <= GROUP_NEAR_MAX:
                a, b = self.index[c.a], self.index[c.b]
                if a == b or self.pinned[a] or self.pinned[b]:
                    continue
                big, small = (a, b) if self.templates[a].area >= self.templates[b].area else (b, a)
                if small not in self.followers[big]:
                    self.followers[big].append(small)

    # ------------------------------------------------------------------ building

    def _add_pair(self, term: Term) -> None:
        key = (min(term.a, term.b), max(term.a, term.b))
        self.pair.setdefault(key, []).append(term)

    def _build_implicit(self) -> None:
        room = self.room
        H = True
        for t in self.templates:
            i = t.idx
            self.unary[i].append(
                Term(
                    implicit_id("inside_room"),
                    H,
                    1.0,
                    i,
                    -1,
                    lambda g, room=room: rules.inside_room(room, g),
                    t.id,
                    "m²",
                )
            )
            if room.obstacles:
                self.unary[i].append(
                    Term(
                        implicit_id("clear_of_obstacles"),
                        H,
                        1.0,
                        i,
                        -1,
                        lambda g, room=room: rules.clear_of_obstacles(room, g),
                        t.id,
                        "m²",
                    )
                )
            if t.zones:
                self.unary[i].append(
                    Term(
                        implicit_id("clearance_zones"),
                        H,
                        1.0,
                        i,
                        -1,
                        lambda g, room=room: rules.own_zones_blocked(room, g),
                        f"{t.id} zones vs walls/obstacles",
                        "m²",
                    )
                )
            if room.swings:
                self.unary[i].append(
                    Term(
                        implicit_id("door_swing"),
                        H,
                        1.0,
                        i,
                        -1,
                        lambda g, room=room: rules.door_swing(room, g),
                        t.id,
                        "m²",
                    )
                )
            if t.wall_mounted:
                self.wall_bound[i] = True
                walls = room.walls
                self.unary[i].append(
                    Term(
                        implicit_id("wall_mount"),
                        H,
                        1.0,
                        i,
                        -1,
                        lambda g, t=t, walls=walls: rules.against_wall(
                            t, g, walls, rules.WALL_MOUNT_GAP
                        ),
                        t.id,
                        "",
                    )
                )
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ti, tj = self.templates[i], self.templates[j]
                self._add_pair(
                    Term(
                        implicit_id("no_overlap"),
                        H,
                        1.0,
                        i,
                        j,
                        rules.no_overlap,
                        f"{ti.id}×{tj.id}",
                        "m²",
                        local=True,
                    )
                )
                if ti.zones or tj.zones:
                    self._add_pair(
                        Term(
                            implicit_id("clearance_zones"),
                            H,
                            1.0,
                            i,
                            j,
                            rules.zones_pair,
                            f"{ti.id}×{tj.id}",
                            "m²",
                            local=True,
                        )
                    )

    def _build_constraints(self) -> None:
        room = self.room
        for c in self.problem.constraints:
            self.report_order.append((c.id, c.hard))
            w = c.weight
            if isinstance(c, Faces):
                a, b = self.index[c.a], self.index[c.b]
                ang = math.radians(c.max_angle_deg)
                self._pair_or_unary(
                    c.id, c.hard, w, a, b, lambda ga, gb, ang=ang: rules.faces(ga, gb, ang), "rad"
                )
                self.partners[a].append(b)
                self.faces_targets[a].append(b)
            elif isinstance(c, Near):
                a, b = self.index[c.a], self.index[c.b]
                d = c.max_distance
                self._pair_or_unary(
                    c.id, c.hard, w, a, b, lambda ga, gb, d=d: rules.near(ga, gb, d), "m"
                )
                self.partners[a].append(b)
                self.partners[b].append(a)
            elif isinstance(c, DistanceAtLeast):
                a, b = self.index[c.a], self.index[c.b]
                d = c.min_distance
                self._pair_or_unary(
                    c.id,
                    c.hard,
                    w,
                    a,
                    b,
                    lambda ga, gb, d=d: rules.distance_at_least(ga, gb, d),
                    "m",
                )
            elif isinstance(c, Parallel):
                a, b = self.index[c.a], self.index[c.b]
                ang = math.radians(c.max_angle_deg)
                self._pair_or_unary(
                    c.id,
                    c.hard,
                    w,
                    a,
                    b,
                    lambda ga, gb, ang=ang: rules.parallel(ga, gb, ang),
                    "rad",
                )
            elif isinstance(c, AgainstWall):
                a = self.index[c.a]
                t = self.templates[a]
                walls = [room.wall_by_id[c.wall_id]] if c.wall_id else room.walls
                self.unary[a].append(
                    Term(
                        c.id,
                        c.hard,
                        w,
                        a,
                        -1,
                        lambda g, t=t, walls=walls, gap=c.max_gap: rules.against_wall(
                            t, g, walls, gap
                        ),
                        t.id,
                        "",
                    )
                )
                if c.hard:
                    self.wall_bound[a] = True
                else:
                    self.wall_soft[a] = True
                if c.wall_id:
                    wi = [k for k, wg in enumerate(room.walls) if wg.id == c.wall_id]
                    prev = self.wall_walls[a]
                    self.wall_walls[a] = wi if prev is None else [k for k in prev if k in wi]
            elif isinstance(c, KeepClear):
                o = room.openings[c.opening_id]
                rect = room.keep_clear_rect(c.opening_id, c.distance)
                rb = rect.bounds
                for t in self.templates:
                    if o.type == OpeningType.window and t.height <= o.sill:
                        continue
                    self.unary[t.idx].append(
                        Term(
                            c.id,
                            c.hard,
                            w,
                            t.idx,
                            -1,
                            lambda g, rect=rect, rb=rb: rules.keep_clear(rect, rb, g),
                            f"{t.id} in front of {o.id}",
                            "m²",
                        )
                    )
            elif isinstance(c, WalkwayWidth):
                self.walkways.append(WalkwayTerm(c.id, c.hard, w, c.min_width))
            elif isinstance(c, Fixed):
                a = self.index[c.a]
                pose0 = self.initial[a]
                assert pose0 is not None  # guaranteed by LayoutProblem validation
                self.unary[a].append(
                    Term(
                        c.id, c.hard, w, a, -1, lambda g, p=pose0: rules.fixed_offset(g, p), c.a, ""
                    )
                )
            else:  # pragma: no cover - closed vocabulary
                raise TypeError(f"unknown constraint type {type(c).__name__}")

    def _pair_or_unary(self, cid, hard, weight, a, b, fn, unit) -> None:
        ta, tb = self.templates[a], self.templates[b]
        if a == b:
            self.notes.append(f"constraint {cid!r} relates {ta.id!r} to itself; ignored")
            self.unary[a].append(Term(cid, hard, weight, a, -1, lambda g: 0.0, ta.id, unit))
            return
        self._add_pair(Term(cid, hard, weight, a, b, fn, f"{ta.id}→{tb.id}", unit))

    # ------------------------------------------------------------------ helpers

    def geom(self, i: int, pose: Pose) -> Posed:
        return self.templates[i].posed(*pose)

    def report(self, poses: Sequence[Pose]) -> tuple[list[ConstraintResult], bool, float]:
        """Full (non-incremental) evaluation: per-constraint results, feasibility, cost."""
        G = [self.geom(i, p) for i, p in enumerate(poses)]
        agg: dict[str, list] = {cid: [hard, 0.0, []] for cid, hard in self.report_order}
        terms = [t for ts in self.unary for t in ts] + [t for ts in self.pair.values() for t in ts]
        for t in terms:
            v = t.fn(G[t.a]) if t.b < 0 else t.fn(G[t.a], G[t.b])
            if v > 0:
                entry = agg[t.cid]
                entry[1] += v
                entry[2].append(f"{t.label}: {v:.3g}{(' ' + t.unit) if t.unit else ''}")
        for wt in self.walkways:
            ev = rules.walkway(self.room, self.templates, G, wt.min_width)
            if ev.violation > 0:
                agg[wt.cid][1] += ev.violation
                agg[wt.cid][2].append("unreachable: " + ", ".join(ev.unreached))
        results, hard_sum, soft_sum = [], 0.0, 0.0
        weights = {c.id: c.weight for c in self.problem.constraints}
        for cid, (hard, v, parts) in agg.items():
            if hard:
                hard_sum += v
            else:
                soft_sum += weights.get(cid, 1.0) * v
            results.append(
                ConstraintResult(
                    constraint_id=cid,
                    hard=hard,
                    satisfied=v <= 0.0,
                    violation=v,
                    detail="; ".join(parts) if parts else None,
                )
            )
        return results, hard_sum <= 0.0, W_HARD * hard_sum + soft_sum


# ---------------------------------------------------------------------------- state


@dataclass
class Proposal:
    changes: dict[int, Pose]
    geoms: dict[int, Posed]
    unary: dict[int, tuple[float, float]]
    pair: dict[tuple[int, int], tuple[float, float]]
    hard: float  # cheap hard sum
    soft: float  # cheap weighted soft sum
    walk_hard: float
    walk_soft: float
    walk_exact: bool

    @property
    def cost(self) -> float:
        return W_HARD * (self.hard + self.walk_hard) + self.soft + self.walk_soft

    @property
    def feasible(self) -> bool:
        return self.hard <= 0.0 and self.walk_exact and self.walk_hard <= 0.0


@dataclass
class State:
    """Incrementally maintained cost of one (possibly partial) layout.

    Inactive assets (pose ``None``) are ignored, which the greedy initializer uses
    to place assets one by one. The walkway check only runs once all are active."""

    model: Model
    poses: list[Pose | None]
    geoms: list[Posed | None] = field(init=False)
    uh: list[float] = field(init=False)
    us: list[float] = field(init=False)
    ph: list[list[float]] = field(init=False)
    ps: list[list[float]] = field(init=False)
    hard: float = 0.0
    soft: float = 0.0
    walk_hard: float = 0.0
    walk_soft: float = 0.0
    walk_exact: bool = True
    evaluations: int = 0

    def __post_init__(self) -> None:
        n = self.model.n
        poses = list(self.poses)
        self.poses = [None] * n
        self.geoms = [None] * n
        self.uh, self.us = [0.0] * n, [0.0] * n
        self.ph = [[0.0] * n for _ in range(n)]
        self.ps = [[0.0] * n for _ in range(n)]
        changes = {i: p for i, p in enumerate(poses) if p is not None}
        if changes:
            self.commit(self.propose(changes))

    @property
    def cost(self) -> float:
        return W_HARD * (self.hard + self.walk_hard) + self.soft + self.walk_soft

    @property
    def feasible(self) -> bool:
        return self.hard <= 0.0 and self.walk_exact and self.walk_hard <= 0.0

    @property
    def complete(self) -> bool:
        return all(p is not None for p in self.poses)

    def _unary(self, i: int, g: Posed) -> tuple[float, float]:
        h = s = 0.0
        for t in self.model.unary[i]:
            v = t.fn(g)
            if v > 0.0:
                if t.hard:
                    h += v
                else:
                    s += t.weight * v
        return h, s

    def _pair(self, i: int, j: int, G: dict[int, Posed]) -> tuple[float, float]:
        h = s = 0.0
        gi, gj = G[i], G[j]
        far = not bboxes_overlap(gi.all_bounds, gj.all_bounds)
        for t in self.model.pair.get((i, j), ()):
            if far and t.local:
                continue
            v = t.fn(G[t.a], G[t.b])
            if v > 0.0:
                if t.hard:
                    h += v
                else:
                    s += t.weight * v
        return h, s

    def propose(self, changes: dict[int, Pose]) -> Proposal:
        m = self.model
        self.evaluations += 1
        new_geoms = {i: m.geom(i, p) for i, p in changes.items()}
        unary = {i: self._unary(i, g) for i, g in new_geoms.items()}
        active = [j for j in range(m.n) if self.poses[j] is not None or j in changes]
        G = {j: (new_geoms[j] if j in new_geoms else self.geoms[j]) for j in active}
        pair = {}
        for i in changes:
            for j in active:
                if j == i or (j in changes and j < i):
                    continue
                key = (i, j) if i < j else (j, i)
                pair[key] = self._pair(key[0], key[1], G)
        # exact totals from the cached per-asset / per-pair values (no running-sum drift)
        hard = soft = 0.0
        for k, i in enumerate(active):
            h, s = unary[i] if i in unary else (self.uh[i], self.us[i])
            hard += h
            soft += s
            ph, ps = self.ph[i], self.ps[i]
            for j in active[k + 1 :]:
                hs = pair.get((i, j))
                if hs is None:
                    hard += ph[j]
                    soft += ps[j]
                else:
                    hard += hs[0]
                    soft += hs[1]
        walk_hard, walk_soft, exact = self.walk_hard, self.walk_soft, self.walk_exact
        if m.walkways and len(active) == m.n:
            if hard <= 0.0:
                geoms = [G[j] for j in range(m.n)]
                walk_hard = walk_soft = 0.0
                for wt in m.walkways:
                    v = rules.walkway(m.room, m.templates, geoms, wt.min_width).violation
                    if wt.hard:
                        walk_hard += v
                    else:
                        walk_soft += wt.weight * v
                exact = True
            else:
                exact = False
        return Proposal(changes, new_geoms, unary, pair, hard, soft, walk_hard, walk_soft, exact)

    def commit(self, p: Proposal) -> None:
        for i, pose in p.changes.items():
            self.poses[i] = pose
            self.geoms[i] = p.geoms[i]
            self.uh[i], self.us[i] = p.unary[i]
        for (i, j), (h, s) in p.pair.items():
            self.ph[i][j] = self.ph[j][i] = h
            self.ps[i][j] = self.ps[j][i] = s
        self.hard, self.soft = p.hard, p.soft
        self.walk_hard, self.walk_soft, self.walk_exact = p.walk_hard, p.walk_soft, p.walk_exact

    def violators(self) -> list[int]:
        """Assets involved in a currently violated cheap hard term."""
        n = self.model.n
        out = []
        for i in range(n):
            if self.poses[i] is None:
                continue
            row = self.ph[i]
            if self.uh[i] > 0.0 or any(row[j] > 0.0 for j in range(n)):
                out.append(i)
        return out

    def hard_terms(self) -> list[str]:
        """Labels of currently violated hard terms (for infeasibility statistics)."""
        m = self.model
        out = []
        for i in range(m.n):
            g = self.geoms[i]
            for t in m.unary[i]:
                if t.hard and t.fn(g) > 0:
                    out.append(f"{t.cid} ({t.label})")
        for ts in m.pair.values():
            for t in ts:
                if t.hard and t.fn(self.geoms[t.a], self.geoms[t.b]) > 0:
                    out.append(f"{t.cid} ({t.label})")
        for wt in m.walkways:
            if wt.hard:
                ev = rules.walkway(m.room, m.templates, self.geoms, wt.min_width)
                if ev.violation > 0:
                    out.append(f"{wt.cid} (unreachable: {', '.join(ev.unreached)})")
        return out
