"""Automated, human-meaningful sensibility checks on a solved layout (M3a).

``check_layout(problem, layout)`` returns one named pass/fail ``CheckResult`` per check that
applies to the problem's furniture, e.g. ``coffee_table_between_seating_and_tv`` or
``desk_chair_at_desk_front[desk:desk_chair]``. They encode what a person looking at the
plan would object to, independently of the constraints the solver saw, and are the
precursor to the VLM critic. Roles come from ``rules.assign_roles`` (shared taxonomy).

Geometry is computed locally with shapely (ADR-0002: no ``roomlayout.solver`` import).
Thresholds are deliberately a bit more lenient than the rules that aim for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon

from roomlayout.schema import Asset, Layout, LayoutProblem, Mount, OpeningType, Pose2D

from .rules import Roles, assign_roles

FACING_ANGLE = 30.0  # deg, "faces" in the checks (rules aim for 20)
TV_FACING_ANGLE = 40.0
SECONDARY_FACING_ANGLE = 45.0
WALL_GAP = 0.1  # m, "against a wall"
WALL_ANGLE = 15.0
DOOR_CLEAR = 0.6  # m in front of a door / passage that must be free
LINE_OF_SIGHT_HEIGHT = 1.0  # m, assets taller than this block the view
WALKER_HEIGHT = 1.8


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# ============================================================================ geometry


class _Placed:
    def __init__(self, asset: Asset, pose: Pose2D):
        self.asset, self.pose = asset, pose
        pts = asset.effective_footprint().points
        self.poly = Polygon([pose.transform(p) for p in pts])
        self.xmin, self.xmax = min(p[0] for p in pts), max(p[0] for p in pts)
        self.ymin, self.ymax = min(p[1] for p in pts), max(p[1] for p in pts)
        self.front = pose.front
        self.z0 = asset.elevation
        self.z1 = asset.elevation + asset.dimensions.height
        back = [p for p in pts if abs(p[1] - self.ymin) < 1e-6]
        self.back = (
            pose.transform((min(p[0] for p in back), self.ymin)),
            pose.transform((max(p[0] for p in back), self.ymin)),
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.pose.x, self.pose.y)

    def local(self, p: tuple[float, float]) -> tuple[float, float]:
        dx, dy = p[0] - self.pose.x, p[1] - self.pose.y
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def angle_to(self, p: tuple[float, float]) -> float:
        """Degrees between this asset's front and the direction to ``p``."""
        vx, vy = p[0] - self.pose.x, p[1] - self.pose.y
        n = math.hypot(vx, vy)
        if n < 1e-9:
            return 0.0
        c = (self.front[0] * vx + self.front[1] * vy) / n
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _against_wall(problem: LayoutProblem, a: _Placed) -> tuple[bool, str]:
    best = (math.inf, "")
    for w in problem.room.walls:
        (sx, sy), (ex, ey) = w.start, w.end
        length = math.hypot(ex - sx, ey - sy)
        ux, uy = (ex - sx) / length, (ey - sy) / length
        nx, ny = -uy, ux

        def seg_dist(p, sx=sx, sy=sy, ux=ux, uy=uy, nx=nx, ny=ny, length=length):
            t = (p[0] - sx) * ux + (p[1] - sy) * uy
            d = (p[0] - sx) * nx + (p[1] - sy) * ny
            return math.hypot(max(0.0, -t, t - length), d)

        gap = max(seg_dist(a.back[0]), seg_dist(a.back[1]))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, a.front[0] * nx + a.front[1] * ny))))
        if gap <= WALL_GAP and ang <= WALL_ANGLE:
            return True, f"back on {w.id}"
        best = min(best, (gap, w.id))
    return False, f"nearest wall {best[1]} is {best[0]:.2f} m from the back edge"


# ============================================================================ checks


def check_layout(problem: LayoutProblem, layout: Layout) -> list[CheckResult]:
    """All checks that apply to this problem's furniture, evaluated on ``layout``."""
    poses = {p.asset_id: p.pose for p in layout.placements}
    P = {a.id: _Placed(a, poses[a.id]) for a in problem.assets if a.id in poses}
    roles = assign_roles(problem)
    out: list[CheckResult] = []
    out += _living(problem, roles, P)
    out += _work_dining(roles, P)
    out += _bedroom(problem, roles, P)
    out += _storage(problem, roles, P)
    out += _doors(problem, P)
    return out


def failed(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if not r.passed]


def _living(problem: LayoutProblem, roles: Roles, P: dict[str, _Placed]) -> list[CheckResult]:
    out = []
    seat, tv, coffee = roles.primary_seating, roles.tv, roles.coffee_table
    if seat and tv:
        s, t = P[seat], P[tv]
        a1, a2 = s.angle_to(t.center), t.angle_to(s.center)
        out.append(CheckResult(f"seating_faces_tv[{seat}:{tv}]", a1 <= FACING_ANGLE,
                               f"{a1:.0f} deg off the seat's front"))
        out.append(CheckResult(f"tv_faces_seating[{tv}:{seat}]", a2 <= TV_FACING_ANGLE,
                               f"screen turned {a2:.0f} deg away from the seat"))
        d = s.poly.distance(t.poly)
        out.append(CheckResult(f"tv_viewing_distance[{seat}:{tv}]", 1.0 <= d <= 4.5,
                               f"{d:.2f} m (want 1.0-4.5 m)"))
        sight = LineString([s.center, t.center])
        blockers = [
            i for i, q in P.items()
            if i not in (seat, tv, roles.tv_stand)
            and q.z1 > LINE_OF_SIGHT_HEIGHT and q.z0 < WALKER_HEIGHT
            and q.poly.intersects(sight)
        ]
        out.append(CheckResult(f"tv_line_of_sight[{seat}:{tv}]", not blockers,
                               "blocked by " + ", ".join(blockers) if blockers else "clear"))
    if seat and coffee:
        s, c = P[seat], P[coffee]
        if tv:
            t = P[tv]
            vx, vy = t.pose.x - s.pose.x, t.pose.y - s.pose.y
            L2 = vx * vx + vy * vy
            u = ((c.pose.x - s.pose.x) * vx + (c.pose.y - s.pose.y) * vy) / L2 if L2 else 0
            lat = abs((c.pose.x - s.pose.x) * vy - (c.pose.y - s.pose.y) * vx) / math.sqrt(L2)
            ok = 0.0 < u < 1.0 and lat <= max(0.5, (s.xmax - s.xmin) / 2)
            out.append(CheckResult(f"coffee_table_between_seating_and_tv[{coffee}]", ok,
                                   f"at {u:.2f} of the seat->TV line, {lat:.2f} m off it"))
        lx, ly = s.local(c.center)
        d = s.poly.distance(c.poly)
        ok = ly > s.ymax and abs(lx) <= (s.xmax - s.xmin) / 2 and d <= 0.8
        out.append(CheckResult(f"coffee_table_in_front_of_seating[{coffee}:{seat}]", ok,
                               f"{d:.2f} m from the seat, {lx:+.2f} m off its centre line"))
    for sec in roles.secondary_seating:
        targets = [x for x in (coffee, tv, seat) if x]
        if not targets:
            continue
        ang = min(P[sec].angle_to(P[x].center) for x in targets)
        out.append(CheckResult(f"secondary_seating_oriented[{sec}]",
                               ang <= SECONDARY_FACING_ANGLE,
                               f"{ang:.0f} deg off the nearest of {', '.join(targets)}"))
    seats = roles.secondary_seating + ([seat] if seat else [])
    for st in roles.side_tables:
        if seats:
            d = min(P[st].poly.distance(P[x].poly) for x in seats)
            out.append(CheckResult(f"side_table_by_seating[{st}]", d <= 0.3,
                                   f"{d:.2f} m from the nearest seat"))
    return out


def _work_dining(roles: Roles, P: dict[str, _Placed]) -> list[CheckResult]:
    out = []
    for d, ch in roles.desk_pairs:
        desk, chair = P[d], P[ch]
        lx, ly = desk.local(chair.center)
        ok = (desk.ymax < ly < desk.ymax + 0.7 and abs(lx) <= (desk.xmax - desk.xmin) / 2
              and chair.angle_to(desk.center) <= FACING_ANGLE)
        out.append(CheckResult(f"desk_chair_at_desk_front[{d}:{ch}]", ok,
                               f"chair at ({lx:+.2f}, {ly:+.2f}) in the desk frame"))
    for t, chairs in roles.dining:
        for ch in chairs:
            d = P[ch].poly.distance(P[t].poly)
            ang = P[ch].angle_to(P[t].center)
            out.append(CheckResult(f"dining_chair_at_table[{ch}]",
                                   d <= 0.3 and ang <= FACING_ANGLE,
                                   f"{d:.2f} m from the table, {ang:.0f} deg off"))
    return out


def _bedroom(problem: LayoutProblem, roles: Roles, P: dict[str, _Placed]) -> list[CheckResult]:
    out = []
    for b, stands in roles.beds:
        bed = P[b]
        ok, why = _against_wall(problem, bed)
        out.append(CheckResult(f"bed_head_against_wall[{b}]", ok, why))
        sides = []
        for ns in stands:
            lx, ly = bed.local(P[ns].center)
            half = (bed.xmax - bed.xmin) / 2
            beside = half - 0.05 <= abs(lx) <= half + 0.8
            head_end = ly <= bed.ymin + 0.7
            sides.append(math.copysign(1, lx))
            out.append(CheckResult(f"nightstand_beside_bed_head[{ns}]", beside and head_end,
                                   f"at ({lx:+.2f}, {ly - bed.ymin:.2f} from the headboard)"))
        if len(stands) == 2:
            out.append(CheckResult(f"nightstands_on_both_sides[{b}]", sides[0] != sides[1],
                                   "opposite sides" if sides[0] != sides[1] else "same side"))
    return out


def _storage(problem: LayoutProblem, roles: Roles, P: dict[str, _Placed]) -> list[CheckResult]:
    out = []
    for s in roles.wall_storage:
        if roles.by_id[s].subtype == "nightstand" or not P[s].asset.movable:
            continue
        ok, why = _against_wall(problem, P[s])
        out.append(CheckResult(f"storage_against_wall[{s}]", ok, why))
    tv = roles.tv
    if tv and roles.by_id[tv].category == "television" and P[tv].asset.mount != Mount.wall:
        ok, why = _against_wall(problem, P[tv])
        out.append(CheckResult(f"tv_against_wall[{tv}]", ok, why))
    return out


def _doors(problem: LayoutProblem, P: dict[str, _Placed]) -> list[CheckResult]:
    out = []
    walls = {w.id: w for w in problem.room.walls}
    for o in problem.room.openings:
        if o.type not in (OpeningType.door, OpeningType.opening):
            continue
        w = walls[o.wall_id]
        (sx, sy), (ex, ey) = w.start, w.end
        length = math.hypot(ex - sx, ey - sy)
        ux, uy = (ex - sx) / length, (ey - sy) / length
        nx, ny = -uy, ux
        t0, t1 = o.offset - o.width / 2, o.offset + o.width / 2
        rect = Polygon([
            (sx + ux * t0, sy + uy * t0),
            (sx + ux * t1, sy + uy * t1),
            (sx + ux * t1 + nx * DOOR_CLEAR, sy + uy * t1 + ny * DOOR_CLEAR),
            (sx + ux * t0 + nx * DOOR_CLEAR, sy + uy * t0 + ny * DOOR_CLEAR),
        ])
        top = o.sill_height + o.height
        blockers = [
            i for i, q in P.items() if q.z0 < top and q.poly.intersection(rect).area > 1e-4
        ]
        out.append(CheckResult(f"door_unblocked[{o.id}]", not blockers,
                               "blocked by " + ", ".join(blockers) if blockers else "clear"))
    return out


__all__ = ["CheckResult", "check_layout", "failed"]
