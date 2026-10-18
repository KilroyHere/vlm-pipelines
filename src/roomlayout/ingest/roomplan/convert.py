"""CapturedRoom -> ``LayoutProblem`` skeleton (room, box assets, initial poses).

Mapping (details and evidence in docs/ingest.md):

- **Frame.** ARKit Y-up -> ours Z-up via :func:`geometry.to_ours` (ADR-0001), here only.
  World xy is kept as-is (no re-centring), so poses map straight back to the capture's
  camera frames. Heights are measured from the floor surface (or the lowest wall bottom).
- **Floor and walls.** Floor polygon from :mod:`walls`. Every floor edge becomes a
  ``Wall`` (``wall_1``... from the south-west corner, CCW) with the height of the RoomPlan
  wall along it. Floor boundary with no RoomPlan wall along it (open-plan sides) becomes
  a ``passage`` opening. Walls with floor on both sides become ``partition_n`` obstacles.
- **Doors, windows, openings** go onto the floor edge nearest their centre (their
  ``parentIdentifier`` wall when present), with offset, width, height and sill. RoomPlan
  doesn't say which way a door swings: we assume into the room, hinge at the wall start,
  which keeps the larger area clear.
- **Objects.** Local +Z is the front (verified on public exports, docs/ingest.md),
  local x the width, y the height. Furniture becomes movable box assets; built-in
  fixtures (kitchen and bath appliances, fireplace) become immovable assets so requests
  can still name them; stairs become obstacles. Attributes refine the subtype
  (ChairType, TableType, StorageType, SofaType). An ``lShapedExtension`` is merged into
  the L-shaped sofa it touches (one asset, L footprint). A cabinet with a TV standing on
  it becomes a ``tv_stand``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from roomlayout.schema import Asset, Dimensions, LayoutProblem
from roomlayout.schema.asset import AssetSource, Mount
from roomlayout.schema.geometry import Polygon2D, Pose2D
from roomlayout.schema.room import DoorSwing, Obstacle, Opening, OpeningType, Room, Wall

from .format import Capture, Element, IngestError, parse_capture, select_room
from .geometry import (
    oriented_rect,
    overlap_area,
    point_in_polygon,
    point_segment_distance,
    rectilinear_union,
    to_ours,
    wrap_angle,
    yaw_from_front,
)
from .walls import (
    WallSeg,
    floor_from_surface,
    floor_from_walls,
    floor_polygon_order,
    orient_and_order,
    surface_segment,
)

FIXTURES = {"refrigerator", "stove", "oven", "dishwasher", "sink", "washerDryer", "toilet",
            "bathtub", "fireplace"}  # fmt: skip
OBSTACLE_CATEGORIES = {"stairs"}
OPENING_TYPES = {"doors": OpeningType.door, "windows": OpeningType.window,
                 "openings": OpeningType.opening}  # fmt: skip
SUBTYPES = {
    ("ChairType", "dining"): "dining_chair",
    ("ChairType", "swivel"): "office_chair",
    ("ChairType", "stool"): "stool",
    ("TableType", "coffee"): "coffee_table",
    ("TableType", "dining"): "dining_table",
    ("StorageType", "shelf"): "bookshelf",
    ("StorageType", "cabinet"): "cabinet",
    ("SofaType", "lShaped"): "sectional",
}
MIN_PASSAGE = 0.8  # m: narrower unwalled stretches are scan gaps: no walker fits (WALKWAY_MIN)
ON_FLOOR = 0.15  # m: objects whose bottom is lower than this stand on the floor
TV_WALL_HEIGHT = 0.6  # m: a TV this high with nothing under it is wall-mounted
STACK_TOL = 0.2  # m: TV bottom vs. top of the furniture under it
TV_WALL_GAP = 0.3  # m: a raised TV counts as wall-mounted only with its back this close
MAX_SWING_DOOR = 1.2  # m: wider doors are sliding/double/patio doors: no single swing arc
WALL_UNIT_HEIGHT = 0.8  # m: storage hung this high is a wall cabinet/shelf (fixed)
KITCHEN_ANCHORS = {"oven", "stove", "sink", "dishwasher", "refrigerator"}
RUN_GAP = 0.1  # m: storage this close to an appliance (or the run) is built-in cabinetry


@dataclass
class IngestResult:
    problem: LayoutProblem
    notes: list[str] = field(default_factory=list)  # what we did, for the curious
    warnings: list[str] = field(default_factory=list)  # what the user should check
    source_ids: dict[str, str] = field(default_factory=dict)  # our id -> RoomPlan identifier
    floor_z: float = 0.0  # floor height in our frame (= ARKit y)

    def report(self) -> dict:
        return {
            "problem_id": self.problem.id,
            "walls": len(self.problem.room.walls),
            "openings": len(self.problem.room.openings),
            "obstacles": len(self.problem.room.obstacles),
            "assets": len(self.problem.assets),
            "floor_z": round(self.floor_z, 4),
            "notes": self.notes,
            "warnings": self.warnings,
            "source_ids": self.source_ids,
        }


def convert(data: dict, *, room: int | None = None, problem_id: str = "roomplan") -> IngestResult:
    """RoomPlan ``CapturedRoom`` (or ``CapturedStructure`` + ``room``) JSON -> problem."""
    raw, notes = select_room(data, room)
    cap = parse_capture(raw)
    res = IngestResult(problem=None, notes=notes)  # type: ignore[arg-type]
    res.warnings.extend(cap.skipped)
    if cap.version not in (None, 2):
        res.warnings.append(f"CapturedRoom version {cap.version!r}; verified against 2 only")

    floor_z = _floor_z(cap)
    res.floor_z = floor_z
    segs = [_wall_seg(w, floor_z) for w in cap.elements["walls"]]
    floor = floor_from_surface(cap.elements["floors"], res.notes, res.warnings)
    if floor is None:
        floor = floor_from_walls(segs, res.notes, res.warnings)
    if floor is None:
        raise IngestError("no floor polygon and too few walls to build one")
    floor = floor_polygon_order(floor)
    segs = orient_and_order(segs, floor, res.warnings)
    heights = [s.height for s in segs if not s.interior] or [2.5]
    ceiling = float(np.median(heights))

    walls = _edge_walls(floor, segs, ceiling)
    openings = _openings(cap, walls, segs, floor_z, res)
    openings += _passages(walls, segs, openings, res)
    obstacles = _partitions(cap, segs, res)
    if obstacles:
        res.warnings.append(
            f"{sum(s.interior for s in segs)} wall(s) stand inside the room: this may be "
            "several rooms scanned together. v1 plans one room; scan rooms separately for "
            "the best results"
        )
    assets, initial, more_obstacles = _objects(cap, floor_z, res, floor)
    obstacles += more_obstacles

    room_model = Room(
        id=problem_id,
        floor=_poly(floor),
        ceiling_height=round(max(ceiling, 2.0), 3),
        walls=walls,
        openings=openings,
        obstacles=obstacles,
    )
    res.problem = LayoutProblem(id=problem_id, room=room_model, assets=assets, initial=initial)
    return res


# ------------------------------------------------------------------------------ helpers


def _poly(pts) -> Polygon2D:
    return Polygon2D(points=[(round(float(x), 4), round(float(y), 4)) for x, y in pts])


def _seg_yaw(s: WallSeg) -> float:
    d = s.b - s.a
    return math.atan2(d[1], d[0])


def _floor_z(cap: Capture) -> float:
    floors = cap.elements["floors"]
    if floors:
        return float(np.median([to_ours(f.center)[2] for f in floors]))
    bottoms = [to_ours(w.center)[2] - w.dimensions[1] / 2 for w in cap.elements["walls"]]
    return float(min(bottoms)) if bottoms else 0.0


def _wall_seg(e: Element, floor_z: float) -> WallSeg:
    a, b = surface_segment(e)
    top = to_ours(e.center)[2] + e.dimensions[1] / 2
    return WallSeg(source=e, a=a, b=b, height=float(max(top - floor_z, 0.5)))


def _edge_walls(floor: np.ndarray, segs: list[WallSeg], default_h: float) -> list[Wall]:
    """One wall per floor edge, with the height of the RoomPlan wall running along it."""
    walls = []
    n = len(floor)
    for i in range(n):
        a, b = floor[i], floor[(i + 1) % n]
        along = [s for s in segs if not s.interior and _covered(s, a, b) > 0.0]
        h = max((s.height for s in along), key=float, default=default_h)
        walls.append(Wall(id=f"wall_{i + 1}", start=tuple(map(float, a)),
                          end=tuple(map(float, b)), height=round(h, 3)))  # fmt: skip
    return walls


def _covered(s: WallSeg, a, b, tol: float = 0.15) -> float:
    """Length of floor edge a-b that RoomPlan wall ``s`` runs along."""
    length = float(np.linalg.norm(b - a))
    if length < 1e-9:
        return 0.0
    da, _ = point_segment_distance(s.a, a, b)
    db, _ = point_segment_distance(s.b, a, b)
    if da > tol or db > tol:
        return 0.0
    u = (b - a) / length
    t0, t1 = sorted((float((s.a - a) @ u), float((s.b - a) @ u)))
    return max(0.0, min(length, t1) - max(0.0, t0))


def _partitions(cap: Capture, segs: list[WallSeg], res: IngestResult) -> list[Obstacle]:
    """Walls with floor on both sides -> thin obstacles, with gaps at their doors."""
    out: list[Obstacle] = []
    passable = cap.elements["doors"] + cap.elements["openings"]
    for s in (x for x in segs if x.interior):
        u = (s.b - s.a) / s.length
        gaps = []
        for e in passable:
            c = to_ours(e.center)[:2]
            dist, t = point_segment_distance(c, s.a, s.b)
            if e.parent == s.source.identifier or dist < 0.2:
                mid = t * s.length
                gaps.append((mid - e.dimensions[0] / 2, mid + e.dimensions[0] / 2))
        pieces, start = [], 0.0
        for g0, g1 in sorted(gaps):
            if g0 > start:
                pieces.append((start, g0))
            start = max(start, g1)
        if start < s.length:
            pieces.append((start, s.length))
        if gaps:
            res.notes.append(f"partition wall {s.source.identifier[:8]}: {len(gaps)} door(s) "
                             "left as gaps")  # fmt: skip
        for p0, p1 in pieces:
            if p1 - p0 < 0.05:
                continue
            mid = s.a + u * (p0 + p1) / 2
            rect = oriented_rect(*mid, _seg_yaw(s), p1 - p0, 0.1)
            out.append(Obstacle(id=f"partition_{len(out) + 1}", label="partition wall",
                                footprint=_poly(rect), height=s.height))  # fmt: skip
    return out


def _place_on_wall(center_xy, width, walls: list[Wall], prefer=None):
    """(wall, offset, width) for an opening centred at ``center_xy`` (ours, xy)."""
    best = None
    for w in walls:
        a, b = np.array(w.start), np.array(w.end)
        d, t = point_segment_distance(center_xy, a, b)
        bonus = 0.0 if prefer is None or not _parallel_near(prefer, a, b) else -0.5
        if best is None or d + bonus < best[0]:
            best = (d + bonus, w, t, d)
    _, wall, t, dist = best
    length = wall.length
    width = min(width, length)
    offset = min(max(t * length, width / 2), length - width / 2)
    return wall, offset, width, dist


def _fit(offset: float, width: float, length: float) -> tuple[float, float]:
    """(offset, width) rounded to 0.1 mm without leaving a wall of ``length``."""
    width = math.floor(min(width, length) * 1e4) / 1e4
    offset = round(offset, 4)
    offset = min(max(offset, width / 2), length - width / 2)
    return offset, width


def _parallel_near(seg: WallSeg, a, b) -> bool:
    return _covered(seg, a, b) > 0.0


def _openings(cap: Capture, walls: list[Wall], segs: list[WallSeg], floor_z: float,
              res: IngestResult) -> list[Opening]:  # fmt: skip
    by_source = {s.source.identifier: s for s in segs}
    out: list[Opening] = []
    counts: Counter = Counter()
    for kind, otype in OPENING_TYPES.items():
        for e in cap.elements[kind]:
            parent = by_source.get(e.parent) if e.parent else None
            if parent is not None and parent.interior:
                continue  # a gap in the partition obstacle (see _partitions)
            c = to_ours(e.center)
            wall, offset, width, dist = _place_on_wall(c[:2], float(e.dimensions[0]), walls, parent)
            if dist > 0.3:
                res.warnings.append(
                    f"{kind[:-1]} {e.identifier[:8]} is {dist:.2f} m from the nearest wall; "
                    f"placed on {wall.id}"
                )
            if width < e.dimensions[0] - 1e-6:
                res.warnings.append(
                    f"{kind[:-1]} {e.identifier[:8]} ({e.dimensions[0]:.2f} m) is wider than "
                    f"{wall.id}; clipped to {width:.2f} m"
                )
            height = float(e.dimensions[1])
            sill = max(0.0, float(c[2] - height / 2 - floor_z))
            if otype != OpeningType.window and sill < 0.1:
                sill = 0.0
            offset, width = _fit(offset, width, wall.length)
            counts[otype] += 1
            oid = f"{'passage' if otype == OpeningType.opening else otype.value}_{counts[otype]}"
            out.append(Opening(
                id=oid, type=otype, wall_id=wall.id, offset=offset, width=width,
                height=round(height, 4), sill_height=round(sill, 4),
                swing=DoorSwing() if otype == OpeningType.door and width <= MAX_SWING_DOOR
                else None,
            ))  # fmt: skip
            res.source_ids[oid] = e.identifier
    if counts[OpeningType.door]:
        res.notes.append("door swing assumed into the room, hinge at the wall start (unknown)")
    wide = [o.id for o in out if o.type == OpeningType.door and o.swing is None]
    if wide:
        res.notes.append(f"doors wider than {MAX_SWING_DOOR} m treated as sliding/double "
                         f"(no swing arc): {', '.join(wide)}")  # fmt: skip
    return out


def _passages(walls: list[Wall], segs: list[WallSeg], existing: list[Opening],
              res: IngestResult) -> list[Opening]:  # fmt: skip
    """Unwalled stretches of the floor boundary become passages (open-plan sides)."""
    out = []
    n = sum(o.type == OpeningType.opening for o in existing)
    for w in walls:
        a, b = np.array(w.start), np.array(w.end)
        length = w.length
        u = (b - a) / length
        spans = []
        for s in segs:
            if s.interior or _covered(s, a, b) <= 0:
                continue
            t0, t1 = sorted((float((s.a - a) @ u), float((s.b - a) @ u)))
            spans.append((max(0.0, t0), min(length, t1)))
        spans += [(o.offset - o.width / 2, o.offset + o.width / 2) for o in existing
                  if o.wall_id == w.id]  # fmt: skip
        gaps, end = [], 0.0
        for s0, s1 in sorted(spans):
            if s0 > end:
                gaps.append((end, s0))
            end = max(end, s1)
        if end < length:
            gaps.append((end, length))
        for g0, g1 in gaps:
            if g1 - g0 >= MIN_PASSAGE:
                n += 1
                offset, width = _fit((g0 + g1) / 2, g1 - g0, length)
                out.append(Opening(id=f"passage_{n}", type=OpeningType.opening, wall_id=w.id,
                                   offset=offset, width=width, height=w.height))  # fmt: skip
                res.notes.append(
                    f"{w.id}: {g1 - g0:.2f} m with no wall scanned; treated as an open passage"
                )
    return out


def _subtype(e: Element) -> str | None:
    for key, value in e.attributes.items():
        sub = SUBTYPES.get((key, str(value)))
        if sub:
            return sub
    return None


@dataclass
class _Obj:
    e: Element
    category: str
    subtype: str | None
    center: np.ndarray  # (2,) ours
    yaw: float
    width: float
    depth: float
    height: float
    bottom: float  # above the floor
    footprint: np.ndarray | None = None  # local, when not a plain rectangle
    label: str = ""
    fixed: bool = False  # immovable for the solver (fixtures, built-in cabinetry)

    @property
    def rect(self) -> np.ndarray:
        return oriented_rect(*self.center, self.yaw, self.width, self.depth)


def _pull_inside(o: _Obj, floor: np.ndarray, res: IngestResult) -> bool:
    """Move an object whose centre is outside the floor just inside the nearest edge.

    Returns False when the object belongs to the next room (outside, facing away from
    this one: seen through a door or passage) and should be dropped.
    """
    if point_in_polygon(o.center, floor):
        return True
    n = len(floor)
    dist, k = min(
        (point_segment_distance(o.center, floor[i], floor[(i + 1) % n])[0], i) for i in range(n)
    )
    a, b = floor[k], floor[(k + 1) % n]
    d = (b - a) / np.linalg.norm(b - a)
    inward = np.array([-d[1], d[0]])  # floor is CCW: interior on the left
    front = np.array([-math.sin(o.yaw), math.cos(o.yaw)])
    if dist > 0.05 and front @ inward < -0.5:
        res.warnings.append(
            f"{o.category} {o.e.identifier[:8]} is {dist:.2f} m outside the floor outline, "
            "facing away from the room: it's in the next room; dropped"
        )
        return False
    _, t = point_segment_distance(o.center, a, b)
    foot = a + t * (b - a)
    reach = abs(float(np.array([math.sin(o.yaw), -math.cos(o.yaw)]) @ inward)) * o.depth / 2
    reach += abs(float(np.array([math.cos(o.yaw), math.sin(o.yaw)]) @ inward)) * o.width / 2
    o.center = foot + inward * (reach + 0.01)
    res.warnings.append(
        f"{o.category} {o.e.identifier[:8]} was {dist:.2f} m outside the floor outline "
        "(scan error?); moved just inside the nearest wall"
    )
    return True


def _objects(cap: Capture, floor_z: float, res: IngestResult, floor: np.ndarray):
    objs: list[_Obj] = []
    obstacles: list[Obstacle] = []
    for e in cap.elements["objects"]:
        c = to_ours(e.center)
        front = to_ours(e.rotation[:, 2])
        if abs(to_ours(e.rotation[:, 1])[2]) < 0.98:
            res.warnings.append(f"{e.category} {e.identifier[:8]} is tilted; yaw is approximate")
        w, h, d = (float(v) for v in e.dimensions)
        bottom = float(c[2] - h / 2 - floor_z)
        o = _Obj(e, e.category, _subtype(e), c[:2], yaw_from_front(front[:2]), w, d, h,
                 bottom if bottom > ON_FLOOR else 0.0)  # fmt: skip
        if e.category in OBSTACLE_CATEGORIES:
            obstacles.append(Obstacle(id=f"{e.category}_{len(obstacles) + 1}", label=e.category,
                                      footprint=_poly(o.rect), height=h))  # fmt: skip
            res.source_ids[obstacles[-1].id] = e.identifier
            continue
        objs.append(o)

    objs = _merge_l_sofas(objs, res)
    objs = [o for o in objs if _pull_inside(o, floor, res)]
    for o in objs:
        o.fixed = o.category in FIXTURES
    _mark_built_ins(objs, res)
    _mark_tv_stands(objs, res)

    counts: Counter = Counter()
    assets, initial = [], {}
    for o in objs:
        key = o.subtype or o.category
        counts[key] += 1
        aid = f"{key}_{counts[key]}"
        mount = Mount.wall if o.label == "wall cabinet" else Mount.floor
        if o.category == "television" and o.bottom > TV_WALL_HEIGHT and not o.label:
            back = o.center - np.array([-math.sin(o.yaw), math.cos(o.yaw)]) * o.depth / 2
            n = len(floor)
            gap = min(point_segment_distance(back, floor[i], floor[(i + 1) % n])[0]
                      for i in range(n))  # fmt: skip
            if gap <= TV_WALL_GAP:
                mount = Mount.wall
            else:
                res.notes.append(f"raised TV {o.e.identifier[:8]} is {gap:.2f} m from a wall: "
                                 "standing on unscanned furniture, not wall-mounted")  # fmt: skip
        sofa_type = o.e.attributes.get("SofaType")
        label = o.label or (f"{sofa_type} sofa" if sofa_type == "singleSeat" else "")
        assets.append(Asset(
            id=aid, category=o.category, subtype=o.subtype, label=label,
            dimensions=Dimensions(width=round(o.width, 4), depth=round(o.depth, 4),
                                  height=round(o.height, 4)),
            footprint=_poly(o.footprint) if o.footprint is not None else None,
            mount=mount, elevation=round(o.bottom, 4), movable=not o.fixed,
            source=AssetSource(dimensions_from="roomplan", geometry_from="box"),
        ))  # fmt: skip
        initial[aid] = Pose2D(x=round(float(o.center[0]), 4), y=round(float(o.center[1]), 4),
                              yaw=round(o.yaw, 5))  # fmt: skip
        res.source_ids[aid] = o.e.identifier
    fixed = [a.id for a in assets if not a.movable]
    if fixed:
        res.notes.append(f"kept in place (fixtures and built-ins): {', '.join(fixed)}")
    return assets, initial, obstacles


def _merge_l_sofas(objs: list[_Obj], res: IngestResult) -> list[_Obj]:
    """Fold ``lShapedExtension`` pieces into the sofa they touch (L or U footprint)."""
    exts = [o for o in objs if o.e.attributes.get("SofaType") == "lShapedExtension"]
    if not exts:
        return objs
    keep = [o for o in objs if o not in exts]
    hosts = [o for o in keep if o.category == "sofa"]
    attached: dict[int, list[tuple[float, float, float, float]]] = {}
    for x in exts:
        best = None
        for h in hosts:
            box = _square_box(h, x)
            if box is not None:
                d = float(np.linalg.norm(h.center - x.center))
                if best is None or d < best[0]:
                    best = (d, h, box)
        if best is None:
            res.warnings.append(
                f"sofa extension {x.e.identifier[:8]} doesn't attach squarely to a sofa; kept "
                "as a separate sofa"
            )
            x.label = "sofa extension"
            keep.append(x)
            continue
        _, host, box = best
        attached.setdefault(id(host), []).append(box)
        host.height = max(host.height, x.height)
        res.notes.append(f"merged sofa extension {x.e.identifier[:8]} into a sectional sofa")
    for host in hosts:
        boxes = attached.get(id(host))
        if not boxes:
            continue
        hw, hd = host.width / 2, host.depth / 2
        outline = rectilinear_union([(-hw, hw, -hd, hd), *boxes])
        if outline is None:
            res.warnings.append(f"sofa {host.e.identifier[:8]}: extensions don't form one "
                                "outline; kept the main piece's rectangle")  # fmt: skip
            continue
        bx0, by0 = outline.min(axis=0)
        bx1, by1 = outline.max(axis=0)
        shift_local = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
        c, s = math.cos(host.yaw), math.sin(host.yaw)
        host.center = host.center + np.array([[c, -s], [s, c]]) @ shift_local
        host.footprint = outline - shift_local
        host.width, host.depth = float(bx1 - bx0), float(by1 - by0)
        host.subtype = "sectional"
    return keep


def _square_box(host: _Obj, ext: _Obj, max_gap: float = 0.1):
    """``ext`` as an axis-aligned box in ``host``'s local frame, if square to it and touching."""
    rel = wrap_angle(ext.yaw - host.yaw)
    k = round(rel / (math.pi / 2))
    if abs(rel - k * math.pi / 2) > math.radians(8):
        return None
    c, s = math.cos(-host.yaw), math.sin(-host.yaw)
    local = (ext.rect - host.center) @ np.array([[c, -s], [s, c]]).T
    x0, y0 = local.min(axis=0)
    x1, y1 = local.max(axis=0)
    hw, hd = host.width / 2, host.depth / 2
    if max(x0 - hw, -hw - x1, y0 - hd, -hd - y1) > max_gap:
        return None

    def snap(v: float, edges: tuple[float, float]) -> float:
        # RoomPlan's pieces don't quite meet (cm gaps/overhangs): snap to the host's edges
        near = min(edges, key=lambda e: abs(v - e))
        return near if abs(v - near) <= max_gap else v

    xs, ys = (-hw, hw), (-hd, hd)
    return (snap(float(x0), xs), snap(float(x1), xs), snap(float(y0), ys), snap(float(y1), ys))


def _mark_built_ins(objs: list[_Obj], res: IngestResult) -> None:
    """Wall-hung storage and kitchen cabinet runs are built in: fixed for the solver."""
    storage = [o for o in objs if o.category == "storage"]
    for o in storage:
        if o.bottom > WALL_UNIT_HEIGHT:
            o.fixed, o.label = True, "wall cabinet"
    run = [o for o in objs if o.category in KITCHEN_ANCHORS]
    frontier = list(run)
    while frontier:
        a = frontier.pop()
        grown = oriented_rect(*a.center, a.yaw, a.width + 2 * RUN_GAP, a.depth + 2 * RUN_GAP)
        for o in storage:
            if o in run or o.bottom > WALL_UNIT_HEIGHT:
                continue
            if overlap_area(grown, o.rect) > 1e-4:
                o.fixed, o.label = True, "built-in cabinet"
                run.append(o)
                frontier.append(o)
    built = [o for o in storage if o.fixed]
    if built:
        res.notes.append(
            f"{len(built)} storage unit(s) are wall-hung or part of a kitchen run: kept in "
            "place as built-ins"
        )


def _mark_tv_stands(objs: list[_Obj], res: IngestResult) -> None:
    for tv in (o for o in objs if o.category == "television" and o.bottom > ON_FLOOR):
        under = [
            o for o in objs
            if o.category in ("storage", "table") and o is not tv
            and abs(o.bottom + o.height - tv.bottom) < STACK_TOL
            and overlap_area(o.rect, tv.rect) > 0.25 * tv.width * tv.depth
        ]  # fmt: skip
        if not under:
            continue
        stand = max(under, key=lambda o: overlap_area(o.rect, tv.rect))
        tv.label = "on furniture"
        if stand.category == "storage" and stand.subtype in (None, "cabinet"):
            stand.subtype = "tv_stand"
            res.notes.append(f"storage {stand.e.identifier[:8]} carries the TV: tv_stand")
