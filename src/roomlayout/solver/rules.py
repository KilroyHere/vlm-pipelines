"""Constraint evaluators: one function per constraint type and implicit rule.

Every evaluator returns a *continuous*, non-negative violation magnitude (0 when
satisfied) so simulated annealing gets a gradient. Units per evaluator:

========================  =======================================================
evaluator                 violation
========================  =======================================================
inside_room               m^2 of footprint outside the floor (+ m centre distance)
no_overlap                m^2 of pairwise footprint intersection
clear_of_obstacles        m^2 of footprint inside obstacles
clearance_zones           m^2 of other footprints / walls / obstacles in zones
door_swing                m^2 of footprint inside interior door-swing discs
against_wall / wall_mount m of gap beyond max_gap + rad beyond 10 deg + m on openings
faces, parallel           rad beyond the allowed angle
near, distance_at_least   m beyond / short of the threshold
keep_clear                m^2 of footprint in the keep-clear rectangle
fixed (soft only)         m of displacement + rad of rotation from ``initial``
walkway_width             sum over unreachable targets of (0.1 + m short), see below
========================  =======================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from .geometry import (
    AssetTemplate,
    Posed,
    RoomGeom,
    WallGeom,
    angle_between,
    bboxes_overlap,
    clean_area,
    overlap_area,
    wrap_angle,
)

WALL_ANGLE_TOL = math.radians(10.0)  # against_wall: front within 10 deg of wall normal
WALL_MOUNT_GAP = 0.05  # implicit wall_mount uses the AgainstWall default gap
WALKWAY_EPS = 0.05  # m of slack when deciding whether a target is reachable
WALKWAY_TARGET_BASE = 0.1  # violation floor per unreachable target
WALKWAY_TARGET_CAP = 3.0  # cap on the per-target distance term
FRONT_TARGET_OFFSET = 0.02  # front-edge target sits this far in front of the asset


# --------------------------------------------------------------------------- implicit


def inside_room(room: RoomGeom, g: Posed) -> float:
    v = room.outside_area(g.fp, g.bounds)
    if v > 0.0:
        v += room.distance_to_floor(g.x, g.y)
    return v


def clear_of_obstacles(room: RoomGeom, g: Posed) -> float:
    return sum(overlap_area(g.fp, g.bounds, ob.poly, ob.bounds) for ob in room.obstacles)


def own_zones_blocked(room: RoomGeom, g: Posed) -> float:
    """Unary part of ``implicit:clearance_zones``: an asset's own zones may not overlap a
    fixed obstacle, and (per ``ZONE_WALL_POLICY``, default "front") the part of a zone at
    or in front of the asset's back-edge line may not leave the floor, i.e. be cut off by
    a wall. Zone parts behind the back line (through the wall the asset stands against)
    are clipped away."""
    v = 0.0
    for z, zb in g.zone_wall:
        v += room.outside_area(z, zb)
    for z, zb in zip(g.zones, g.zone_bounds, strict=True):
        for ob in room.obstacles:
            v += overlap_area(z, zb, ob.poly, ob.bounds)
    return v


def zones_pair(ga: Posed, gb: Posed) -> float:
    """Pair part of ``implicit:clearance_zones``: a's zones vs b's footprint and vice versa,
    skipping zones whose ``allowed_categories`` contain the other asset's category. Zones
    may overlap other zones. (The contract clips zones to the floor first; since footprints
    must lie inside the floor, zone-in-floor ∩ footprint equals zone ∩ footprint for every
    feasible layout, so the clip is skipped here.)"""
    v = 0.0
    for z, zb, allowed in zip(ga.zones, ga.zone_bounds, ga.zone_allowed, strict=True):
        if gb.category not in allowed:
            v += overlap_area(z, zb, gb.fp, gb.bounds)
    for z, zb, allowed in zip(gb.zones, gb.zone_bounds, gb.zone_allowed, strict=True):
        if ga.category not in allowed:
            v += overlap_area(z, zb, ga.fp, ga.bounds)
    return v


def no_overlap(ga: Posed, gb: Posed) -> float:
    return overlap_area(ga.fp, ga.bounds, gb.fp, gb.bounds)


def door_swing(room: RoomGeom, g: Posed) -> float:
    return sum(overlap_area(g.fp, g.bounds, poly, b) for _, poly, b in room.swings)


# --------------------------------------------------------------------------- vocabulary


def against_wall(tmpl: AssetTemplate, g: Posed, walls: list[WallGeom], max_gap: float) -> float:
    """Best (minimum) violation over the candidate walls. For one wall it is
    gap excess + angle excess + length of the back edge lying across a door/passage
    (or a window, for assets taller than the sill / wall-mounted)."""
    best = math.inf
    (b0x, b0y), (b1x, b1y) = g.back
    for w in walls:
        gap = max(w.segment_distance(b0x, b0y), w.segment_distance(b1x, b1y))
        ang = angle_between(g.fx, g.fy, w.nx, w.ny)
        v = max(0.0, gap - max_gap) + max(0.0, ang - WALL_ANGLE_TOL)
        if v >= best:
            continue
        spans = w.blocked_spans(tmpl.height, tmpl.wall_mounted)
        if spans:
            ta, _ = w.to_local(b0x, b0y)
            tb, _ = w.to_local(b1x, b1y)
            lo, hi = min(ta, tb), max(ta, tb)
            for s0, s1 in spans:
                v += max(0.0, min(hi, s1) - max(lo, s0))
        best = min(best, v)
    return best if best > 1e-9 else 0.0


def faces_angle(ga: Posed, gb: Posed) -> float:
    return angle_between(ga.fx, ga.fy, gb.x - ga.x, gb.y - ga.y)


def faces(ga: Posed, gb: Posed, max_angle: float) -> float:
    v = faces_angle(ga, gb) - max_angle
    return v if v > 1e-9 else 0.0


def footprint_distance(ga: Posed, gb: Posed) -> float:
    if bboxes_overlap(ga.bounds, gb.bounds) and ga.fp.intersects(gb.fp):
        return 0.0
    return ga.fp.distance(gb.fp)


def near(ga: Posed, gb: Posed, max_distance: float) -> float:
    v = footprint_distance(ga, gb) - max_distance
    return v if v > 1e-9 else 0.0


def distance_at_least(ga: Posed, gb: Posed, min_distance: float) -> float:
    v = min_distance - footprint_distance(ga, gb)
    return v if v > 1e-9 else 0.0


def parallel_angle(ga: Posed, gb: Posed) -> float:
    d = abs(wrap_angle(ga.yaw - gb.yaw))
    return min(d, math.pi - d)


def parallel(ga: Posed, gb: Posed, max_angle: float) -> float:
    v = parallel_angle(ga, gb) - max_angle
    return v if v > 1e-9 else 0.0


def keep_clear(rect: BaseGeometry, rect_bounds, g: Posed) -> float:
    return overlap_area(g.fp, g.bounds, rect, rect_bounds)


def fixed_offset(g: Posed, pose0: tuple[float, float, float]) -> float:
    v = math.hypot(g.x - pose0[0], g.y - pose0[1]) + abs(wrap_angle(g.yaw - pose0[2]))
    return v if v > 1e-9 else 0.0


# --------------------------------------------------------------------------- walkway


@dataclass
class WalkwayTarget:
    label: str
    geom: BaseGeometry
    kind: str  # "opening" | "zone" (access zones) | "front" (front edge, no access zone)


@dataclass
class WalkwayEval:
    violation: float
    unreached: list[str]
    free: BaseGeometry  # eroded free space (centres a walker of min_width can occupy)


def walkway_targets(
    room: RoomGeom, templates: list[AssetTemplate], geoms: list[Posed]
) -> list[WalkwayTarget]:
    """Doors/passages (centre of the opening on the wall line) and, per floor asset, its
    access-purpose clearance zones, or else its front edge (just in front of it).
    Wall-mounted assets are not walked up to and get no target."""
    targets = [
        WalkwayTarget(o.id, Point(o.center), "opening")
        for o in room.openings.values()
        if o.type.value in ("door", "opening")
    ]
    for t, g in zip(templates, geoms, strict=True):
        if t.wall_mounted:
            continue
        access = [z for z, p in zip(g.zones, g.zone_purposes, strict=True) if p == "access"]
        if access:
            geom = access[0] if len(access) == 1 else shapely.union_all(access)
            kind = "zone"
        else:
            kind = "front"
            off = FRONT_TARGET_OFFSET
            fe = g.front_edge + np.array([g.fx * off, g.fy * off])
            geom = LineString(fe) if np.ptp(fe[:, 0]) + np.ptp(fe[:, 1]) > 1e-9 else Point(fe[0])
        targets.append(WalkwayTarget(t.id, geom, kind))
    return targets


def walkway(
    room: RoomGeom,
    templates: list[AssetTemplate],
    geoms: list[Posed],
    min_width: float,
) -> WalkwayEval:
    """Free floor = floor - footprints - obstacles, eroded by r = min_width / 2, computed
    as ``erode(floor, r) - union(dilate(footprint, r), dilate(obstacle, r))`` (dilations are
    cached per pose). A target is *reached* by a connected component C of the eroded free
    space if ``distance(target, C) <= r + WALKWAY_EPS``, i.e. a walker of width min_width
    standing in C touches the target. The component reaching the most targets (ties:
    largest area) is the walkway. One level of chaining: a front-edge target (asset
    without an access zone, e.g. a nightstand) also counts as reached if it lies within
    ``WALKWAY_EPS`` of a *reached* access zone of another asset (e.g. the bed's side
    zone), since access zones are themselves guaranteed-free standing space.
    Each target still missed adds ``0.1 + min(distance - r - eps, 3)``."""
    r = min_width / 2
    targets = walkway_targets(room, templates, geoms)
    blockers = [g.buffered(r) for g in geoms] + room.obstacle_buffers(r)
    floor_er = room.eroded_floor(r)
    free = floor_er.difference(shapely.union_all(blockers)) if blockers else floor_er
    if not targets:
        return WalkwayEval(0.0, [], free)
    comps = [c for c in getattr(free, "geoms", [free]) if not c.is_empty]
    if not comps:
        return WalkwayEval(
            len(targets) * (WALKWAY_TARGET_BASE + WALKWAY_TARGET_CAP),
            [t.label for t in targets],
            free,
        )
    reach = r + WALKWAY_EPS
    dist = shapely.distance(
        np.array(comps, dtype=object)[:, None], np.array([t.geom for t in targets])[None, :]
    )
    reached = dist <= reach
    counts = reached.sum(axis=1)
    best = max(range(len(comps)), key=lambda i: (counts[i], comps[i].area))
    row = reached[best].copy()
    reached_zones = [t.geom for j, t in enumerate(targets) if row[j] and t.kind == "zone"]
    if reached_zones:
        for j, t in enumerate(targets):
            if not row[j] and t.kind == "front":
                row[j] = any(t.geom.distance(z) <= WALKWAY_EPS for z in reached_zones)
    v = 0.0
    unreached = []
    for j, t in enumerate(targets):
        if not row[j]:
            v += WALKWAY_TARGET_BASE + min(float(dist[best, j]) - reach, WALKWAY_TARGET_CAP)
            unreached.append(t.label)
    return WalkwayEval(v, unreached, free)


__all__ = [
    "against_wall",
    "clean_area",
    "clear_of_obstacles",
    "distance_at_least",
    "door_swing",
    "faces",
    "fixed_offset",
    "inside_room",
    "keep_clear",
    "near",
    "no_overlap",
    "own_zones_blocked",
    "parallel",
    "walkway",
    "zones_pair",
]
