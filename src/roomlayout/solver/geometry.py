"""World-frame geometry for the solver.

Everything here is derived from the contract (``roomlayout.schema``) once per problem:

- :class:`RoomGeom`: floor polygon (prepared), walls with interior normals, openings
  (door swing quarter discs, keep-clear rectangles), obstacles.
- :class:`AssetTemplate`: an asset's local-frame footprint/clearance zones as arrays,
  plus a per-template cache of :class:`Posed` world geometry keyed by ``(x, y, yaw)``.

Frames follow ADR-0001: meters, yaw CCW about +Z, asset front = local +Y,
``front = (-sin yaw, cos yaw)``. Walls run CCW, interior on the left of start->end.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from roomlayout.schema import Asset, Mount, OpeningType, Room

AREA_EPS = 1e-6  # m^2; overlaps smaller than this count as touching
HALF_PI = math.pi / 2
TWO_PI = 2 * math.pi
BUFFER_QUAD_SEGS = 4  # resolution of round buffers (walkway erosion)
_CACHE_LIMIT = 8192

Bounds = tuple[float, float, float, float]


# --------------------------------------------------------------------------- angles


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % TWO_PI - math.pi


def front_vector(yaw: float) -> tuple[float, float]:
    return (-math.sin(yaw), math.cos(yaw))


def yaw_facing(vx: float, vy: float) -> float:
    """Yaw at which an asset's front (local +Y) points along (vx, vy)."""
    return math.atan2(-vx, vy)


def angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    """Unsigned angle between two vectors in radians (0 if either is ~zero)."""
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = (ax * bx + ay * by) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


# --------------------------------------------------------------------------- areas


def clean_area(a: float) -> float:
    return a if a > AREA_EPS else 0.0


def bboxes_overlap(a: Bounds, b: Bounds) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlap_area(pa: BaseGeometry, ba: Bounds, pb: BaseGeometry, bb: Bounds) -> float:
    """Intersection area with a bounding-box early-out."""
    if not bboxes_overlap(ba, bb):
        return 0.0
    return clean_area(pa.intersection(pb).area)


# --------------------------------------------------------------------------- room


@dataclass(frozen=True)
class OpeningGeom:
    id: str
    type: OpeningType
    wall_id: str
    t0: float  # span along the wall (from wall start)
    t1: float
    sill: float
    center: tuple[float, float]  # world point on the wall line
    inward: tuple[float, float]  # interior normal of its wall
    swing: Polygon | None  # interior door-swing quarter disc, if any


@dataclass
class WallGeom:
    id: str
    sx: float
    sy: float
    ex: float
    ey: float
    length: float
    ux: float  # unit direction start -> end
    uy: float
    nx: float  # interior normal (left of the direction)
    ny: float
    yaw_in: float  # yaw at which an asset's front equals the interior normal
    openings: list[OpeningGeom] = field(default_factory=list)

    def to_local(self, px: float, py: float) -> tuple[float, float]:
        """(t along the wall from its start, signed distance towards the interior)."""
        dx, dy = px - self.sx, py - self.sy
        return dx * self.ux + dy * self.uy, dx * self.nx + dy * self.ny

    def point(self, t: float, d: float) -> tuple[float, float]:
        return (self.sx + self.ux * t + self.nx * d, self.sy + self.uy * t + self.ny * d)

    def segment_distance(self, px: float, py: float) -> float:
        t, d = self.to_local(px, py)
        if t < 0:
            return math.hypot(t, d)
        if t > self.length:
            return math.hypot(t - self.length, d)
        return abs(d)

    def blocked_spans(self, height: float, wall_mounted: bool) -> list[tuple[float, float]]:
        """Spans along this wall where an asset cannot stand with its back to the wall:
        doors and passages always (there is no wall there); windows when the asset is
        taller than the sill, or wall-mounted (nothing is hung on glass)."""
        spans = []
        for o in self.openings:
            if o.type in (OpeningType.door, OpeningType.opening):
                spans.append((o.t0, o.t1))
            elif wall_mounted or height > o.sill:
                spans.append((o.t0, o.t1))
        return spans


def _swing_polygon(w: WallGeom, t0: float, t1: float, hinge: str, n_arc: int = 12) -> Polygon:
    width = t1 - t0
    if hinge == "start":
        th, dx, dy = t0, w.ux, w.uy
    else:
        th, dx, dy = t1, -w.ux, -w.uy
    hx, hy = w.point(th, 0.0)
    a0 = math.atan2(dy, dx)
    da = wrap_angle(math.atan2(w.ny, w.nx) - a0)
    pts = [(hx, hy)]
    for k in range(n_arc + 1):
        a = a0 + da * k / n_arc
        pts.append((hx + width * math.cos(a), hy + width * math.sin(a)))
    return Polygon(pts)


@dataclass(frozen=True)
class ObstacleGeom:
    id: str
    label: str
    poly: Polygon
    bounds: Bounds
    height: float


class RoomGeom:
    def __init__(self, room: Room):
        self.room = room
        self.floor = Polygon(room.floor.points)
        shapely.prepare(self.floor)
        self.bounds: Bounds = self.floor.bounds
        self.area = self.floor.area
        self.walls: list[WallGeom] = []
        for w in room.walls:
            (sx, sy), (ex, ey) = w.start, w.end
            length = math.hypot(ex - sx, ey - sy)
            ux, uy = (ex - sx) / length, (ey - sy) / length
            nx, ny = -uy, ux
            self.walls.append(
                WallGeom(w.id, sx, sy, ex, ey, length, ux, uy, nx, ny, yaw_facing(nx, ny))
            )
        self.wall_by_id = {w.id: w for w in self.walls}
        self.openings: dict[str, OpeningGeom] = {}
        for o in room.openings:
            w = self.wall_by_id[o.wall_id]
            t0, t1 = o.offset - o.width / 2, o.offset + o.width / 2
            swing = None
            if o.type == OpeningType.door and o.swing is not None and o.swing.into_room:
                swing = _swing_polygon(w, t0, t1, o.swing.hinge)
            og = OpeningGeom(
                o.id,
                o.type,
                o.wall_id,
                t0,
                t1,
                o.sill_height,
                w.point(o.offset, 0.0),
                (w.nx, w.ny),
                swing,
            )
            self.openings[o.id] = og
            w.openings.append(og)
        self.swings = [(o.id, o.swing, o.swing.bounds) for o in self.openings.values() if o.swing]
        self.obstacles = [
            ObstacleGeom(
                ob.id,
                ob.label,
                Polygon(ob.footprint.points),
                Polygon(ob.footprint.points).bounds,
                ob.height,
            )
            for ob in room.obstacles
        ]
        self.yaws = self._allowed_yaws()
        self._eroded: dict[float, BaseGeometry] = {}
        self._obstacle_buffers: dict[float, list[BaseGeometry]] = {}
        self._keep_clear: dict[tuple[str, float], Polygon] = {}

    def _allowed_yaws(self) -> list[float]:
        """Multiples of 90 deg, plus every wall's facing yaw (and its 90 deg rotations), so
        assets can sit square to non-axis-aligned walls."""
        offsets = [0.0]
        for w in self.walls:
            o = w.yaw_in % HALF_PI
            if all(abs(o - p) > 1e-6 and abs(abs(o - p) - HALF_PI) > 1e-6 for p in offsets):
                offsets.append(o)
        yaws = []
        for o in offsets:
            for k in range(4):
                y = wrap_angle(o + k * HALF_PI)
                if abs(y + math.pi) < 1e-9:
                    y = math.pi  # prefer +pi to -pi for readability
                yaws.append(y)
        return sorted(yaws)

    def snap_yaw(self, yaw: float) -> float:
        return min(self.yaws, key=lambda y: abs(wrap_angle(y - yaw)))

    # -- derived regions -------------------------------------------------------

    def keep_clear_rect(self, opening_id: str, distance: float) -> Polygon:
        key = (opening_id, distance)
        if key not in self._keep_clear:
            o = self.openings[opening_id]
            w = self.wall_by_id[o.wall_id]
            self._keep_clear[key] = Polygon(
                [
                    w.point(o.t0, 0),
                    w.point(o.t1, 0),
                    w.point(o.t1, distance),
                    w.point(o.t0, distance),
                ]
            )
        return self._keep_clear[key]

    def eroded_floor(self, r: float) -> BaseGeometry:
        if r not in self._eroded:
            self._eroded[r] = self.floor.buffer(-r, quad_segs=BUFFER_QUAD_SEGS)
        return self._eroded[r]

    def obstacle_buffers(self, r: float) -> list[BaseGeometry]:
        if r not in self._obstacle_buffers:
            self._obstacle_buffers[r] = [
                ob.poly.buffer(r, quad_segs=BUFFER_QUAD_SEGS) for ob in self.obstacles
            ]
        return self._obstacle_buffers[r]

    # -- queries ---------------------------------------------------------------

    def outside_area(self, poly: BaseGeometry, bounds: Bounds) -> float:
        """Area of ``poly`` outside the floor polygon (i.e. in or beyond the walls)."""
        fb = self.bounds
        if (
            bounds[0] >= fb[0]
            and bounds[1] >= fb[1]
            and bounds[2] <= fb[2]
            and bounds[3] <= fb[3]
            and self.floor.contains(poly)
        ):
            return 0.0
        return clean_area(poly.difference(self.floor).area)

    def distance_to_floor(self, x: float, y: float) -> float:
        if shapely.contains_xy(self.floor, x, y):
            return 0.0
        return self.floor.distance(Point(x, y))

    def random_point(self, rng: random.Random) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        for _ in range(200):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if shapely.contains_xy(self.floor, x, y):
                return x, y
        c = self.floor.representative_point()
        return c.x, c.y


# --------------------------------------------------------------------------- assets


@dataclass(eq=False)
class Posed:
    """An asset's world-frame geometry at one pose."""

    x: float
    y: float
    yaw: float
    fp: Polygon
    bounds: Bounds
    zones: list[Polygon]
    zone_bounds: list[Bounds]
    zone_purposes: list[str]
    fx: float  # world front unit vector
    fy: float
    back: np.ndarray  # (2, 2) world endpoints of the back edge
    front_edge: np.ndarray  # (2, 2) world endpoints of the front edge
    all_bounds: Bounds  # bounds of footprint + zones
    _buffers: dict[float, BaseGeometry] = field(default_factory=dict)

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def buffered(self, r: float) -> BaseGeometry:
        if r not in self._buffers:
            self._buffers[r] = self.fp.buffer(r, quad_segs=BUFFER_QUAD_SEGS)
        return self._buffers[r]


def _edge(pts: np.ndarray, use_min: bool) -> np.ndarray:
    ys = pts[:, 1]
    yv = ys.min() if use_min else ys.max()
    sel = pts[np.abs(ys - yv) <= 1e-6, 0]
    return np.array([[sel.min(), yv], [sel.max(), yv]])


class AssetTemplate:
    """Local-frame data of one asset plus a cache of its posed world geometry."""

    def __init__(self, idx: int, asset: Asset):
        self.idx = idx
        self.asset = asset
        self.id = asset.id
        self.label = asset.label or asset.id
        self.category = asset.category
        self.height = asset.dimensions.height
        self.mount = asset.mount
        self.wall_mounted = asset.mount == Mount.wall
        self.pts = np.asarray(asset.effective_footprint().points, dtype=float)
        self.zones = [
            (z.name, z.purpose, np.asarray(z.polygon.points, dtype=float))
            for z in asset.clearance_zones
        ]
        self.back = _edge(self.pts, use_min=True)
        self.front = _edge(self.pts, use_min=False)
        self.xmin, self.ymin = self.pts.min(axis=0)
        self.xmax, self.ymax = self.pts.max(axis=0)
        self.area = asset.effective_footprint().area
        self.zone_area = (
            shapely.union_all([Polygon(z) for _, _, z in self.zones]).area if (self.zones) else 0.0
        )
        all_pts = np.vstack([self.pts, *[z for _, _, z in self.zones]])
        self.reach = float(np.hypot(all_pts[:, 0], all_pts[:, 1]).max())
        self.signature = (
            asset.category,
            asset.mount.value,
            tuple(np.round(self.pts, 3).ravel()),
            tuple((p, tuple(np.round(z, 3).ravel())) for _, p, z in self.zones),
            round(self.height, 3),
        )
        self._cache: dict[tuple[float, float, float], Posed] = {}
        # all vertices (footprint, zones, back edge, front edge) in one array -> one matmul
        rings = [self.pts, *[z for _, _, z in self.zones]]
        self._n_rings = len(rings)
        self._ring_len = len(self.pts) if all(len(r) == len(self.pts) for r in rings) else 0
        self._splits = np.cumsum([len(r) for r in rings])[:-1]
        self._stack = np.vstack([*rings, self.back, self.front])
        self._n_ring_pts = sum(len(r) for r in rings)
        self._purposes = [p for _, p, _ in self.zones]

    def posed(self, x: float, y: float, yaw: float) -> Posed:
        key = (x, y, yaw)
        g = self._cache.get(key)
        if g is not None:
            return g
        c, s = math.cos(yaw), math.sin(yaw)
        w = self._stack @ np.array([[c, s], [-s, c]])  # row vectors: world = local @ rot + t
        w += (x, y)
        ring_pts = w[: self._n_ring_pts]
        if self._ring_len:
            arr = ring_pts.reshape(self._n_rings, self._ring_len, 2)
            polys = list(shapely.polygons(arr))
            mins, maxs = arr.min(axis=1), arr.max(axis=1)
        else:
            parts = np.split(ring_pts, self._splits)
            polys = [shapely.polygons(r) for r in parts]
            mins = np.array([r.min(axis=0) for r in parts])
            maxs = np.array([r.max(axis=0) for r in parts])
        bounds = [
            (a[0], a[1], b[0], b[1]) for a, b in zip(mins.tolist(), maxs.tolist(), strict=True)
        ]
        n = self._n_ring_pts
        g = Posed(
            x,
            y,
            yaw,
            polys[0],
            bounds[0],
            polys[1:],
            bounds[1:],
            self._purposes,
            -s,
            c,
            w[n : n + 2],
            w[n + 2 : n + 4],
            (*mins.min(axis=0).tolist(), *maxs.max(axis=0).tolist()),
        )
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = g
        return g
