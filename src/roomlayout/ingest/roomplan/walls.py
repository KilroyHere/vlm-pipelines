"""Floor polygon and walls in our frame.

Floor polygon, in order of preference:
1. RoomPlan's floor surface ``polygonCorners`` (iOS 17+). In real exports wall endpoints
   lie exactly on it, and its vertices come out clockwise in our frame (we re-orient).
2. A loop built from the wall segments: endpoints within ``snap`` are merged, and a
   single open chain is closed with a straight edge (noted below ``max_gap``, warned above).
3. Walls that don't form one chain: the convex hull of their endpoints, with a warning.

Walls are oriented so the floor is on their left (our CCW convention), ordered along the
floor boundary starting from the south-west-most corner, and named ``wall_1``... Walls
with floor on *both* sides (a partition standing inside the room) become obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .format import Element
from .geometry import (
    clean_polygon,
    point_in_polygon,
    point_segment_distance,
    signed_area,
    start_at_southwest,
    to_ours,
)

SNAP = 0.15  # m: endpoints closer than this are the same corner
MAX_GAP = 1.0  # m: closing a longer gap in the wall loop is a warning, not a note
SIDE_PROBE = 0.05  # m: how far to the side of a wall we test for floor


@dataclass
class WallSeg:
    source: Element
    a: np.ndarray  # (2,) ours
    b: np.ndarray
    height: float
    interior: bool = False  # floor on both sides -> obstacle, not a boundary wall

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


def surface_segment(e: Element) -> tuple[np.ndarray, np.ndarray]:
    """Floor-plan segment of a vertical surface: centre +- local x * width / 2."""
    c, ux, half = e.center, e.rotation[:, 0], e.dimensions[0] / 2
    return to_ours(c - ux * half)[:2], to_ours(c + ux * half)[:2]


def floor_from_surface(floors: list[Element], notes: list[str], warnings: list[str]):
    """Floor polygon (ours, xy) from the floor surfaces' polygonCorners, or None."""
    usable = [f for f in floors if f.polygon_corners is not None and len(f.polygon_corners) >= 3]
    if not usable:
        return None
    polys = []
    for f in usable:
        local = np.c_[f.polygon_corners, np.ones(len(f.polygon_corners))]
        world = (f.transform @ local.T).T[:, :3]
        ours = to_ours(world)
        if np.ptp(ours[:, 2]) > 0.05:
            warnings.append(f"floor {f.identifier[:8]} isn't level ({np.ptp(ours[:, 2]):.2f} m)")
        polys.append(clean_polygon(ours[:, :2]))
    if len(polys) > 1:
        areas = [abs(signed_area(p)) for p in polys]
        warnings.append(
            f"{len(polys)} floor surfaces; using the largest ({max(areas):.1f} m2). v1 ingests "
            "one room"
        )
        return polys[int(np.argmax(areas))]
    notes.append("floor polygon from RoomPlan floor polygonCorners")
    return polys[0]


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in pts})

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def floor_from_walls(
    segs: list[WallSeg], notes: list[str], warnings: list[str], snap: float = SNAP
):
    """Floor polygon from the wall loop (see module docstring). None if impossible."""
    if len(segs) < 2:
        return None
    nodes: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []

    def node_of(p):
        for i, n in enumerate(nodes):
            if np.linalg.norm(n - p) <= snap:
                members[i].append(p)
                nodes[i] = np.mean(members[i], axis=0)
                return i
        nodes.append(np.array(p, dtype=float))
        members.append([p])
        return len(nodes) - 1

    edges = []
    for s in segs:
        u, v = node_of(s.a), node_of(s.b)
        if u != v:
            edges.append((u, v))
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    degrees = {n: len(nb) for n, nb in adj.items()}

    # Connected?
    seen, stack = set(), [next(iter(adj))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(adj[n])
    simple = len(seen) == len(adj) and all(d in (1, 2) for d in degrees.values())
    ends = [n for n, d in degrees.items() if d == 1]
    if simple and len(ends) in (0, 2):
        start = ends[0] if ends else next(iter(adj))
        order, prev, cur = [start], None, start
        while True:
            nxt = [m for m in adj[cur] if m != prev]
            if not nxt or (nxt[0] == start and len(order) > 2):
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            order.append(cur)
        poly = np.array([nodes[i] for i in order])
        if ends:
            gap = float(np.linalg.norm(nodes[ends[0]] - nodes[ends[1]]))
            msg = f"wall loop is open: closed a {gap:.2f} m gap with a straight edge"
            if gap > MAX_GAP:
                warnings.append(msg + "; the floor shape there is a guess (missing wall?)")
            else:
                notes.append(msg + " (missing wall segment or wall-less passage)")
        if len(poly) >= 3 and abs(signed_area(poly)) > 1e-3:
            notes.append("floor polygon built from the wall loop (no RoomPlan floor polygon)")
            return clean_polygon(poly)
    hull = _convex_hull(np.array([p for s in segs for p in (s.a, s.b)]))
    if len(hull) < 3:
        return None
    warnings.append(
        "walls don't form a single closed loop (branches or several pieces); using the "
        "convex hull of the wall endpoints as the floor. Check the scan"
    )
    return clean_polygon(hull)


def orient_and_order(
    segs: list[WallSeg], floor: np.ndarray, warnings: list[str]
) -> list[WallSeg]:
    """Orient each wall with the floor on its left; flag partitions; sort along boundary."""
    boundary = []
    for s in segs:
        d = s.b - s.a
        if s.length < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / s.length
        mid = (s.a + s.b) / 2
        left = point_in_polygon(mid + SIDE_PROBE * n, floor)
        right = point_in_polygon(mid - SIDE_PROBE * n, floor)
        if left and right:
            s.interior = True
        elif right and not left:
            s.a, s.b = s.b, s.a
        elif not left and not right:
            warnings.append(
                f"wall {s.source.identifier[:8]} lies outside the floor polygon; oriented "
                "towards the floor centroid"
            )
            c = floor.mean(axis=0)
            if np.dot(c - mid, n) < 0:
                s.a, s.b = s.b, s.a
        boundary.append(s)

    # Sort boundary walls by where their midpoint falls along the floor perimeter.
    edges = [(floor[i], floor[(i + 1) % len(floor)]) for i in range(len(floor))]
    cum = np.concatenate([[0.0], np.cumsum([np.linalg.norm(b - a) for a, b in edges])])

    def perimeter_param(s: WallSeg) -> float:
        mid = (s.a + s.b) / 2
        best = min(
            (point_segment_distance(mid, a, b) + (k,) for k, (a, b) in enumerate(edges)),
            key=lambda r: r[0],
        )
        _, t, k = best
        return float(cum[k] + t * (cum[k + 1] - cum[k]))

    return sorted(boundary, key=lambda s: (s.interior, perimeter_param(s)))


def uncovered_boundary(floor: np.ndarray, walls: list[WallSeg], tol: float = 0.05) -> float:
    """Length of floor boundary with no wall along it (open sides of the room)."""
    total = 0.0
    for i in range(len(floor)):
        a, b = floor[i], floor[(i + 1) % len(floor)]
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        u = (b - a) / length
        spans = []
        for w in walls:
            if w.interior:
                continue
            da, _ = point_segment_distance(w.a, a, b)
            db, _ = point_segment_distance(w.b, a, b)
            if da < tol and db < tol:
                ta, tb = sorted((float((w.a - a) @ u), float((w.b - a) @ u)))
                spans.append((max(0.0, ta), min(length, tb)))
        spans.sort()
        covered, end = 0.0, 0.0
        for s, e in spans:
            if e > end:
                covered += e - max(s, end)
                end = e
        total += max(0.0, length - covered)
    return total


def floor_polygon_order(floor: np.ndarray) -> np.ndarray:
    return start_at_southwest(clean_polygon(floor))
