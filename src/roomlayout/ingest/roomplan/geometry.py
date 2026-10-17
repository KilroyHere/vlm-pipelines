"""Frame conversion and small polygon helpers (numpy only; no shapely in ingest).

ARKit/RoomPlan world: right-handed, +Y up, meters. Ours (ADR-0001): right-handed, +Z up.
The conversion ``(x, y, z)_ark -> (x, -z, y)_ours`` is a proper rotation (+90 deg about X),
so handedness and polygon winding are preserved.
"""

from __future__ import annotations

import math

import numpy as np

ARK_TO_OURS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def to_ours(v) -> np.ndarray:
    """ARKit vector/point (3,) or (N, 3) -> our frame."""
    return np.asarray(v, dtype=float) @ ARK_TO_OURS.T


def to_ark(v) -> np.ndarray:
    """Our frame -> ARKit (inverse of :func:`to_ours`)."""
    return np.asarray(v, dtype=float) @ ARK_TO_OURS


def yaw_from_front(front_xy) -> float:
    """Yaw (CCW about +Z) whose local +Y maps to ``front_xy``: front = (-sin, cos)."""
    fx, fy = float(front_xy[0]), float(front_xy[1])
    return wrap_angle(math.atan2(-fx, fy))


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_polygon(p, poly: np.ndarray) -> bool:
    """Even-odd ray casting; points on the boundary may go either way."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def point_segment_distance(p, a, b) -> tuple[float, float]:
    """(distance, t) of point p to segment a-b, t in [0, 1] along the segment."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab))), t


def clean_polygon(poly: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Drop repeated and collinear vertices (within ``tol`` meters) and orient CCW."""
    pts = [np.asarray(p, dtype=float) for p in poly]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        out = []
        n = len(pts)
        for i in range(n):
            prev, cur, nxt = pts[i - 1], pts[i], pts[(i + 1) % n]
            if np.linalg.norm(cur - prev) < tol:
                changed = True
                continue
            d, _ = point_segment_distance(cur, prev, nxt)
            if d < tol and np.linalg.norm(nxt - prev) > tol:
                changed = True
                continue
            out.append(cur)
        if len(out) >= 3:
            pts = out
        else:
            break
    arr = np.array(pts)
    if signed_area(arr) < 0:
        arr = arr[::-1]
    return arr


def start_at_southwest(poly: np.ndarray) -> np.ndarray:
    """Rotate vertex order to start at the min (y, x) vertex (deterministic ids)."""
    i = min(range(len(poly)), key=lambda k: (round(poly[k][1], 4), round(poly[k][0], 4)))
    return np.roll(poly, -i, axis=0)


def oriented_rect(cx: float, cy: float, yaw: float, w: float, d: float) -> np.ndarray:
    """Corners (CCW) of a w x d rectangle centred at (cx, cy) rotated by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: intersection of polygon ``subject`` with convex CCW ``clip``."""
    out = [np.asarray(p, dtype=float) for p in subject]
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        if not out:
            break

        def side(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        inp, out = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            sc, sp = side(cur), side(prev)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def overlap_area(p: np.ndarray, q: np.ndarray) -> float:
    """Intersection area of two convex polygons."""
    inter = clip_convex(p, q if signed_area(q) > 0 else q[::-1])
    return abs(signed_area(inter)) if len(inter) >= 3 else 0.0


def rectilinear_union(rects: list[tuple[float, float, float, float]]) -> np.ndarray | None:
    """Outline (CCW) of a union of axis-aligned rectangles (x0, x1, y0, y1).

    Returns None when the union isn't a single simple polygon without holes.
    """
    xs = sorted({round(v, 6) for r in rects for v in r[:2]})
    ys = sorted({round(v, 6) for r in rects for v in r[2:]})
    nx, ny = len(xs) - 1, len(ys) - 1
    cov = np.zeros((nx, ny), dtype=bool)
    for x0, x1, y0, y1 in rects:
        for i in range(nx):
            mx = (xs[i] + xs[i + 1]) / 2
            if not x0 < mx < x1:
                continue
            for j in range(ny):
                my = (ys[j] + ys[j + 1]) / 2
                if y0 < my < y1:
                    cov[i, j] = True

    def covered(i, j):
        return 0 <= i < nx and 0 <= j < ny and cov[i, j]

    edges: dict[tuple, tuple] = {}
    for i in range(nx):
        for j in range(ny):
            if not cov[i, j]:
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if not covered(i, j - 1):
                edges[(x0, y0)] = edges.get((x0, y0), ()) + ((x1, y0),)
            if not covered(i + 1, j):
                edges[(x1, y0)] = edges.get((x1, y0), ()) + ((x1, y1),)
            if not covered(i, j + 1):
                edges[(x1, y1)] = edges.get((x1, y1), ()) + ((x0, y1),)
            if not covered(i - 1, j):
                edges[(x0, y1)] = edges.get((x0, y1), ()) + ((x0, y0),)
    if not edges or any(len(v) != 1 for v in edges.values()):
        return None
    start = min(edges)
    loop = [start]
    cur = edges[start][0]
    while cur != start:
        loop.append(cur)
        if cur not in edges or len(loop) > len(edges):
            return None
        cur = edges[cur][0]
    if len(loop) != len(edges):  # more than one loop: disjoint parts or holes
        return None
    return clean_polygon(np.array(loop), tol=1e-6)
