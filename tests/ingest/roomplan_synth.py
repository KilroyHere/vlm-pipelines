"""Build RoomPlan ``CapturedRoom`` JSON from geometry given in *our* frame.

Encodes exactly what the importer reads (docs/ingest.md): ARKit Y-up world, 4x4
transforms as 16 floats in column-major order, Swift-enum categories as ``{"case": {}}``,
surfaces with local x along the surface, objects with local +Z as the front.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

_ids = itertools.count(1)


def ark(v_ours) -> np.ndarray:
    """Our (x, y, z) -> ARKit (x, z, -y): inverse of the importer's conversion."""
    x, y, z = v_ours
    return np.array([x, z, -y], dtype=float)


def transform(center_ours, x_axis_ours, z_axis_ours) -> list[float]:
    """Column-major 4x4 with local axes given in our frame (local y = up)."""
    x = ark(x_axis_ours)
    z = ark(z_axis_ours)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, ark(center_ours)
    return [float(v) for v in m.T.reshape(-1)]  # column-major


def ident(prefix: str = "ID") -> str:
    return f"{prefix}-{next(_ids):04d}-0000-0000-0000-000000000000"


def surface(a, b, height, category="wall", payload=None, *, z0=0.0, parent=None, sid=None):
    """Vertical surface along our-frame segment a->b, bottom at height z0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    ux = np.array([d[0], d[1], 0.0]) / length
    normal = np.array([ux[1], -ux[0], 0.0])  # local z; with local y = up, z = x × y
    mid = (a + b) / 2
    out = {
        "identifier": sid or ident(category.upper()),
        "category": {category: payload or {}},
        "confidence": {"high": {}},
        "dimensions": [length, height, 0.0],
        "transform": transform((mid[0], mid[1], z0 + height / 2), ux, normal),
        "story": 0,
        "completedEdges": [],
        "curve": None,
    }
    if parent:
        out["parentIdentifier"] = parent
    return out


def floor(points_ours, z=0.0):
    """Floor surface with polygonCorners (identity transform: corners in world coords)."""
    corners = [list(ark((x, y, z))) for x, y in points_ours]
    return {
        "identifier": ident("FLOOR"),
        "category": {"floor": {}},
        "confidence": {"high": {}},
        "dimensions": [0.0, 0.0, 0.0],
        "transform": [float(v) for v in np.eye(4).T.reshape(-1)],
        "polygonCorners": corners,
        "story": 0,
        "completedEdges": [],
        "curve": None,
    }


def obj(category, center_xy, front_deg, w, d, h, *, bottom=0.0, attributes=None):
    """Object box: ``front_deg`` is the direction of its front in our frame (0 = +x)."""
    f = (math.cos(math.radians(front_deg)), math.sin(math.radians(front_deg)), 0.0)
    x = (-f[1], f[0], 0.0)  # right-handed with local y = up: x = up × front
    return {
        "identifier": ident(category.upper()),
        "category": {category: {}},
        "confidence": {"high": {}},
        "dimensions": [w, h, d],
        "transform": transform((center_xy[0], center_xy[1], bottom + h / 2), x, f),
        "attributes": attributes or {},
        "story": 0,
    }


def rect_room(w=5.0, d=4.0, h=2.6, *, with_floor=True):
    """CapturedRoom dict for an axis-aligned w x d room with corner (0, 0)."""
    pts = [(0, 0), (w, 0), (w, d), (0, d)]
    walls = [surface(pts[i], pts[(i + 1) % 4], h) for i in range(4)]
    return {
        "version": 2,
        "story": 0,
        "walls": walls,
        "doors": [],
        "windows": [],
        "openings": [],
        "floors": [floor(pts)] if with_floor else [],
        "objects": [],
        "sections": [],
    }
