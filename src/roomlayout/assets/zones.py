"""Clearance-zone geometry: sweep the footprint outward from one side.

A zone of depth ``d`` on side ``s`` is the region the footprint sweeps when translated by
``d`` in direction ``s``, minus the footprint itself (Minkowski sum with a segment). For
a rectangle that's the usual strip. For an L-sectional it also covers the notch in
front of the main seats that the chaise leaves free, where a coffee table would otherwise
fit inside the bounding box and block the legroom.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from roomlayout.schema import ClearanceZone, Polygon2D

from .categories import ZoneSpec

SIDE_DIRECTIONS = {
    "front": (0.0, 1.0),
    "back": (0.0, -1.0),
    "left": (-1.0, 0.0),  # the asset's left when it faces +Y is -X
    "right": (1.0, 0.0),
}


def sweep_zone(footprint: Polygon, direction: tuple[float, float], depth: float) -> list[Polygon]:
    """Region swept by ``footprint`` moved ``depth`` along ``direction``, minus itself.

    Usually one polygon. A concave footprint can give several: for an L-sectional the
    strip in front of the chaise and the notch in front of the main seats are disjoint
    when the notch is deeper than the zone. Largest first.
    """
    v = np.asarray(direction, dtype=float) * depth
    ring = np.asarray(footprint.exterior.coords)
    pieces = [footprint, affinity.translate(footprint, v[0], v[1])]
    for a, b in zip(ring[:-1], ring[1:], strict=True):
        quad = Polygon([a, b, b + v, a + v])
        if quad.area > 1e-12:
            pieces.append(quad)
    swept = shapely.union_all(pieces)
    zone = swept.difference(footprint)
    # Remove numerical slivers along edges parallel to the sweep.
    zone = zone.buffer(-1e-6, join_style="mitre").buffer(1e-6, join_style="mitre")
    parts = [g for g in getattr(zone, "geoms", [zone]) if isinstance(g, Polygon) and g.area > 1e-4]
    out = []
    for part in sorted(parts, key=lambda p: p.area, reverse=True):
        simple = Polygon(part.exterior).simplify(0.005, preserve_topology=True)
        if isinstance(simple, Polygon) and simple.area > 1e-4:
            out.append(orient(simple, sign=1.0))
    return out


def build_zones(footprint: Polygon2D, specs: tuple[ZoneSpec, ...]) -> list[ClearanceZone]:
    poly = Polygon(footprint.points)
    zones = []
    for spec in specs:
        parts = sweep_zone(poly, SIDE_DIRECTIONS[spec.side], spec.depth)
        for i, zp in enumerate(parts):
            pts = [(round(float(x), 5), round(float(y), 5)) for x, y in zp.exterior.coords[:-1]]
            name = spec.name if i == 0 else f"{spec.name}_{i + 1}"
            zones.append(
                ClearanceZone(name=name, polygon=Polygon2D(points=pts), purpose=spec.purpose)
            )
    return zones
