"""Floor footprint of a mesh in its local frame, as one simplified ``Polygon2D``.

Algorithm (rasterize and trace; an exact shapely union of triangles was ~10 s for an 80k
face mesh, the raster is <1 s):

1. Rasterize every triangle's XY projection (vertical faces included, drawn as lines, so
   zero-thickness panels survive) into an occupancy grid at ``resolution`` (default:
   longest side / 400, clamped to 2-10 mm). Each triangle is drawn with its outline, so
   adjacent triangles never leave cracks. The cost is that the region grows by under one
   pixel.
2. Trace: turn each row's runs of occupied pixels into boxes and union them (exact pixel
   boundary, holes included).
3. Close small gaps (morphological closing by ``closing`` = 1 cm) so parts that nearly
   touch merge.
4. Multiple parts: drop parts below ``min_part_fraction`` (1 %) of the area. If several
   parts remain (e.g. a two-piece sectional with a gap) take the **convex hull** of them.
   That is conservative: nothing can be placed between the pieces, which a single simple
   polygon couldn't express anyway. A chair's legs don't trigger this because the seat
   covers them in projection.
5. Fill holes (``Polygon2D`` is hole-free; a hole in a projection is enclosed by the asset).
6. Clip to the bounding box ``[-W/2, W/2] x [-D/2, D/2]`` so the footprint never exceeds
   it.
7. Simplify (Douglas-Peucker, topology-preserving) at ``tolerance`` (default 1.5 cm),
   growing the tolerance by 25 % until there are at most ``max_vertices`` (32). The
   vertices are a subset of the clipped boundary, so they stay inside the bounding box.
8. If the result covers >= ``rect_snap`` (98 %) of the bounding box, emit the exact
   rectangle instead (4 vertices, cheapest for the solver).

The report records the vertex count, area, the Hausdorff distance between the raw traced
region and the final polygon (the worst-case error introduced by steps 4-8), and which
multi-part rule fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from roomlayout.schema import Polygon2D


@dataclass
class FootprintOptions:
    resolution: float | None = None  # meters per pixel; None = auto
    closing: float = 0.01
    min_part_fraction: float = 0.01
    tolerance: float = 0.015
    max_vertices: int = 32
    rect_snap: float = 0.98


@dataclass
class FootprintResult:
    polygon: Polygon2D
    raw: Polygon | MultiPolygon = field(repr=False)
    resolution: float
    parts_found: int
    parts_dropped: int
    rule: str  # "single" | "convex_hull_of_disjoint_parts" | "rectangle"
    tolerance_used: float
    hausdorff: float
    bbox_fill: float

    @property
    def vertex_count(self) -> int:
        return len(self.polygon.points)


def rasterize_projection(mesh: trimesh.Trimesh, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy grid of the XY projection. Returns (mask[row, col], origin_xy).

    PIL truncates coordinates, so pixel (row r, col c) covers
    ``origin + [c, c+1) x [r, r+1) * resolution`` and the filled region contains the true
    projection, overshooting by under one pixel (conservative for collisions).
    """
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)][:, :, :2]
    lo = tri.reshape(-1, 2).min(axis=0)
    hi = tri.reshape(-1, 2).max(axis=0)
    pad = 2
    origin = lo - pad * resolution
    ncols, nrows = (np.ceil((hi - lo) / resolution).astype(int) + 2 * pad + 1).tolist()
    img = Image.new("1", (ncols, nrows), 0)
    draw = ImageDraw.Draw(img)
    pix = (tri - origin) / resolution
    for t in pix.tolist():
        draw.polygon([tuple(p) for p in t], fill=1, outline=1)
    return np.array(img, dtype=bool), origin


def trace_mask(mask: np.ndarray, origin: np.ndarray, resolution: float):
    """Union of per-row run boxes in world coordinates (exact pixel boundary)."""
    boxes = []
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    for r in range(mask.shape[0]):
        starts = np.flatnonzero(diff[r] == 1)
        ends = np.flatnonzero(diff[r] == -1)  # exclusive
        y0 = origin[1] + r * resolution
        y1 = y0 + resolution
        for c0, c1 in zip(starts, ends, strict=True):
            x0 = origin[0] + c0 * resolution
            x1 = origin[0] + c1 * resolution
            boxes.append(box(x0, y0, x1, y1))
    if not boxes:
        return Polygon()
    return shapely.union_all(boxes)


def _parts(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def _simplify(poly: Polygon, tol: float, max_vertices: int) -> tuple[Polygon, float]:
    current = poly
    while True:
        s = poly.simplify(tol, preserve_topology=True)
        if isinstance(s, Polygon) and not s.is_empty and s.is_valid:
            current = s
            if len(s.exterior.coords) - 1 <= max_vertices:
                return s, tol
        tol *= 1.25
        if tol > 10 * max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]):
            # Pathological: fall back to the convex hull, simplified.
            hull = poly.convex_hull.simplify(tol, preserve_topology=True)
            return (hull if isinstance(hull, Polygon) else current), tol


def to_polygon2d(poly: Polygon, half_w: float, half_d: float, ndigits: int = 5) -> Polygon2D:
    """Shapely polygon -> Polygon2D (CCW, open), rounded and clamped to the bbox."""
    poly = orient(poly, sign=1.0)
    pts = np.asarray(poly.exterior.coords)[:-1]
    pts = np.round(pts, ndigits)
    pts[:, 0] = np.clip(pts[:, 0], -half_w, half_w)
    pts[:, 1] = np.clip(pts[:, 1], -half_d, half_d)
    # Drop consecutive duplicates introduced by rounding/clamping.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 0, axis=1)
    if len(pts) > 1 and np.all(pts[0] == pts[-1]):
        keep[-1] = False
    pts = pts[keep]
    return Polygon2D(points=[(float(x), float(y)) for x, y in pts])


def extract_footprint(
    mesh: trimesh.Trimesh,
    width: float,
    depth: float,
    options: FootprintOptions | None = None,
) -> FootprintResult:
    """Footprint of ``mesh`` (already in its local frame, centred on the bbox footprint)."""
    opt = options or FootprintOptions()
    hw, hd = width / 2, depth / 2
    bbox = box(-hw, -hd, hw, hd)
    res = opt.resolution or float(np.clip(max(width, depth) / 400, 0.002, 0.01))

    mask, origin = rasterize_projection(mesh, res)
    raw = trace_mask(mask, origin, res)
    closed = raw.buffer(opt.closing, join_style="mitre").buffer(-opt.closing, join_style="mitre")
    if closed.is_empty:
        closed = raw

    parts = _parts(closed)
    total = sum(p.area for p in parts) or 1.0
    kept = [p for p in parts if p.area / total >= opt.min_part_fraction]
    dropped = len(parts) - len(kept)
    if not kept:
        raise ValueError("empty footprint")
    if len(kept) == 1:
        region = Polygon(kept[0].exterior)
        rule = "single"
    else:
        region = shapely.union_all(kept).convex_hull
        rule = "convex_hull_of_disjoint_parts"

    clipped = region.intersection(bbox)
    clipped_parts = _parts(clipped)
    if not clipped_parts:
        raise ValueError("footprint does not intersect the bounding box")
    clipped = Polygon(max(clipped_parts, key=lambda p: p.area).exterior)

    simple, tol_used = _simplify(clipped, opt.tolerance, opt.max_vertices)
    bbox_fill = simple.area / bbox.area
    if bbox_fill >= opt.rect_snap:
        simple = bbox
        rule = "rectangle" if rule == "single" else rule
    polygon = to_polygon2d(simple, hw, hd)

    final = Polygon(polygon.points)
    reference = raw.intersection(bbox)
    hausdorff = float(final.hausdorff_distance(reference)) if not reference.is_empty else 0.0
    return FootprintResult(
        polygon=polygon,
        raw=raw,
        resolution=res,
        parts_found=len(parts),
        parts_dropped=dropped,
        rule=rule,
        tolerance_used=tol_used,
        hausdorff=hausdorff,
        bbox_fill=float(final.area / bbox.area),
    )
