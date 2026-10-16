"""Levelling, horizontal alignment and re-centring in the Z-up frame.

Levelling: generator output can be a few degrees off vertical. That matters because a 2
degree tilt on a 1.8 m bookshelf adds 6 cm to its 35 cm depth before fitting. We find the
support plane, i.e. the faces of the 3D convex hull whose outward normal is within
``max_tilt_deg`` (8) of straight down, and rotate their area-weighted mean normal onto -Z.
Furniture rests on its legs/base, which are exactly those hull faces. Larger tilts are
left alone and reported (that is a wrong up axis, not a tilt; see ``fit.up_check``).

After the up-axis conversion the mesh stands upright but has an arbitrary yaw. We take the
minimum-area rectangle of the projected footprint (rotating calipers on the convex hull)
and rotate so its edges lie along X/Y, with the longer edge along X. That fixes yaw up to
a multiple of 90 degrees; the front heuristic (``front``) picks among the four.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
import trimesh

from .frames import homogeneous, yaw_matrix_rad


def support_normal(mesh: trimesh.Trimesh, max_tilt_deg: float = 8.0) -> np.ndarray | None:
    """Area-weighted mean outward normal of the convex-hull faces facing (nearly) down."""
    hull = mesh.convex_hull
    normals = hull.face_normals
    areas = hull.area_faces
    down = normals[:, 2] <= -math.cos(math.radians(max_tilt_deg))
    if not down.any() or areas[down].sum() <= 0:
        return None
    n = (normals[down] * areas[down, None]).sum(axis=0)
    return n / np.linalg.norm(n)


def level(mesh: trimesh.Trimesh, max_tilt_deg: float = 8.0) -> tuple[trimesh.Trimesh, float]:
    """Rotate so the support plane is horizontal. Returns (mesh, tilt removed in degrees).

    If no hull face points within ``max_tilt_deg`` of straight down the mesh is returned
    unchanged with tilt ``nan``.
    """
    n = support_normal(mesh, max_tilt_deg)
    if n is None:
        return mesh.copy(), float("nan")
    target = np.array([0.0, 0.0, -1.0])
    tilt = math.degrees(math.acos(float(np.clip(n @ target, -1.0, 1.0))))
    out = mesh.copy()
    if tilt > 1e-6:
        axis = np.cross(n, target)
        rot = trimesh.transformations.rotation_matrix(math.radians(tilt), axis)
        out.apply_transform(rot)
    return out, tilt


def footprint_principal_yaw(mesh: trimesh.Trimesh) -> float:
    """Yaw (radians) of the longer edge of the min-area rectangle of the XY projection."""
    pts = np.asarray(mesh.vertices)[:, :2]
    hull = shapely.MultiPoint(pts).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        raise ValueError("mesh projects to a line or a point; cannot find horizontal axes")
    rect = shapely.minimum_rotated_rectangle(hull)
    xy = np.asarray(rect.exterior.coords)[:4]
    e0 = xy[1] - xy[0]
    e1 = xy[2] - xy[1]
    edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    yaw = math.atan2(edge[1], edge[0])
    # Fold into [-pi/2, pi/2): an edge direction and its opposite are the same axis.
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def align_principal_axes(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float]:
    """Rotate about Z so the footprint's min-area rectangle is axis-aligned (long side on X).

    Returns the rotated copy and the yaw (radians) that was *removed*.
    """
    yaw = footprint_principal_yaw(mesh)
    out = mesh.copy()
    out.apply_transform(homogeneous(yaw_matrix_rad(-yaw)))
    return out, yaw


def recenter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate so the footprint bounding-box centre is at (0, 0) and the bottom at z = 0."""
    lo, hi = mesh.bounds
    shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    out = mesh.copy()
    out.apply_translation(shift)
    return out


def extents(mesh: trimesh.Trimesh) -> np.ndarray:
    lo, hi = mesh.bounds
    return hi - lo
