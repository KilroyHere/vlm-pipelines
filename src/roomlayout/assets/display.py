"""Display cleanup: make a fitted mesh look tidy in the viewer without changing its size.

Runs after the footprint is extracted (``normalize``), so nothing here can change what the
solver sees. Generator meshes (SAM 3D) come out with small holes, faceted per-triangle
shading and ragged bottoms that make furniture float or sink a few millimetres. Steps:

1. merge duplicate vertices, drop degenerate faces;
2. fill small holes (``trimesh.repair.fill_holes``: closes boundary loops of a few edges);
3. light Taubin smoothing (``lamb``/``nu`` pair that doesn't shrink the mesh), off for
   ``smooth_iterations=0``;
4. flatten the base: vertices within ``base_band`` of the lowest point go to z = 0, so
   the piece stands flat on the floor;
5. restore the exact fitted bounding box (smoothing moves extremes by a little);
6. recompute smooth vertex normals.

Vertex colours stay attached: every step moves vertices, none reorders them, except the
merge, which keeps the first vertex's colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class DisplayOptions:
    smooth_iterations: int = 5
    # Smooth only dense meshes (generator output). Low-poly CAD-like meshes are already
    # clean, and smoothing a 12-triangle box would round it into a blob.
    min_faces_to_smooth: int = 2000
    smooth_lamb: float = 0.5
    smooth_nu: float = 0.53
    base_band: float = 0.01  # fraction of height treated as "the base"
    fill_holes: bool = True


@dataclass
class DisplayReport:
    holes_filled: bool
    base_vertices: int
    max_shift_m: float  # largest vertex movement from smoothing + base flattening


def display_cleanup(
    mesh: trimesh.Trimesh, opt: DisplayOptions | None = None
) -> tuple[trimesh.Trimesh, DisplayReport]:
    """A cleaned copy of ``mesh`` (local frame, base at z = 0) with the same bounding box."""
    opt = opt or DisplayOptions()
    m = mesh.copy()
    lo0, hi0 = m.bounds.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    before = np.asarray(m.vertices).copy()

    filled = False
    if opt.fill_holes and not m.is_watertight:
        filled = bool(trimesh.repair.fill_holes(m))

    if opt.smooth_iterations > 0 and len(m.faces) >= opt.min_faces_to_smooth:
        trimesh.smoothing.filter_taubin(
            m, lamb=opt.smooth_lamb, nu=opt.smooth_nu, iterations=opt.smooth_iterations
        )

    v = np.asarray(m.vertices).copy()
    # Restore the fitted box: per-axis affine map of the current bounds onto the original.
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    v = lo0 + (v - lo) / span * (hi0 - lo0)
    # Flatten the base onto z = lo0[2] (0 for a normalized asset).
    height = float(hi0[2] - lo0[2])
    band = lo0[2] + opt.base_band * height
    base = v[:, 2] <= band
    v[base, 2] = lo0[2]
    m.vertices = v

    n = min(len(before), len(v))
    shift = float(np.linalg.norm(v[:n] - before[:n], axis=1).max()) if n else 0.0
    m.fix_normals()
    _ = m.vertex_normals  # computed (area-weighted, smooth) and cached for export
    return m, DisplayReport(holes_filled=filled, base_vertices=int(base.sum()), max_shift_m=shift)
