"""Floating-debris removal.

Generators (SAM 3D included) often emit small disconnected blobs around the object. They
matter a lot here because a single blob far from the object inflates the bounding box,
which then corrupts the fit to measured dimensions.

Rule (documented in docs/assets.md): split the mesh into connected components (vertices
welded by position, so UV seams don't split parts) and drop a component when **both**

- its surface area is below ``area_fraction`` of the total, and
- its bounding-box diagonal is below ``size_fraction`` of the whole mesh's diagonal.

The largest component is always kept. Requiring both conditions keeps thin but large parts
(chair legs, table pedestals) while removing specks. Small separate parts that are real
(a drawer knob modelled as its own island) can be removed too; that costs a little visual
detail but never changes the footprint by more than the knob itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class DebrisResult:
    mesh: trimesh.Trimesh
    components: int
    removed_components: int
    removed_faces: int
    removed_area_fraction: float


def face_component_labels(mesh: trimesh.Trimesh, weld_tol: float | None = None) -> np.ndarray:
    """Connected-component label per face, with vertices welded by position."""
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if weld_tol is None:
        diag = float(np.linalg.norm(np.ptp(verts, axis=0))) or 1.0
        weld_tol = 1e-6 * diag
    keys = np.round(verts / weld_tol).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    welded = welded.reshape(-1)
    f = welded[faces]
    rows = np.concatenate([f[:, 0], f[:, 1]])
    cols = np.concatenate([f[:, 1], f[:, 2]])
    n = int(welded.max()) + 1
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, vlabels = connected_components(graph, directed=False)
    return vlabels[f[:, 0]]


def remove_debris(
    mesh: trimesh.Trimesh,
    area_fraction: float = 0.005,
    size_fraction: float = 0.10,
) -> DebrisResult:
    labels = face_component_labels(mesh)
    ncomp = int(labels.max()) + 1
    if ncomp == 1:
        return DebrisResult(mesh.copy(), 1, 0, 0, 0.0)

    areas = np.bincount(labels, weights=mesh.area_faces, minlength=ncomp)
    total_area = float(areas.sum()) or 1.0
    diag_all = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))) or 1.0

    tri = mesh.vertices[mesh.faces]  # (F, 3, 3)
    lo = np.full((ncomp, 3), np.inf)
    hi = np.full((ncomp, 3), -np.inf)
    np.minimum.at(lo, labels, tri.min(axis=1))
    np.maximum.at(hi, labels, tri.max(axis=1))
    diags = np.linalg.norm(hi - lo, axis=1)

    debris = (areas / total_area < area_fraction) & (diags / diag_all < size_fraction)
    debris[int(np.argmax(areas))] = False
    if not debris.any():
        return DebrisResult(mesh.copy(), ncomp, 0, 0, 0.0)

    keep = ~debris[labels]
    out = mesh.copy()
    out.update_faces(keep)
    out.remove_unreferenced_vertices()
    return DebrisResult(
        mesh=out,
        components=ncomp,
        removed_components=int(debris.sum()),
        removed_faces=int((~keep).sum()),
        removed_area_fraction=float(areas[debris].sum() / total_area),
    )
