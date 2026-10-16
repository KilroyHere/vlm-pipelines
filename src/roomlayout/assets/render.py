"""Orthographic PNG renders of the four front candidates (for a VLM or the user to pick).

Pure matplotlib (Agg canvas, no pyplot state): triangles are projected orthographically,
depth-sorted (painter's algorithm) and Lambert-shaded. Vertex/face colours are used when
the mesh has them (SAM 3D outputs vertex colours), otherwise a neutral grey.

Each candidate view looks straight *at* the candidate front side from 25 degrees above, so
it shows that side and the top and nothing of the neighbouring sides. A three-quarter view
reads better as 3D, but a 30-degree offset let the view of side D show the open shelves of
side C on a bookshelf, which made the question ambiguous.
The contact sheet (``<stem>.front_candidates.png``) is one image to send to a VLM with the
question "which letter is the front?". It holds a top view and two three-quarter
overviews from opposite corners, all with the letters A-D on the floor next to their
sides, and below them the four straight-on side views. The sheet doesn't mark the
heuristic's pick, so a VLM's answer stays independent of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from .front import LABELS, FrontDecision, candidate_direction

MAX_RENDER_FACES = 60_000
GREY = np.array([0.72, 0.72, 0.70])


def _face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        kind = mesh.visual.kind
        if kind == "texture":
            vc = mesh.visual.to_color().vertex_colors
            return vc[mesh.faces][:, :, :3].mean(axis=1) / 255.0
        if kind == "vertex":
            return mesh.visual.vertex_colors[mesh.faces][:, :, :3].mean(axis=1) / 255.0
        if kind == "face":
            return mesh.visual.face_colors[:, :3] / 255.0
    except Exception:  # visuals are best-effort; geometry is what matters
        pass
    return np.tile(GREY, (len(mesh.faces), 1))


def _prepare_for_render(mesh: trimesh.Trimesh, colors: np.ndarray):
    """Even out triangle sizes for the painter's algorithm.

    Large triangles (CAD-like or synthetic meshes) are subdivided so depth sorting by
    centroid doesn't leave slivers. Huge meshes are decimated by vertex clustering.
    """
    if len(mesh.faces) <= MAX_RENDER_FACES:
        max_edge = float(np.ptp(mesh.vertices, axis=0).max()) / 25
        verts, faces, idx = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge, max_iter=8, return_index=True
        )
        if len(faces) <= MAX_RENDER_FACES:
            return np.asarray(verts), np.asarray(faces), colors[idx]
        return np.asarray(mesh.vertices), np.asarray(mesh.faces), colors
    cell = float(np.ptp(mesh.vertices, axis=0).max()) / 150
    while True:
        keys = np.round(mesh.vertices / cell).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        faces = inv[mesh.faces]
        f0, f1, f2 = faces.T
        ok = (f0 != f1) & (f1 != f2) & (f0 != f2)
        if ok.sum() <= MAX_RENDER_FACES:
            break
        cell *= 1.3
    counts = np.bincount(inv)
    verts = np.zeros((counts.size, 3))
    np.add.at(verts, inv, mesh.vertices)
    verts /= counts[:, None]
    return verts, faces[ok], colors[ok]


def _basis(view_dir: np.ndarray):
    v = view_dir / np.linalg.norm(view_dir)
    right = np.cross(-v, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:  # looking straight down: screen = world XY
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -v)
    return v, right, up


def _draw(ax, verts, faces, colors, view_dir: np.ndarray, normals: np.ndarray, extra=None):
    """Draw the mesh; ``extra`` are 3D points that must also fit in the frame."""
    v, right, up = _basis(view_dir)
    tri = verts[faces]
    screen = np.stack([tri @ right, tri @ up], axis=-1)
    depth = (tri @ v).mean(axis=1)
    light = v + 0.6 * up + 0.3 * right
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ light)
    rgb = np.clip(colors * shade[:, None], 0, 1)
    order = np.argsort(depth)
    pc = PolyCollection(
        screen[order], facecolors=rgb[order], edgecolors=rgb[order], linewidths=0.25
    )
    ax.add_collection(pc)
    pts = screen.reshape(-1, 2)
    if extra is not None:
        pts = np.vstack([pts, np.stack([extra @ right, extra @ up], axis=-1)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.08 * float((hi - lo).max())
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return right, up


def _side_view_direction(k: int, elevation_deg: float = 25.0) -> np.ndarray:
    d = candidate_direction(k)
    el = np.radians(elevation_deg)
    return np.array([np.cos(el) * d[0], np.cos(el) * d[1], np.sin(el)])


def _corner_view_direction(corner_xy, elevation_deg: float = 32.0) -> np.ndarray:
    c = np.asarray(corner_xy, dtype=float)
    c /= np.linalg.norm(c)
    el = np.radians(elevation_deg)
    return np.array([np.cos(el) * c[0], np.cos(el) * c[1], np.sin(el)])


def _normals(verts, faces) -> np.ndarray:
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm > 0, norm, 1.0)


def _label_points(verts) -> list[np.ndarray]:
    """Floor-level points just outside the middle of each candidate side (A-D order)."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = (lo + hi) / 2
    half = (hi - lo) / 2
    span = float((hi - lo)[:2].max())
    pts = []
    for k in range(4):
        d = candidate_direction(k)
        xy = centre[:2] + d * (np.abs(d) @ half[:2] + 0.22 * span)
        pts.append(np.array([xy[0], xy[1], lo[2]]))
    return pts


def _labelled_view(ax, verts, faces, colors, normals, view_dir, title) -> None:
    pts = _label_points(verts)
    right, up = _draw(ax, verts, faces, colors, view_dir, normals, extra=np.array(pts))
    for k, (label, p) in enumerate(zip(LABELS, pts, strict=True)):
        # Only label sides facing the camera; a far side's label would land on the object.
        if view_dir[2] < 0.99 and candidate_direction(k) @ view_dir[:2] <= 1e-9:
            continue
        ax.text(
            p @ right,
            p @ up,
            label,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="crimson",
            zorder=5,
            bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": "crimson", "lw": 1.2},
        )
    ax.set_title(title, fontsize=11)


def render_front_candidates(
    mesh: trimesh.Trimesh,
    decision: FrontDecision | None,
    out_dir: str | Path,
    stem: str,
    category: str = "",
    size_px: int = 512,
) -> dict[str, Path]:
    """Write ``<stem>.front_<L>.png`` per candidate and ``<stem>.front_candidates.png``.

    ``mesh`` is the aligned (pre-front) mesh that the decision's hypotheses refer to.
    ``decision`` is accepted for API symmetry; the sheet deliberately doesn't show it.
    """
    del decision
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = _face_colors(mesh)
    verts, faces, colors = _prepare_for_render(mesh, colors)
    normals = _normals(verts, faces)
    dpi = 100
    paths: dict[str, Path] = {}

    for k, label in enumerate(LABELS):
        fig = Figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.02, 0.02, 0.96, 0.88))
        _draw(ax, verts, faces, colors, _side_view_direction(k), normals)
        fig.suptitle(f"{label}: looking straight at side {label}", fontsize=12, y=0.97)
        p = out_dir / f"{stem}.front_{label}.png"
        fig.savefig(p, dpi=dpi, facecolor="white")
        paths[label] = p

    fig = Figure(figsize=(16, 9), dpi=dpi)
    FigureCanvasAgg(fig)
    what = f"this {category.replace('_', ' ')}" if category else "this object"
    fig.suptitle(f"Which letter marks the FRONT of {what} (the side you use it from)?", fontsize=16)
    w, h = 0.235, 0.40
    xs = [0.015, 0.26, 0.505, 0.75]
    ax = fig.add_axes((xs[0], 0.50, w, h))
    _labelled_view(ax, verts, faces, colors, normals, np.array([0.0, 0.0, 1.0]), "top view")
    ax = fig.add_axes((xs[1], 0.50, w * 1.5, h))
    _labelled_view(
        ax,
        verts,
        faces,
        colors,
        normals,
        _corner_view_direction((1, 1)),
        "overview from A/B corner",
    )
    ax = fig.add_axes((xs[2] + w * 0.5 + 0.01, 0.50, w * 1.5, h))
    _labelled_view(
        ax,
        verts,
        faces,
        colors,
        normals,
        _corner_view_direction((-1, -1)),
        "overview from C/D corner",
    )
    for k, label in enumerate(LABELS):
        ax = fig.add_axes((xs[k], 0.03, w, h))
        _draw(ax, verts, faces, colors, _side_view_direction(k), normals)
        ax.set_title(f"side {label} (straight on)", fontsize=11)
    p = out_dir / f"{stem}.front_candidates.png"
    fig.savefig(p, dpi=dpi, facecolor="white")
    paths["sheet"] = p
    return paths
