"""Synthetic furniture meshes in our canonical frame (Z-up, front = +Y, bottom-centre
origin) and a simulator that turns them into "generator output": random yaw, random
scale, random offset, SAM 3D's Y-up GLB convention and floating debris.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon, box

from roomlayout.schema import Dimensions


@dataclass
class Synthetic:
    name: str
    category: str
    mesh: trimesh.Trimesh  # canonical
    dims: Dimensions
    footprint: Polygon  # true projection in the canonical frame


def _box(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return b


def _cyl(r, z0, z1, sections=64) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    c.apply_translation((0, 0, (z0 + z1) / 2))
    return c


def _finish(name, category, parts, footprint) -> Synthetic:
    mesh = trimesh.util.concatenate(parts)
    lo, hi = mesh.bounds
    assert np.allclose([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], 0, atol=1e-12)
    ext = hi - lo
    dims = Dimensions(width=float(ext[0]), depth=float(ext[1]), height=float(ext[2]))
    return Synthetic(name, category, mesh, dims, footprint)


def l_sectional() -> Synthetic:
    """2.8 x 1.6 x 0.85 L-sectional: chaise on the right-front, backrest at the back."""
    parts = [
        _box(-1.4, 1.4, -0.8, 0.1, 0.0, 0.45),  # main seat block
        _box(0.6, 1.4, 0.1, 0.8, 0.0, 0.45),  # chaise
        _box(-1.4, 1.4, -0.8, -0.6, 0.45, 0.85),  # backrest
        _box(-1.4, -1.2, -0.8, 0.1, 0.45, 0.65),  # left arm
    ]
    fp = box(-1.4, -0.8, 1.4, 0.1).union(box(0.6, 0.1, 1.4, 0.8))
    return _finish("l_sectional", "sectional", parts, fp)


def chair() -> Synthetic:
    """0.48 wide x 0.52 deep x 0.9 dining-style chair: seat, backrest, four legs.

    Deeper than wide on purpose: the front is a *short* side.
    """
    w, d = 0.48, 0.52
    hw, hd = w / 2, d / 2
    parts = [
        _box(-hw, hw, -hd, hd, 0.42, 0.47),  # seat
        _box(-hw, hw, -hd, -hd + 0.05, 0.47, 0.90),  # backrest
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * (hw - 0.04), sy * (hd - 0.04)
            parts.append(_box(cx - 0.02, cx + 0.02, cy - 0.02, cy + 0.02, 0.0, 0.42))
    return _finish("chair", "chair", parts, box(-hw, -hd, hw, hd))


def round_table() -> Synthetic:
    """1.2 m round table: disc top, pedestal, base."""
    parts = [_cyl(0.6, 0.71, 0.75), _cyl(0.06, 0.05, 0.71, 24), _cyl(0.3, 0.0, 0.05, 48)]
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    fp = Polygon(np.c_[0.6 * np.cos(angles), 0.6 * np.sin(angles)])
    return _finish("round_table", "round_table", parts, fp)


def bookshelf() -> Synthetic:
    """0.9 x 0.35 x 1.8 bookshelf, open at the front (+Y), back panel at -Y."""
    hw, hd, h, t = 0.45, 0.175, 1.8, 0.02
    parts = [
        _box(-hw, hw, -hd, -hd + t, 0.0, h),  # back panel
        _box(-hw, -hw + t, -hd, hd, 0.0, h),  # left side
        _box(hw - t, hw, -hd, hd, 0.0, h),  # right side
        _box(-hw + t, hw - t, -hd + t, hd, 0.0, 0.06),  # plinth
        _box(-hw + t, hw - t, -hd + t, hd, h - t, h),  # top
    ]
    for z in (0.45, 0.85, 1.25):
        parts.append(_box(-hw + t, hw - t, -hd + t, hd, z, z + t))
    return _finish("bookshelf", "bookshelf", parts, box(-hw, -hd, hw, hd))


ALL = {f.__name__: f for f in (l_sectional, chair, round_table, bookshelf)}


def debris_blobs(mesh: trimesh.Trimesh, rng: np.random.Generator, n: int = 2) -> list:
    """Tiny floating cubes outside the object's bounding box (canonical frame)."""
    lo, hi = mesh.bounds
    blobs = []
    for _ in range(n):
        b = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
        side = rng.choice([-1, 1])
        b.apply_translation(
            (
                side * (hi[0] + 0.25),
                rng.uniform(lo[1], hi[1]),
                rng.uniform(lo[2] + 0.1, hi[2] + 0.2),
            )
        )
        blobs.append(b)
    return blobs


SAM3D_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def as_generator_output(
    synth: Synthetic, seed: int, debris: bool = True
) -> tuple[trimesh.Trimesh, dict]:
    """Random yaw/scale/offset, optional debris, then SAM 3D's Z-up -> Y-up rotation.

    SAM 3D's ``to_glb`` does ``vertices @ [[1,0,0],[0,0,-1],[0,1,0]]``, i.e. row vectors,
    (x, y, z) -> (x, z, -y). We reproduce exactly that.
    """
    rng = np.random.default_rng(seed)
    parts = [synth.mesh.copy()]
    if debris:
        parts += debris_blobs(synth.mesh, rng)
    m = trimesh.util.concatenate(parts)
    yaw = float(rng.uniform(0, 2 * np.pi))
    scale = float(rng.uniform(0.3, 1.5))
    offset = rng.uniform(-2, 2, size=3)
    rot = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    m.apply_transform(rot)
    m.apply_scale(scale)
    m.apply_translation(offset)
    m.vertices = m.vertices @ SAM3D_ZUP_TO_YUP
    return m, {"yaw": yaw, "scale": scale, "offset": offset, "debris": 2 if debris else 0}
