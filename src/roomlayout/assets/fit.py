"""Fit a canonical mesh to measured dimensions (ADR-0004: measured dimensions win).

The mesh is scaled per axis so its bounding box equals the measured W x D x H exactly.
Anisotropic scaling is always applied; the *report* says how much the generator's
proportions disagreed:

- ``scale``: per-axis factors (sx, sy, sz) = (W/w, D/d, H/h).
- ``aspect_mismatch``: max_i |s_i / s_geo - 1|, with s_geo the geometric mean of the
  factors. It is 0 when one uniform scale would fit all three dims, and 0.15 means some
  axis had to be stretched or squashed 15 % relative to the others.
- ``flagged`` when ``aspect_mismatch > tolerance`` (default 0.15). This is a warning, not
  a failure: the asset is still produced at the measured size, but the shape may look
  distorted (or the front/up guess may be wrong). Callers should show it to the user.

``up_check`` asks whether another axis of the mesh as the vertical would match the
measured proportions much better. That catches generator output that isn't upright, or a
wrong ``--up`` flag.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import trimesh

from roomlayout.schema import Dimensions


@dataclass
class FitReport:
    mesh_extents: tuple[float, float, float]  # before fitting, in the oriented frame
    target: tuple[float, float, float]
    scale: tuple[float, float, float]
    aspect_mismatch: float
    per_axis_mismatch: tuple[float, float, float]
    tolerance: float
    flagged: bool
    uniform_scale: float
    uniform_residual_m: tuple[float, float, float]  # |extent * s_geo - target| per axis


@dataclass
class UpCheck:
    current_mismatch: float
    best_alternative_axis: str | None  # "x" or "y" of the oriented mesh, if better
    best_alternative_mismatch: float | None
    suspicious: bool


def aspect_mismatch(extents: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, float]:
    s = target / extents
    s_geo = float(np.exp(np.mean(np.log(s))))
    per_axis = np.abs(s / s_geo - 1.0)
    return float(per_axis.max()), per_axis, s_geo


def fit_to_dimensions(
    mesh: trimesh.Trimesh, dims: Dimensions, tolerance: float = 0.15
) -> tuple[trimesh.Trimesh, FitReport]:
    lo, hi = mesh.bounds
    ext = hi - lo
    if np.any(ext <= 0):
        raise ValueError(f"degenerate mesh extents {ext.tolist()}")
    target = np.array([dims.width, dims.depth, dims.height], dtype=float)
    scale = target / ext
    mismatch, per_axis, s_geo = aspect_mismatch(ext, target)

    out = mesh.copy()
    # Scale about the bottom-centre of the bbox so the local-frame origin stays put.
    anchor = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    m = np.eye(4)
    m[:3, :3] = np.diag(scale)
    m[:3, 3] = anchor - scale * anchor
    out.apply_transform(m)
    # Snap: remove floating-point drift so bounds equal the target to ~1e-12.
    lo2, hi2 = out.bounds
    out.apply_translation(-np.array([(lo2[0] + hi2[0]) / 2, (lo2[1] + hi2[1]) / 2, lo2[2]]))

    report = FitReport(
        mesh_extents=tuple(float(v) for v in ext),
        target=tuple(float(v) for v in target),
        scale=tuple(float(v) for v in scale),
        aspect_mismatch=mismatch,
        per_axis_mismatch=tuple(float(v) for v in per_axis),
        tolerance=tolerance,
        flagged=mismatch > tolerance,
        uniform_scale=s_geo,
        uniform_residual_m=tuple(float(v) for v in np.abs(ext * s_geo - target)),
    )
    return out, report


def up_check(extents: np.ndarray, dims: Dimensions, tolerance: float = 0.15) -> UpCheck:
    """Would treating the mesh's x or y axis as vertical match the measurements better?"""
    target = np.array([dims.width, dims.depth, dims.height], dtype=float)
    current, _, _ = aspect_mismatch(np.asarray(extents, dtype=float), target)
    best_axis, best = None, None
    for axis, name in ((0, "x"), (1, "y")):
        horiz = [i for i in range(3) if i != axis]
        for a, b in itertools.permutations(horiz):
            alt = np.array([extents[a], extents[b], extents[axis]], dtype=float)
            mm, _, _ = aspect_mismatch(alt, target)
            if best is None or mm < best:
                best_axis, best = name, mm
    suspicious = bool(current > tolerance and best is not None and best < 0.5 * current)
    return UpCheck(
        current_mismatch=current,
        best_alternative_axis=best_axis if suspicious else None,
        best_alternative_mismatch=best if suspicious else None,
        suspicious=suspicious,
    )
