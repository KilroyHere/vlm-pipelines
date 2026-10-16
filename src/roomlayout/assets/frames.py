"""Frame conversions between source mesh conventions and our canonical frame (ADR-0001).

Our frame: right-handed, meters, +Z up. An asset's front is local +Y, so its own right
(front x up) is +X.

glTF 2.0 (spec, "Coordinate System and Units"): right-handed, +Y up, the front of an asset
faces +Z, and the asset's right is -X. Mapping our (right, up, front) = (+X, +Z, +Y) onto
glTF's (-X, +Y, +Z) gives the proper rotation

    (x, y, z)_ours  ->  (-x, z, y)_gltf

which is its own inverse. Exporting with it means a GLB we write faces +Z exactly as glTF
viewers expect, and loading a glTF/GLB with ``up="y"`` applies the same matrix.

Generators do not agree on yaw (SAM 3D's ``to_glb`` maps its internal Z-up frame with
``(x, y, z) -> (x, z, -y)``, which differs from the matrix above by a 180 degree turn about
the up axis). That is harmless: normalization re-derives yaw from the geometry and the
front heuristic, so only the *up axis* of the source matters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

UpAxis = Literal["y", "z"]

# 3x3 rotation, ours -> glTF. Symmetric and involutory (M @ M == I), det = +1.
OURS_TO_GLTF = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
GLTF_TO_OURS = OURS_TO_GLTF.T


def homogeneous(rot3: np.ndarray, translation: np.ndarray | None = None) -> np.ndarray:
    """4x4 transform from a 3x3 rotation (and optional translation)."""
    m = np.eye(4)
    m[:3, :3] = rot3
    if translation is not None:
        m[:3, 3] = translation
    return m


def source_to_ours(up: UpAxis) -> np.ndarray:
    """3x3 rotation taking source-frame coordinates to our Z-up frame."""
    if up == "y":
        return GLTF_TO_OURS.copy()
    if up == "z":
        return np.eye(3)
    raise ValueError(f"unsupported up axis {up!r} (expected 'y' or 'z')")


def yaw_matrix(quarter_turns: int) -> np.ndarray:
    """Exact 3x3 rotation about +Z by ``quarter_turns * 90`` degrees (CCW from above)."""
    k = quarter_turns % 4
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def yaw_matrix_rad(theta: float) -> np.ndarray:
    """3x3 rotation about +Z by ``theta`` radians (CCW from above)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
