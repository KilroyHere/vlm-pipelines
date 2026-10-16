"""Mesh loading and GLB export.

Loading flattens any scene graph into one ``trimesh.Trimesh`` with node transforms
applied, then rotates it into our Z-up frame. Export writes glTF's Y-up convention
(see ``frames``), so ``load_mesh(export_glb(m), up="y")`` gives back ``m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .frames import OURS_TO_GLTF, UpAxis, homogeneous, source_to_ours


class MeshLoadError(ValueError):
    """The file could not be turned into a triangle mesh."""


def as_single_mesh(obj: trimesh.parent.Geometry) -> trimesh.Trimesh:
    """Flatten a Scene (applying node transforms) or pass a Trimesh through."""
    if isinstance(obj, trimesh.Trimesh):
        mesh = obj
    elif isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not meshes:
            raise MeshLoadError("scene contains no triangle meshes")
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        raise MeshLoadError(f"expected a triangle mesh, got {type(obj).__name__}")
    if len(mesh.faces) == 0:
        raise MeshLoadError("mesh has no faces")
    return mesh


def load_mesh(path: str | Path, up: UpAxis = "y") -> trimesh.Trimesh:
    """Load any trimesh-readable file (glb/gltf/obj/ply/stl/...) into our Z-up frame.

    ``up`` names the source's up axis: ``"y"`` for glTF/GLB and SAM 3D Objects output
    (the default), ``"z"`` for files that are already Z-up.
    """
    path = Path(path)
    if not path.exists():
        raise MeshLoadError(f"no such file: {path}")
    try:
        loaded = trimesh.load(str(path), process=False)
    except Exception as exc:  # trimesh raises many exception types for bad files
        raise MeshLoadError(f"could not read {path}: {exc}") from exc
    mesh = as_single_mesh(loaded)
    return to_ours(mesh, up)


def to_ours(mesh: trimesh.Trimesh, up: UpAxis) -> trimesh.Trimesh:
    """Copy of ``mesh`` rotated from the source up-axis convention into ours (Z-up)."""
    out = mesh.copy()
    rot = source_to_ours(up)
    if not np.allclose(rot, np.eye(3)):
        out.apply_transform(homogeneous(rot))
    return out


def export_glb(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write ``mesh`` (in our Z-up local frame) as a GLB in glTF's Y-up, front = +Z."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = mesh.copy()
    out.apply_transform(homogeneous(OURS_TO_GLTF))
    path.write_bytes(out.export(file_type="glb"))
    return path
