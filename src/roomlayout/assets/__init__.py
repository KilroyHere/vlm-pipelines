"""Asset normalization (WS-B, milestone M2a).

Turns a raw generator mesh (SAM 3D Objects, or any glb/gltf/obj/ply) into a solver-ready
``roomlayout.schema.Asset``: Z-up canonical frame with +Y = front (ADR-0001), fitted to
measured dimensions (ADR-0004), with a simplified footprint, default clearance zones and
a GLB written in glTF's Y-up convention. See docs/assets.md.

Imports ``roomlayout.schema`` only (never the solver or any generator code).
"""

from .categories import ROOMPLAN_CATEGORIES, CategoryProfile, lookup, profile_for
from .footprint import FootprintOptions, extract_footprint
from .meshio import MeshLoadError, export_glb, load_mesh
from .pipeline import NormalizeOptions, NormalizeResult, normalize, write_outputs
from .zones import with_default_zones

__all__ = [
    "ROOMPLAN_CATEGORIES",
    "CategoryProfile",
    "FootprintOptions",
    "MeshLoadError",
    "NormalizeOptions",
    "NormalizeResult",
    "export_glb",
    "extract_footprint",
    "load_mesh",
    "lookup",
    "normalize",
    "profile_for",
    "with_default_zones",
    "write_outputs",
]
