"""A furniture asset in its canonical local frame (ADR-0001).

Local frame: origin at the centre of the asset's bounding-box footprint on the floor
(z = 0 is the bottom of the asset), +Z up, +Y is the asset's *front*
(the side you use it from: the seat side of a sofa, the screen side of a TV).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, PositiveFloat

from .geometry import Polygon2D


class Dimensions(BaseModel):
    width: PositiveFloat  # along local X
    depth: PositiveFloat  # along local Y
    height: PositiveFloat  # along Z


class Mount(StrEnum):
    floor = "floor"
    wall = "wall"  # wall-mounted (TV, shelf): back must touch a wall


class ClearanceZone(BaseModel):
    """Local-frame region that must stay free of *other assets' footprints* for
    this asset to be usable. Clearance zones may overlap each other
    (two chairs can share the same walking space)."""

    name: str
    polygon: Polygon2D
    purpose: Literal["access", "swing", "use"] = "access"


class MeshRef(BaseModel):
    """Visual mesh, already normalized into the asset's canonical local frame."""

    uri: str
    format: Literal["glb", "obj", "ply"] = "glb"


class AssetSource(BaseModel):
    dimensions_from: Literal["roomplan", "listing", "manual", "mesh"]
    geometry_from: Literal["box", "sam3d", "catalog", "object_capture", "manual"] = "box"


class Asset(BaseModel):
    id: str
    # RoomPlan category where possible: sofa, chair, table, bed, storage, television, ...
    category: str
    label: str = ""
    dimensions: Dimensions
    footprint: Polygon2D | None = None  # local frame; None means the bounding rectangle
    clearance_zones: list[ClearanceZone] = []
    mount: Mount = Mount.floor
    movable: bool = True
    mesh: MeshRef | None = None
    source: AssetSource = AssetSource(dimensions_from="manual")
    front_confirmed: bool = False

    def effective_footprint(self) -> Polygon2D:
        if self.footprint is not None:
            return self.footprint
        return Polygon2D.rectangle(self.dimensions.width, self.dimensions.depth)
