"""Box assets straight from a product listing (the v1 fast path, ADR-0009).

A listing gives the manufacturer's width, depth and height. That is enough to answer
"will it fit?" before any 3D model exists: the e2e run showed box assets from listing
sizes give the same feasibility and request results as SAM 3D meshes
(spikes/e2e_arkitscenes/RESULTS.md). The mesh, when it arrives, only changes the picture.

Product files: ``examples/products/*.json`` (``{"products": [ProductSpec, ...]}``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from roomlayout.schema import Asset, AssetSource, Dimensions, Polygon2D

from .zones import with_default_zones


class LShape(BaseModel):
    """An L-shaped piece (sectional with chaise): a full-width seat strip along the back
    plus a chaise running to the front on one side."""

    seat_depth_m: PositiveFloat  # depth of the main strip (the listing's "Depth")
    chaise_width_m: PositiveFloat
    chaise_side: Literal["left", "right"] = "right"  # as seen from the front


class ProductSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    url: str | None = None
    category: str
    subtype: str | None = None
    label: str = ""
    width_m: PositiveFloat
    depth_m: PositiveFloat  # overall depth (chaise included for an L)
    height_m: PositiveFloat
    l_shape: LShape | None = None
    count: int = Field(1, ge=1)


def l_footprint(width: float, depth: float, shape: LShape) -> Polygon2D:
    """Counter-clockwise L in the asset's local frame (origin = bbox centre, +Y = front).

    The back strip spans the full width at y in [-d/2, -d/2 + seat_depth]; the chaise
    spans the full depth on one side. "Right as seen from the front" is local -X (a viewer
    in front looks along -Y)."""
    w, d = width / 2, depth / 2
    sd = min(shape.seat_depth_m, depth)
    cw = min(shape.chaise_width_m, width)
    y_seat = -d + sd
    pts = [(-w, -d), (w, -d), (w, y_seat), (-w + cw, y_seat), (-w + cw, d), (-w, d)]
    if shape.chaise_side == "left":  # mirror in X, then restore CCW order
        pts = [(-x, y) for x, y in reversed(pts)]
    return Polygon2D(points=[(round(x, 4), round(y, 4)) for x, y in pts])


def box_assets(spec: ProductSpec) -> list[Asset]:
    """``spec.count`` box assets (ids ``<id>``, or ``<id>_1``... when count > 1), with the
    category's default clearance zones and ``dimensions_from="listing"``."""
    footprint = l_footprint(spec.width_m, spec.depth_m, spec.l_shape) if spec.l_shape else None
    base = Asset(
        id=spec.id,
        category=spec.category,
        subtype=spec.subtype,
        label=spec.label or spec.name,
        dimensions=Dimensions(width=spec.width_m, depth=spec.depth_m, height=spec.height_m),
        footprint=footprint,
        source=AssetSource(dimensions_from="listing", geometry_from="box"),
    )
    base = with_default_zones(base)
    if spec.count == 1:
        return [base]
    return [
        base.model_copy(update={"id": f"{spec.id}_{i}", "label": f"{base.label} {i}"})
        for i in range(1, spec.count + 1)
    ]


def load_products(path: str | Path, ids: list[str] | None = None) -> list[ProductSpec]:
    """Product specs from a products JSON file, optionally only ``ids`` (in that order)."""
    specs = [ProductSpec.model_validate(p) for p in json.loads(Path(path).read_text())["products"]]
    if ids is None:
        return specs
    by_id = {s.id: s for s in specs}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown product ids {missing}; have {sorted(by_id)}")
    return [by_id[i] for i in ids]
