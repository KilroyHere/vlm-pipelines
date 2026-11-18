"""Box assets from product listings (roomlayout.assets.listing)."""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import Polygon

from roomlayout.assets import LShape, ProductSpec, box_assets, l_footprint, load_products
from roomlayout.schema import LayoutProblem

PRODUCTS = Path(__file__).resolve().parents[2] / "examples" / "products" / "ikea_2026-11.json"


def _poly(fp) -> Polygon:
    return Polygon(fp.points)


@pytest.mark.parametrize("side", ["left", "right"])
def test_l_footprint_area_bbox_and_orientation(side):
    shape = LShape(seat_depth_m=1.0, chaise_width_m=0.9, chaise_side=side)
    poly = _poly(l_footprint(3.0, 1.5, shape))
    assert poly.is_valid and poly.exterior.is_ccw
    assert poly.area == pytest.approx(3.0 * 1.0 + 0.9 * 0.5)
    assert poly.bounds == pytest.approx((-1.5, -0.75, 1.5, 0.75))
    # The chaise reaches the front (+Y) on its side only. Right as seen from the front
    # is local -X: a viewer in front of the sofa looks along -Y.
    chaise_x = -1.0 if side == "right" else 1.0
    assert poly.contains(Polygon([(chaise_x, 0.5), (chaise_x + 0.1, 0.5), (chaise_x, 0.6)]))
    assert not poly.contains(Polygon([(-chaise_x, 0.5), (-chaise_x + 0.1, 0.5), (-chaise_x, 0.6)]))


def test_box_assets_count_ids_and_provenance():
    spec = ProductSpec(
        id="chair", name="x", category="chair", subtype="dining_chair",
        width_m=0.46, depth_m=0.51, height_m=0.8, count=3,
    )  # fmt: skip
    assets = box_assets(spec)
    assert [a.id for a in assets] == ["chair_1", "chair_2", "chair_3"]
    for a in assets:
        assert a.source.dimensions_from == "listing" and a.source.geometry_from == "box"
        assert a.clearance_zones  # category defaults applied
        assert a.footprint is None  # plain box


def test_shipped_products_load_and_drop_into_a_problem():
    specs = load_products(PRODUCTS)
    assert {s.id for s in specs} >= {"soderhamn_sectional", "lisabo_chair", "billy_bookcase"}
    assets = [a for s in specs for a in box_assets(s)]
    assert len(assets) == 9  # 4 chairs
    sectional = next(a for a in assets if a.id == "soderhamn_sectional")
    assert sectional.footprint is not None
    assert sectional.effective_footprint().area == pytest.approx(3.37, abs=0.01)
    # Valid inside a LayoutProblem (ids unique, footprints inside dims).
    problem = LayoutProblem.model_validate_json(
        (PRODUCTS.parents[1] / "problems" / "living_room_basic.json").read_text()
    )
    data = problem.model_dump(mode="json")
    data["assets"] += [a.model_dump(mode="json") for a in assets]
    LayoutProblem.model_validate(data)


def test_load_products_by_id_and_unknown():
    assert [s.id for s in load_products(PRODUCTS, ["micke_desk", "lisabo_table"])] == [
        "micke_desk",
        "lisabo_table",
    ]
    with pytest.raises(KeyError):
        load_products(PRODUCTS, ["nope"])
