"""roomlayout.fit: listing-size boxes, "will it fit?", and advice when it doesn't."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomlayout.assets import box_assets, load_products
from roomlayout.fit import _resized, _without, check_fit, free_floor_area
from roomlayout.pipeline import empty_room
from roomlayout.schema import LayoutProblem

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "examples" / "products" / "ikea_2026-11.json"


def room(name: str) -> LayoutProblem:
    p = LayoutProblem.model_validate_json(
        (ROOT / "examples" / "problems" / f"{name}.json").read_text()
    )
    return empty_room(p)[0]


def test_fits_returns_layouts_and_no_advice():
    r = check_fit(room("bedroom_basic"), load_products(PRODUCTS, ["malm_bed_queen"]), seconds=3)
    assert r.fits and r.layouts and r.advice == [] and r.drops == []
    assert 0 < r.furniture_m2 < r.free_floor_m2


def test_does_not_fit_gives_drop_and_width_advice():
    specs = load_products(PRODUCTS, ["soderhamn_sectional", "micke_desk"])
    r = check_fit(room("infeasible_tiny_bedroom"), specs, seconds=3)
    assert not r.fits and r.reasons
    drops = {d.product_id: d.fits for d in r.drops}
    assert drops == {"soderhamn_sectional": True, "micke_desk": False}
    assert r.width is not None and r.width.product_id == "soderhamn_sectional"
    assert r.width.max_width_m is not None and r.width.max_width_m < 2.911
    assert r.advice[0].startswith("Without the gray L-shaped sectional")


def test_variants_drop_constraints_and_keep_l_shape():
    specs = load_products(PRODUCTS, ["soderhamn_sectional", "lisabo_chair"])
    r = check_fit(room("living_room_basic"), specs, seconds=2, advise=False)
    full = r.built.problem
    chair_ids = {f"lisabo_chair_{i}" for i in range(1, 5)}
    less = _without(full, chair_ids)
    assert not chair_ids & {a.id for a in less.assets}
    assert all(
        not ({getattr(c, "a", None), getattr(c, "b", None)} & chair_ids) for c in less.constraints
    )
    narrow = _resized(full, specs[0], 2.2)
    sec = next(a for a in narrow.assets if a.id == "soderhamn_sectional")
    assert sec.dimensions.width == pytest.approx(2.2)
    assert len(sec.effective_footprint().points) == 6  # still an L
    assert sec.clearance_zones  # rebuilt for the new size


def test_product_id_clash_is_an_error():
    from roomlayout.pipeline import furnish

    spec = load_products(PRODUCTS, ["micke_desk"])
    clashing = furnish(room("bedroom_basic"), box_assets(spec[0]))  # desk already there
    with pytest.raises(ValueError, match="already used"):
        check_fit(clashing, spec, seconds=1)


def test_free_floor_subtracts_fixtures():
    r0 = room("bedroom_basic")
    assert free_floor_area(r0) <= r0.room.floor.area
