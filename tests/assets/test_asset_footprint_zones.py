import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon, box

from roomlayout.assets import FootprintOptions, extract_footprint
from roomlayout.assets.categories import PROFILES, ROOMPLAN_CATEGORIES, lookup, profile_for
from roomlayout.assets.zones import build_zones
from roomlayout.schema import Polygon2D

from .synthetic import _box, l_sectional, round_table


def _centered(mesh):
    lo, hi = mesh.bounds
    mesh = mesh.copy()
    mesh.apply_translation(-np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]))
    return mesh, mesh.extents


def test_disjoint_parts_use_convex_hull():
    # Two separate pieces 0.5 m apart (e.g. a two-piece sectional): one simple polygon
    # can't express the gap, so the conservative hull is used.
    mesh = trimesh.util.concatenate([_box(0, 1, 0, 1, 0, 0.8), _box(1.5, 2.5, 0, 1, 0, 0.8)])
    mesh, ext = _centered(mesh)
    fp = extract_footprint(mesh, ext[0], ext[1])
    assert fp.rule == "convex_hull_of_disjoint_parts"
    assert fp.parts_found == 2
    assert fp.polygon.area == pytest.approx(2.5, rel=0.01)


def test_legs_only_table_gets_hull_and_tiny_specks_are_dropped():
    legs = [_box(x, x + 0.05, y, y + 0.05, 0, 0.7) for x in (0, 0.95) for y in (0, 0.55)]
    speck = _box(0.5, 0.501, 0.3, 0.301, 0, 0.001)
    mesh, ext = _centered(trimesh.util.concatenate([*legs, speck]))
    fp = extract_footprint(mesh, ext[0], ext[1])
    assert fp.parts_found == 5
    assert fp.parts_dropped == 1
    assert fp.polygon.area == pytest.approx(1.0 * 0.6, rel=0.02)


def test_vertex_cap_and_bbox_respected():
    mesh, ext = _centered(round_table().mesh)
    for cap in (32, 12, 6):
        fp = extract_footprint(mesh, ext[0], ext[1], FootprintOptions(max_vertices=cap))
        assert len(fp.polygon.points) <= cap
        poly = Polygon(fp.polygon.points)
        assert poly.difference(box(-0.6, -0.6, 0.6, 0.6)).area < 1e-12
    fp32 = extract_footprint(mesh, ext[0], ext[1])
    assert fp32.polygon.area == pytest.approx(math.pi * 0.36, rel=0.03)
    assert fp32.hausdorff <= 0.02


def test_zero_thickness_panel_survives():
    # A single vertical quad (no thickness) next to a box: drawn as a line, not lost.
    panel = trimesh.Trimesh(
        vertices=[[0, -0.5, 0], [0, 0.5, 0], [0, 0.5, 1], [0, -0.5, 1]],
        faces=[[0, 1, 2], [0, 2, 3]],
    )
    seat = _box(0, 0.6, -0.5, 0.5, 0, 0.4)
    mesh, ext = _centered(trimesh.util.concatenate([panel, seat]))
    fp = extract_footprint(mesh, ext[0], ext[1])
    assert fp.rule == "rectangle"


def test_every_profile_maps_to_a_roomplan_category():
    for p in PROFILES.values():
        assert p.roomplan in ROOMPLAN_CATEGORIES, p.key
        for z in p.zones:
            assert 0.2 <= z.depth <= 1.2


def test_aliases_and_case():
    assert lookup("Couch").key == "sofa"
    assert lookup("L-shaped sofa").key == "sectional"
    assert lookup("bookcase").roomplan == "storage"
    assert lookup("washerDryer").key == "washerDryer"
    assert lookup("TV").roomplan == "television"
    assert lookup("desk").roomplan == "table"
    assert lookup("piano") is None
    assert profile_for("piano").zones == ()


def test_television_has_no_clearance_zone():
    assert lookup("television").zones == ()


def test_legroom_zone_is_in_front_and_adjacent():
    fp = Polygon2D.rectangle(2.2, 0.95)
    (z,) = build_zones(fp, lookup("sofa").zones)
    assert z.name == "legroom" and z.purpose == "use"
    zp = Polygon(z.polygon.points)
    assert zp.bounds == pytest.approx((-1.1, 0.475, 1.1, 0.875))
    assert zp.intersection(Polygon(fp.points)).area < 1e-9


def test_l_sofa_legroom_covers_the_notch_too():
    synth = l_sectional()
    fp = Polygon2D(points=list(synth.footprint.exterior.coords)[:-1])
    zones = build_zones(fp, lookup("sectional").zones)
    names = [z.name for z in zones]
    assert names == ["legroom", "legroom_2"]
    union = Polygon(zones[0].polygon.points).union(Polygon(zones[1].polygon.points))
    assert union.contains(box(-1.3, 0.15, 0.5, 0.45))  # notch in front of the main seats
    assert union.contains(box(0.65, 0.85, 1.35, 1.15))  # in front of the chaise


def test_bed_has_access_on_both_sides_and_foot():
    fp = Polygon2D.rectangle(1.6, 2.1)
    zones = {z.name: Polygon(z.polygon.points) for z in build_zones(fp, lookup("bed").zones)}
    assert set(zones) == {"access_left", "access_right", "access_foot"}
    assert zones["access_left"].bounds == pytest.approx((-1.4, -1.05, -0.8, 1.05))
    assert zones["access_right"].bounds == pytest.approx((0.8, -1.05, 1.4, 1.05))
    assert zones["access_foot"].bounds == pytest.approx((-0.8, 1.05, 0.8, 1.65))


def test_dining_table_has_zones_on_all_four_sides():
    fp = Polygon2D.rectangle(1.6, 0.9)
    zones = build_zones(fp, lookup("dining_table").zones)
    assert len(zones) == 4
