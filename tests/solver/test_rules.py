"""One test group per constraint type and implicit rule, on hand-placed poses."""

import math

import pytest
from solver_helpers import (
    FACE_E,
    FACE_N,
    FACE_S,
    box,
    box_room,
    front_zone,
    make_problem,
    rect,
    results,
)

from roomlayout.schema import Pose2D
from roomlayout.solver import evaluate
from roomlayout.solver.model import IMPLICIT_RULES


def door(offset=1.0, width=0.8, wall="w_s", swing=True, into=True, hinge="start"):
    d = {
        "id": "door",
        "type": "door",
        "wall_id": wall,
        "offset": offset,
        "width": width,
        "height": 2.0,
    }
    if swing:
        d["swing"] = {"into_room": into, "hinge": hinge}
    return d


def window(offset=2.0, width=1.0, wall="w_s", sill=0.9):
    return {
        "id": "win",
        "type": "window",
        "wall_id": wall,
        "offset": offset,
        "width": width,
        "height": 1.0,
        "sill_height": sill,
    }


# ----------------------------------------------------------------------------- implicit


def test_every_constraint_and_implicit_rule_is_reported():
    p = make_problem(
        [box("a", 1, 1)],
        [{"type": "against_wall", "id": "c1", "a": "a"}, {"type": "walkway_width", "id": "c2"}],
    )
    res, _ = results(p, {"a": (1, 1, 0)})
    assert set(res) == {f"implicit:{r}" for r in IMPLICIT_RULES} | {"c1", "c2"}
    assert all(r.hard for k, r in res.items() if k.startswith("implicit:"))


def test_inside_room():
    p = make_problem([box("a", 1, 1)])
    res, ok = results(p, {"a": (1, 1, 0)})
    assert ok and res["implicit:inside_room"].satisfied
    res, ok = results(p, {"a": (0.25, 1.5, 0)})  # a quarter of the box is through the wall
    assert not ok
    assert res["implicit:inside_room"].violation == pytest.approx(0.25)


def test_no_overlap():
    p = make_problem([box("a", 1, 1), box("b", 1, 1)])
    res, ok = results(p, {"a": (1, 1, 0), "b": (1.5, 1, 0)})
    assert not ok and res["implicit:no_overlap"].violation == pytest.approx(0.5)
    res, ok = results(p, {"a": (1, 1, 0), "b": (2, 1, 0)})  # touching is fine
    assert ok and res["implicit:no_overlap"].satisfied


def test_clear_of_obstacles():
    room = box_room(
        obstacles=[
            {"id": "col", "label": "column", "height": 2.5, "footprint": rect(2, 1, 2.4, 1.4)}
        ]
    )
    p = make_problem([box("a", 1, 1)], room=room)
    res, ok = results(p, {"a": (2, 1, 0)})
    assert not ok and res["implicit:clear_of_obstacles"].violation == pytest.approx(0.16)
    res, ok = results(p, {"a": (1, 1, 0)})
    assert ok


def test_clearance_zones_other_footprint_wall_and_shared_zones():
    a = box("a", 1, 1, zones=[front_zone(1, 1, 0.5)])
    p = make_problem([a, box("b", 1, 1, zones=[front_zone(1, 1, 0.5)])])
    # b's footprint 0.25 m into a's zone
    res, ok = results(p, {"a": (1, 1, FACE_N), "b": (1, 2.25, FACE_N)})
    assert not ok
    assert res["implicit:clearance_zones"].violation == pytest.approx(
        0.25 + 0.25
    )  # + b's zone in wall
    # facing each other: zones coincide, which is allowed
    res, ok = results(p, {"a": (1, 1, FACE_N), "b": (1, 2.5, FACE_S)})
    assert ok and res["implicit:clearance_zones"].satisfied
    # a's zone sticking through the north wall
    p1 = make_problem([a])
    res, ok = results(p1, {"a": (1, 2.25, FACE_N)})
    assert not ok and res["implicit:clearance_zones"].violation == pytest.approx(0.25)


def test_clearance_zone_into_obstacle():
    room = box_room(
        obstacles=[
            {"id": "col", "label": "column", "height": 2.5, "footprint": rect(0.8, 1.6, 1.2, 2.0)}
        ]
    )
    p = make_problem([box("a", 1, 1, zones=[front_zone(1, 1, 0.6)])], room=room)
    res, ok = results(p, {"a": (1, 1, FACE_N)})
    assert not ok and res["implicit:clearance_zones"].violation == pytest.approx(0.16)


def test_door_swing_is_a_quarter_disc_on_the_interior_side():
    p = make_problem([box("a", 0.2, 0.2)], room=box_room(openings=[door()]))
    # hinge at (0.6, 0), radius 0.8
    res, ok = results(p, {"a": (0.8, 0.3, 0)})
    assert not ok and res["implicit:door_swing"].violation == pytest.approx(0.04)
    # inside the swing's bounding square but outside the disc
    res, ok = results(p, {"a": (1.3, 0.7, 0)})
    assert ok and res["implicit:door_swing"].satisfied
    # door opening outwards: nothing to keep clear inside
    p2 = make_problem([box("a", 0.2, 0.2)], room=box_room(openings=[door(into=False)]))
    res, ok = results(p2, {"a": (0.8, 0.3, 0)})
    assert ok
    # a small box near the start jamb: inside the disc hinged at (0.6, 0), outside the one
    # hinged at the other jamb (1.4, 0)
    for hinge, blocked in (("start", True), ("end", False)):
        ph = make_problem([box("a", 0.1, 0.1)], room=box_room(openings=[door(hinge=hinge)]))
        res, _ = results(ph, {"a": (0.7, 0.6, 0)})
        assert res["implicit:door_swing"].satisfied is not blocked


def test_wall_mount():
    tv = box("tv", 1.0, 0.1, 0.6, mount="wall")
    p = make_problem([tv])
    res, ok = results(p, {"tv": (2, 0.05, FACE_N)})
    assert ok and res["implicit:wall_mount"].satisfied
    res, ok = results(p, {"tv": (2, 0.05, FACE_S)})  # facing the wall
    assert not ok and not res["implicit:wall_mount"].satisfied
    res, ok = results(p, {"tv": (2, 1.5, FACE_N)})  # floating mid-room
    assert not ok
    pw = make_problem([tv], room=box_room(openings=[window()]))
    res, ok = results(pw, {"tv": (2, 0.05, FACE_N)})  # hung on the window
    assert not ok and res["implicit:wall_mount"].violation == pytest.approx(1.0)


# ----------------------------------------------------------------------------- vocabulary


def _shelf_problem(h=1.8, room=None, **kw):
    c = {"type": "against_wall", "id": "aw", "a": "s", "hard": True, **kw}
    return make_problem([box("s", 1.0, 0.4, h)], [c], room=room)


def test_against_wall_flush_and_gap():
    p = _shelf_problem()
    res, ok = results(p, {"s": (2, 0.2, FACE_N)})
    assert ok and res["aw"].satisfied
    res, _ = results(p, {"s": (2, 0.24, FACE_N)})  # 4 cm gap, max_gap 5 cm
    assert res["aw"].satisfied
    res, ok = results(p, {"s": (2, 0.4, FACE_N)})  # 20 cm gap
    assert not ok and res["aw"].violation == pytest.approx(0.15)


def test_against_wall_requires_facing_away_from_the_wall():
    p = _shelf_problem()
    res, _ = results(p, {"s": (0.2, 1.5, FACE_E)})  # back on the west wall, facing into room
    assert res["aw"].satisfied
    res, _ = results(p, {"s": (0.5, 1.5, FACE_N)})  # side touches the west wall
    assert not res["aw"].satisfied
    res, _ = results(p, {"s": (2, 0.2, math.radians(15))})  # 15 deg off the wall normal
    assert not res["aw"].satisfied


def test_against_wall_with_wall_id():
    p = _shelf_problem(wall_id="w_n")
    res, _ = results(p, {"s": (2, 0.2, FACE_N)})
    assert not res["aw"].satisfied
    res, _ = results(p, {"s": (2, 2.8, FACE_S)})
    assert res["aw"].satisfied


def test_against_wall_not_across_doors_or_in_front_of_windows_when_tall():
    pd = _shelf_problem(room=box_room(openings=[door(offset=2.0, swing=False)]))
    res, _ = results(pd, {"s": (2, 0.2, FACE_N)})
    assert not res["aw"].satisfied and res["aw"].violation == pytest.approx(0.8)
    tall = _shelf_problem(h=1.8, room=box_room(openings=[window()]))
    low = _shelf_problem(h=0.5, room=box_room(openings=[window()]))
    assert not results(tall, {"s": (2, 0.2, FACE_N)})[0]["aw"].satisfied
    assert results(low, {"s": (2, 0.2, FACE_N)})[0]["aw"].satisfied


def _pair_problem(c):
    return make_problem([box("a", 0.4, 0.4), box("b", 0.4, 0.4)], [c])


def test_faces():
    p = _pair_problem({"type": "faces", "id": "f", "a": "a", "b": "b", "max_angle_deg": 20})
    res, _ = results(p, {"a": (1, 1, FACE_N), "b": (1, 2.5, 0)})
    assert res["f"].satisfied
    bx = 1 + 1.5 * math.tan(math.radians(30))
    res, _ = results(p, {"a": (1, 1, FACE_N), "b": (bx, 2.5, 0)})
    assert res["f"].violation == pytest.approx(math.radians(10), abs=1e-6)
    res, _ = results(p, {"a": (1, 1, FACE_S), "b": (1, 2.5, 0)})  # back to it
    assert res["f"].violation == pytest.approx(math.radians(160), abs=1e-6)


def test_near():
    p = _pair_problem({"type": "near", "id": "n", "a": "a", "b": "b", "max_distance": 0.3})
    res, _ = results(p, {"a": (1, 1, 0), "b": (1.8, 1, 0)})  # 0.4 apart
    assert res["n"].violation == pytest.approx(0.1)
    res, _ = results(p, {"a": (1, 1, 0), "b": (1.6, 1, 0)})  # 0.2 apart
    assert res["n"].satisfied


def test_distance_at_least():
    p = _pair_problem(
        {"type": "distance_at_least", "id": "d", "a": "a", "b": "b", "min_distance": 0.5}
    )
    res, _ = results(p, {"a": (1, 1, 0), "b": (1.8, 1, 0)})  # 0.4 apart
    assert res["d"].violation == pytest.approx(0.1)
    res, _ = results(p, {"a": (1, 1, 0), "b": (2.0, 1, 0)})  # 0.6 apart
    assert res["d"].satisfied


def test_parallel():
    p = _pair_problem({"type": "parallel", "id": "p", "a": "a", "b": "b"})
    assert results(p, {"a": (1, 1, 0), "b": (2, 2, math.pi)})[0]["p"].satisfied
    assert results(p, {"a": (1, 1, 0), "b": (2, 2, math.radians(4))})[0]["p"].satisfied
    res, _ = results(p, {"a": (1, 1, 0), "b": (2, 2, math.pi / 2)})
    assert res["p"].violation == pytest.approx(math.radians(85))


def test_keep_clear_door_and_window():
    room = box_room(openings=[door(offset=1.0, swing=False), window(offset=2.0, wall="w_n")])
    cs = [
        {"type": "keep_clear", "id": "kd", "opening_id": "door", "distance": 0.6},
        {"type": "keep_clear", "id": "kw", "opening_id": "win", "distance": 0.5},
    ]
    p = make_problem([box("lo", 0.4, 0.4, 0.5), box("hi", 0.4, 0.4, 1.5)], cs, room=room)
    res, _ = results(p, {"lo": (1, 0.4, 0), "hi": (3.5, 1.5, 0)})
    assert res["kd"].violation == pytest.approx(0.16) and res["kw"].satisfied
    # only assets taller than the sill count for a window
    res, _ = results(p, {"lo": (2, 2.7, 0), "hi": (3.5, 1.5, 0)})
    assert res["kw"].satisfied and res["kd"].satisfied
    res, _ = results(p, {"lo": (3.5, 1.5, 0), "hi": (2, 2.7, 0)})
    assert not res["kw"].satisfied


def _corridor(blocker: bool):
    """1.2 m wide, 4 m long room: door at the south end, a shelf with an access zone at
    the north end, optionally a cabinet narrowing the corridor to 0.6 m."""
    room = box_room(1.2, 4.0, openings=[door(offset=0.6, width=0.8, swing=False)])
    assets = [box("shelf", 1.0, 0.35, 1.8, zones=[front_zone(1.0, 0.35, 0.6)])]
    poses = {"shelf": (0.6, 3.825, FACE_S)}
    if blocker:
        assets.append(box("cab", 0.6, 0.5, 0.9))
        poses["cab"] = (0.3, 2.0, FACE_E)
    p = make_problem(
        assets, [{"type": "walkway_width", "id": "ww", "min_width": 0.8, "hard": True}], room=room
    )
    return p, poses


def test_walkway_width():
    p, poses = _corridor(blocker=False)
    res, ok = results(p, poses)
    assert ok and res["ww"].satisfied
    p, poses = _corridor(blocker=True)
    res, ok = results(p, poses)
    assert not ok and not res["ww"].satisfied
    assert "unreachable" in res["ww"].detail


def test_walkway_reaches_front_edge_through_another_assets_access_zone():
    """A nightstand-like box tucked in a 0.6 m nook is reachable through the big asset's
    side access zone (one level of chaining); without that zone it is not."""
    side_zone = {"name": "side", "purpose": "access", "polygon": rect(-1.1, -0.6, -0.5, 1.0)}
    room = box_room(3.0, 3.0, openings=[door(offset=1.5, width=0.8, wall="w_e", swing=False)])
    wc = [{"type": "walkway_width", "id": "ww", "min_width": 0.8, "hard": True}]
    poses = {"big": (1.1, 1.0, FACE_N), "small": (0.3, 0.2, FACE_N)}
    with_zone = make_problem(
        [box("big", 1.0, 2.0, zones=[side_zone]), box("small", 0.6, 0.4)], wc, room=room
    )
    no_zone = make_problem([box("big", 1.0, 2.0), box("small", 0.6, 0.4)], wc, room=room)
    assert results(with_zone, poses)[0]["ww"].satisfied
    res, _ = results(no_zone, poses)
    assert not res["ww"].satisfied and "small" in res["ww"].detail


def test_fixed_soft_measures_displacement():
    p = make_problem(
        [box("a", 0.5, 0.5)],
        [{"type": "fixed", "id": "fx", "a": "a", "weight": 2.0}],
        initial={"a": {"x": 1, "y": 1, "yaw": 0}},
    )
    res, _ = results(p, {"a": (1, 1, 0)})
    assert res["fx"].satisfied
    res, _ = results(p, {"a": (1.3, 1.4, 0)})
    assert res["fx"].violation == pytest.approx(0.5)
    _, feasible, cost = evaluate(p, {"a": Pose2D(x=1.3, y=1.4)})
    assert feasible and cost == pytest.approx(2.0 * 0.5)


def test_l_shaped_footprint_with_zones():
    """Non-rectangular footprints (e.g. an L-sofa from the asset pipeline) are used as-is,
    not as their bounding box."""
    l_sofa = box(
        "l",
        2.0,
        1.6,
        footprint={"points": [[-1, -0.8], [1, -0.8], [1, 0.8], [0.2, 0.8], [0.2, 0], [-1, 0]]},
        zones=[{"name": "legroom", "purpose": "use", "polygon": rect(-1, 0, 0.2, 0.5)}],
    )
    p = make_problem([l_sofa, box("t", 0.4, 0.4)])
    # table in the notch of the L but partly inside the legroom zone
    res, ok = results(p, {"l": (2, 1.5, 0), "t": (1.5, 2.1, 0)})
    assert not ok and not res["implicit:clearance_zones"].satisfied
    assert res["implicit:no_overlap"].satisfied  # the notch is not part of the footprint
    # table on the solid arm of the L
    res, ok = results(p, {"l": (2, 1.5, 0), "t": (2.5, 2.1, 0)})
    assert res["implicit:no_overlap"].violation == pytest.approx(0.16)
    # table in the notch, above the legroom zone: fine
    res, ok = results(p, {"l": (2, 1.5, 0), "t": (1.3, 2.25, 0)})
    assert ok


def test_clearance_zone_allowed_categories():
    """Contract v0.2.0: footprints whose category is listed in the zone's
    allowed_categories may enter it (a chair in a desk's chair zone); others may not."""
    chair_zone = front_zone(1.2, 0.6, 0.7, "use", "chair")
    chair_zone["allowed_categories"] = ["chair"]
    desk = box("desk", 1.2, 0.6, zones=[chair_zone], category="table")
    p = make_problem(
        [desk, box("c", 0.5, 0.5, category="chair"), box("bin", 0.5, 0.5, category="storage")]
    )
    poses = {"desk": (2, 0.3, FACE_N), "c": (2, 0.9, FACE_S), "bin": (3.5, 2.5, 0)}
    res, ok = results(p, poses)
    assert ok and res["implicit:clearance_zones"].satisfied
    res, ok = results(p, {**poses, "c": (3.5, 2.5, 0), "bin": (2, 0.9, 0)})
    assert not ok and res["implicit:clearance_zones"].violation == pytest.approx(0.25)


def test_zone_through_the_wall_behind_the_asset_is_clipped():
    """A zone that pokes through the wall the asset stands against (behind its back
    edge, back flush with the wall) is clipped to the floor, not a violation; one cut off
    by a wall in front or at the side, or behind an asset that is merely pushed near a
    wall, still is (docs/solver.md, ZONE_WALL_POLICY = "front")."""
    margin = {"name": "margin", "purpose": "access", "polygon": rect(-0.7, -0.5, 0.7, 0.8)}
    p = make_problem([box("cab", 1.0, 0.6, zones=[margin])])
    res, ok = results(p, {"cab": (2, 0.3, FACE_N)})  # back on the south wall
    assert ok and res["implicit:clearance_zones"].satisfied
    res, ok = results(p, {"cab": (0.3, 1.5, FACE_E)})  # back on the west wall instead
    assert ok  # the 0.2 m of margin behind it is clipped
    res, ok = results(p, {"cab": (2, 0.45, FACE_N)})  # 15 cm off the wall: not "against"
    assert not ok and res["implicit:clearance_zones"].violation == pytest.approx(0.05 * 1.4)
    res, ok = results(p, {"cab": (0.5, 0.3, FACE_N)})  # pushed into the SW corner
    assert not ok  # the margin's left side is cut off by the west wall
    assert res["implicit:clearance_zones"].violation == pytest.approx(0.2 * 1.1)
