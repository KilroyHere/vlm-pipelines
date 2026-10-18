"""RoomPlan ingest (M4a): synthetic captures with known geometry -> LayoutProblem."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from roomlayout.ingest.roomplan import IngestError, convert
from roomlayout.ingest.roomplan.__main__ import main as cli
from roomlayout.ingest.roomplan.geometry import to_ark, to_ours, yaw_from_front
from roomlayout.schema import LayoutProblem, OpeningType

from . import roomplan_synth as R


def _front(pose) -> np.ndarray:
    return np.array([-math.sin(pose.yaw), math.cos(pose.yaw)])


def _asset(problem: LayoutProblem, aid: str):
    return next(a for a in problem.assets if a.id == aid)


# ------------------------------------------------------------------------------- frame


def test_frame_conversion_is_a_proper_rotation():
    v = np.array([[1.0, 2.0, 3.0], [-0.5, 0.1, 4.0]])
    assert np.allclose(to_ark(to_ours(v)), v)
    assert np.allclose(to_ours([0, 1, 0]), [0, 0, 1])  # ARKit up -> our up
    assert np.isclose(np.linalg.det(np.array([to_ours(e) for e in np.eye(3)])), 1.0)
    assert np.allclose(_front_from_yaw(yaw_from_front((0, -1))), (0, -1))


def _front_from_yaw(yaw):
    return (-math.sin(yaw), math.cos(yaw))


# ------------------------------------------------------------------------------- room


def test_rectangular_room_walls_floor_and_ceiling():
    res = convert(R.rect_room(5, 4, 2.6), problem_id="r")
    room = res.problem.room
    assert [w.id for w in room.walls] == ["wall_1", "wall_2", "wall_3", "wall_4"]
    assert room.walls[0].start == pytest.approx((0, 0)) and room.walls[0].end == pytest.approx(
        (5, 0)
    )
    assert room.ceiling_height == pytest.approx(2.6)
    assert not room.openings and not room.obstacles
    assert "floor polygon from RoomPlan floor polygonCorners" in res.notes


def test_floor_from_wall_loop_when_no_floor_polygon():
    res = convert(R.rect_room(4, 3, with_floor=False))
    pts = np.array(res.problem.room.floor.points)
    assert pts.min(axis=0) == pytest.approx((0, 0), abs=1e-6)
    assert pts.max(axis=0) == pytest.approx((4, 3), abs=1e-6)
    assert any("wall loop" in n for n in res.notes)


def test_doors_windows_and_openings_land_on_their_walls():
    cap = R.rect_room(5, 4)
    south, east = cap["walls"][0], cap["walls"][1]
    cap["doors"].append(R.surface((1.0, 0), (1.9, 0), 2.1, "door", {"isOpen": False},
                                  parent=south["identifier"]))  # fmt: skip
    cap["windows"].append(R.surface((5, 1.0), (5, 2.5), 1.2, "window", z0=0.9,
                                    parent=east["identifier"]))  # fmt: skip
    cap["doors"].append(R.surface((2.5, 4), (4.5, 4), 2.1, "door", {"isOpen": False}))
    res = convert(cap)
    by = {o.id: o for o in res.problem.room.openings}
    door, window, patio = by["door_1"], by["window_1"], by["door_2"]
    assert door.wall_id == "wall_1" and door.offset == pytest.approx(1.45, abs=1e-3)
    assert door.width == pytest.approx(0.9, abs=1e-3) and door.sill_height == 0.0
    assert door.swing is not None and door.swing.into_room
    assert window.type == OpeningType.window and window.wall_id == "wall_2"
    assert window.sill_height == pytest.approx(0.9, abs=1e-3)
    assert patio.wall_id == "wall_3" and patio.swing is None  # 2 m wide: sliding/double
    assert res.source_ids["door_1"] == cap["doors"][0]["identifier"]


def test_unwalled_side_becomes_a_passage():
    cap = R.rect_room(5, 4)
    del cap["walls"][2]  # the north side was never scanned as a wall (open plan)
    res = convert(cap)
    passages = [o for o in res.problem.room.openings if o.type == OpeningType.opening]
    assert len(passages) == 1 and passages[0].wall_id == "wall_3"
    assert passages[0].width == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------- objects


def test_objects_keep_pose_size_and_front():
    cap = R.rect_room(5, 4)
    cap["objects"] += [
        R.obj("sofa", (2.5, 3.5), -90, 2.0, 0.9, 0.85),  # back to the north wall, facing south
        R.obj("table", (2.5, 2.3), 0, 1.0, 0.6, 0.45, attributes={"TableType": "coffee"}),
        R.obj("chair", (1.0, 1.0), 45, 0.5, 0.5, 0.9, attributes={"ChairType": "dining"}),
        R.obj("storage", (4.7, 2.0), 180, 0.8, 0.4, 1.8, attributes={"StorageType": "shelf"}),
    ]
    p = convert(cap).problem
    assert [a.id for a in p.assets] == ["sofa_1", "coffee_table_1", "dining_chair_1", "bookshelf_1"]
    sofa = _asset(p, "sofa_1")
    assert (sofa.dimensions.width, sofa.dimensions.depth) == pytest.approx((2.0, 0.9))
    assert sofa.source.dimensions_from == "roomplan" and sofa.movable
    assert p.initial["sofa_1"].x == pytest.approx(2.5) and p.initial["sofa_1"].y == pytest.approx(
        3.5
    )
    assert _front(p.initial["sofa_1"]) == pytest.approx((0, -1), abs=1e-4)
    assert _front(p.initial["dining_chair_1"]) == pytest.approx((0.7071, 0.7071), abs=1e-3)
    assert _front(p.initial["bookshelf_1"]) == pytest.approx((-1, 0), abs=1e-4)


def test_fixtures_and_built_in_cabinetry_are_immovable():
    cap = R.rect_room(5, 4)
    cap["objects"] += [
        R.obj("oven", (0.3, 2.0), 0, 0.6, 0.6, 0.9),
        R.obj("storage", (0.3, 2.65), 0, 0.7, 0.6, 0.9),  # touches the oven: kitchen run
        R.obj("storage", (0.3, 3.35), 0, 0.7, 0.6, 0.9),  # touches that cabinet: same run
        R.obj("storage", (0.2, 1.0), 0, 0.6, 0.35, 0.7, bottom=1.5),  # wall cabinet
        R.obj("storage", (4.0, 0.3), 90, 1.2, 0.45, 0.8),  # free-standing sideboard
        R.obj("fireplace", (2.5, 0.2), 90, 1.5, 0.4, 1.1),
        R.obj("stairs", (4.0, 3.0), 180, 1.0, 1.5, 2.5),
    ]
    res = convert(cap)
    p = res.problem
    fixed = {a.id for a in p.assets if not a.movable}
    assert fixed == {"oven_1", "storage_1", "storage_2", "storage_3", "fireplace_1"}
    assert _asset(p, "storage_3").mount.value == "wall"
    assert _asset(p, "storage_3").elevation == pytest.approx(1.5)
    assert _asset(p, "storage_4").movable
    assert [o.label for o in p.room.obstacles] == ["stairs"]


def test_tv_on_a_cabinet_makes_it_the_tv_stand():
    cap = R.rect_room(5, 4)
    cap["objects"] += [
        R.obj("storage", (2.5, 0.25), 90, 1.6, 0.45, 0.5, attributes={"StorageType": "cabinet"}),
        R.obj("television", (2.5, 0.25), 90, 1.2, 0.08, 0.7, bottom=0.5),
        R.obj("television", (0.06, 2.0), 0, 1.2, 0.08, 0.7, bottom=1.2),  # on the wall
    ]
    p = convert(cap).problem
    assert {a.subtype for a in p.assets if a.category == "storage"} == {"tv_stand"}
    tvs = sorted((a for a in p.assets if a.category == "television"), key=lambda a: a.elevation)
    assert tvs[0].mount.value == "floor" and tvs[0].elevation == pytest.approx(0.5)
    assert tvs[1].mount.value == "wall"


def test_raised_tv_away_from_walls_is_not_wall_mounted():
    cap = R.rect_room(5, 4)
    cap["objects"].append(R.obj("television", (2.5, 2.0), 90, 1.2, 0.08, 0.7, bottom=0.9))
    res = convert(cap)
    assert res.problem.assets[0].mount.value == "floor"
    assert any("unscanned furniture" in n for n in res.notes)


def test_sofa_extensions_merge_into_one_sectional():
    cap = R.rect_room(6, 5)
    # main piece 3 m wide facing south against the north wall, chaise at its west end
    cap["objects"] += [
        R.obj("sofa", (3.0, 4.55), -90, 3.0, 0.9, 0.85, attributes={"SofaType": "lShaped"}),
        R.obj(
            "sofa", (1.92, 3.48), 0, 1.2, 0.84, 0.85, attributes={"SofaType": "lShapedExtension"}
        ),  # fmt: skip
    ]
    res = convert(cap)
    (sofa,) = res.problem.assets
    assert sofa.subtype == "sectional"
    assert len(sofa.footprint.points) == 6  # an L
    assert sofa.dimensions.width == pytest.approx(3.0, abs=0.02)
    assert sofa.dimensions.depth == pytest.approx(0.9 + 1.2, abs=0.05)
    assert any("merged sofa extension" in n for n in res.notes)


def test_objects_outside_the_floor():
    cap = R.rect_room(5, 4)
    cap["objects"] += [
        R.obj("chair", (5.05, 2.0), 180, 0.5, 0.5, 0.9),  # slightly outside, facing in
        R.obj("television", (5.3, 1.0), 0, 1.0, 0.1, 0.6, bottom=1.0),  # next room, facing away
    ]
    res = convert(cap)
    (chair,) = res.problem.assets
    pose = res.problem.initial[chair.id]
    assert pose.x + chair.dimensions.depth / 2 <= 5.0  # pulled inside the east wall
    assert any("next room" in w for w in res.warnings)
    assert any("moved just inside" in w for w in res.warnings)


# ----------------------------------------------------------------------- odd captures


def test_structure_with_several_rooms_needs_a_room_index():
    room = R.rect_room(4, 3)
    structure = {"rooms": [room, R.rect_room(3, 3)], "walls": [], "doors": [], "windows": []}
    with pytest.raises(IngestError, match="2 rooms"):
        convert(structure)
    res = convert(structure, room=1)
    assert np.array(res.problem.room.floor.points).max(axis=0) == pytest.approx((3, 3))


def test_multi_story_capture_is_rejected():
    cap = R.rect_room(4, 3)
    cap["walls"][0]["story"] = 1
    with pytest.raises(IngestError, match="several stories"):
        convert(cap)


def test_malformed_elements_are_skipped_with_a_warning():
    cap = R.rect_room(4, 3)
    bad = R.obj("chair", (1, 1), 0, 0.5, 0.5, 0.9)
    bad["transform"][0] = 3.0  # not a rotation
    cap["objects"].append(bad)
    res = convert(cap)
    assert not res.problem.assets
    assert any("objects[0] skipped" in w for w in res.warnings)


def test_not_a_roomplan_file():
    with pytest.raises(IngestError, match="not a RoomPlan"):
        convert({"hello": "world"})


# --------------------------------------------------------------------------------- CLI


def test_cli_writes_problem_and_report(tmp_path):
    cap = R.rect_room(5, 4)
    cap["objects"].append(R.obj("sofa", (2.5, 3.5), -90, 2.0, 0.9, 0.85))
    src = tmp_path / "Room.json"
    src.write_text(json.dumps(cap))
    out, rep = tmp_path / "p.json", tmp_path / "r.json"
    assert cli([str(src), "--out", str(out), "--report", str(rep)]) == 0
    p = LayoutProblem.model_validate_json(out.read_text())
    assert p.id == "Room" and [a.id for a in p.assets] == ["sofa_1"]
    assert json.loads(rep.read_text())["source_ids"]["sofa_1"] == cap["objects"][0]["identifier"]
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert cli([str(bad), "--out", str(tmp_path / "x.json")]) == 2


def test_partition_wall_keeps_a_gap_at_its_door():
    cap = R.rect_room(6, 4)
    part = R.surface((3, 0.5), (3, 3.5), 2.6)  # floor on both sides
    cap["walls"].append(part)
    cap["doors"].append(R.surface((3, 1.5), (3, 2.4), 2.1, "door", {"isOpen": True},
                                  parent=part["identifier"]))  # fmt: skip
    res = convert(cap)
    parts = res.problem.room.obstacles
    assert len(parts) == 2 and all(o.label == "partition wall" for o in parts)
    ys = sorted(max(p[1] for p in o.footprint.points) for o in parts)
    assert ys[0] == pytest.approx(1.5, abs=0.06)  # the lower piece stops at the door
    assert not res.problem.room.openings  # the door is a gap, not a boundary opening
    assert any("several rooms" in w for w in res.warnings)
