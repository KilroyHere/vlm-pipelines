"""Stage composition (ADR-0008): RoomPlan scan -> zones -> rules -> layouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from roomlayout.assets import with_default_zones
from roomlayout.pipeline import main, plan, problem_from_roomplan
from roomlayout.schema import Asset, Dimensions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
import roomplan_synth as R  # noqa: E402


def _living_room(tmp_path) -> Path:
    cap = R.rect_room(5, 4)
    cap["doors"].append(R.surface((0.3, 0), (1.2, 0), 2.1, "door", {"isOpen": False}))
    cap["objects"] += [
        R.obj("sofa", (2.5, 3.5), -90, 2.0, 0.9, 0.85),
        R.obj("table", (2.5, 2.3), 0, 1.0, 0.6, 0.45, attributes={"TableType": "coffee"}),
        R.obj("storage", (2.5, 0.25), 90, 1.6, 0.45, 0.5),
        R.obj("television", (2.5, 0.25), 90, 1.2, 0.08, 0.7, bottom=0.5),
    ]
    path = tmp_path / "Room.json"
    path.write_text(json.dumps(cap))
    return path


def test_default_zones_fill_only_missing_zones():
    a = Asset(id="s", category="sofa", dimensions=Dimensions(width=2, depth=0.9, height=0.8))
    with_zones = with_default_zones(a)
    assert [z.name for z in with_zones.clearance_zones] == ["legroom"]
    assert with_default_zones(with_zones) is with_zones


def test_scan_to_feasible_layouts_with_rules(tmp_path):
    problem, ingest = problem_from_roomplan(_living_room(tmp_path))
    assert {a.subtype for a in problem.assets} >= {"coffee_table", "tv_stand"}
    assert all(a.clearance_zones for a in problem.assets if a.category == "sofa")
    built, result = plan(problem, k=2, seconds=8.0)
    assert built.from_rules and result.layouts and result.layouts[0].feasible, result.notes


def test_cli_writes_layouts(tmp_path):
    out = tmp_path / "out"
    assert main([str(_living_room(tmp_path)), "--out", str(out), "--seconds", "6", "--k", "1"]) == 0
    for name in (
        "problem.json",
        "ingest_report.json",
        "result.json",
        "current.png",
        "layout_1.png",
    ):
        assert (out / name).exists(), name


def test_empty_room_keeps_fixtures_and_furnish_adds_new_furniture(tmp_path):
    from roomlayout.pipeline import empty_room, furnish, load_asset

    cap = json.loads(_living_room(tmp_path).read_text())
    cap["objects"].append(R.obj("fireplace", (0.25, 2.0), 0, 1.2, 0.4, 1.0))
    (tmp_path / "Room.json").write_text(json.dumps(cap))
    problem, _ = problem_from_roomplan(tmp_path / "Room.json")

    room, removed = empty_room(problem)
    assert [a.id for a in room.assets] == ["fireplace_1"] and "fireplace_1" in room.initial
    assert set(removed) == {"sofa_1", "coffee_table_1", "tv_stand_1", "television_1"}

    dims = Dimensions(width=2.2, depth=0.95, height=0.85)
    new = Asset(id="sofa", category="sofa", dimensions=dims)
    (tmp_path / "sofa.asset.json").write_text(new.model_dump_json())
    sofa = load_asset(tmp_path / "sofa.asset.json")
    furnished = furnish(room, [sofa, sofa])
    assert [a.id for a in furnished.assets] == ["fireplace_1", "sofa", "sofa_2"]
    assert all(a.clearance_zones for a in furnished.assets if a.category == "sofa")
    _, result = plan(furnished, k=1, seconds=6.0)
    assert result.layouts[0].feasible, result.notes


def test_cli_empty_and_add(tmp_path):
    new = Asset(id="armchair", category="chair", subtype="armchair",
                dimensions=Dimensions(width=0.8, depth=0.8, height=0.9))  # fmt: skip
    asset = tmp_path / "armchair.asset.json"
    asset.write_text(new.model_dump_json())
    out = tmp_path / "out"
    args = [str(_living_room(tmp_path)), "--out", str(out), "--seconds", "5", "--k", "1"]
    assert main([*args, "--empty", "--add", str(asset)]) == 0
    placed = json.loads((out / "result.json").read_text())["layouts"][0]["placements"]
    assert [p["asset_id"] for p in placed] == ["armchair"]
    assert not (out / "current.png").exists()
