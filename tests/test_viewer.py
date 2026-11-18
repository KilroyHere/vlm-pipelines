"""roomlayout.viewer: scene export, lock checks, and the edit server's endpoints."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from roomlayout.schema import LayoutProblem
from roomlayout.solver import solve
from roomlayout.viewer import export, scene_json
from roomlayout.viewer.__main__ import check_lock, locked_problem, serve

EX = Path(__file__).resolve().parents[1] / "examples" / "problems"


@pytest.fixture(scope="module")
def solved():
    p = LayoutProblem.model_validate_json((EX / "living_room_basic.json").read_text())
    return p, solve(p, k=2, seed=0, time_budget_s=3)


def test_scene_json_has_room_assets_layouts_and_lines(solved):
    p, res = solved
    s = scene_json(p, res, fit={"fits": True, "advice": []}, request="sofa facing the TV")
    assert s["version"] == 1 and s["request"] == "sofa facing the TV"
    assert len(s["room"]["walls"]) == len(p.room.walls)
    assert all(w["name"].endswith("wall") for w in s["room"]["walls"])
    assert {a["id"] for a in s["assets"]} == {a.id for a in p.assets}
    lay = s["layouts"][0]
    assert set(lay["placements"]) == {a.id for a in p.assets}
    assert lay["lines"] and {ln["kind"] for ln in lay["lines"]} <= {"why", "bent", "broken"}
    json.dumps(s)  # JSON-ready


def test_export_writes_page_scene_and_problem(solved, tmp_path):
    p, res = solved
    page = export(p, res, tmp_path / "v")
    names = {f.name for f in (tmp_path / "v").iterdir()}
    assert page.name == "index.html"
    assert {"index.html", "viewer.js", "viewer.css", "scene.json", "problem.json"} <= names
    html = page.read_text()
    assert "three@0.160.0" in html  # pinned version


def test_lock_inside_room_ok_outside_refused(solved):
    p, res = solved
    pose = next(pl.pose for pl in res.layouts[0].placements if pl.asset_id == "sofa")
    ok = {"id": "sofa", "x": pose.x, "y": pose.y, "yaw": pose.yaw}
    check_lock(p, ok)
    lp = locked_problem(p, [ok])
    assert any(c.id == "lock:sofa" and c.hard and c.origin == "user" for c in lp.constraints)
    assert lp.initial["sofa"].x == pytest.approx(pose.x)
    with pytest.raises(ValueError, match="sticks out of the room"):
        check_lock(p, {**ok, "x": -5.0})
    with pytest.raises(KeyError):
        locked_problem(p, [{**ok, "id": "nope"}])


def test_server_ping_evaluate_and_resolve(solved, tmp_path):
    p, res = solved
    export(p, res, tmp_path / "v")
    httpd = serve(tmp_path / "v", port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})  # fmt: skip
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    try:
        with urllib.request.urlopen(base + "/api/ping", timeout=10) as r:
            assert json.loads(r.read())["ok"]
        with urllib.request.urlopen(base + "/scene.json", timeout=10) as r:
            assert json.loads(r.read())["version"] == 1
        placements = {
            pl.asset_id: {"x": pl.pose.x, "y": pl.pose.y, "yaw": pl.pose.yaw}
            for pl in res.layouts[0].placements
        }
        ev = post("/api/evaluate", {"placements": placements})
        assert ev["id"] == "edited" and ev["feasible"] is res.layouts[0].feasible
        sofa = {"id": "sofa", **placements["sofa"]}
        out = post("/api/solve", {"locks": [sofa]})
        assert out["layouts"]
        for lay in out["layouts"]:
            assert lay["placements"]["sofa"]["x"] == pytest.approx(sofa["x"])
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/solve", {"locks": [{**sofa, "x": -9.0}]})
        assert e.value.code == 400
    finally:
        httpd.shutdown()
