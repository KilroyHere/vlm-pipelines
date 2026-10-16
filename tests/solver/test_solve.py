"""solve(): determinism, pinned assets, result consistency, infeasibility notes, CLI."""

import json
import math
from pathlib import Path

import pytest
from solver_helpers import box, box_room, front_zone, make_problem

from roomlayout.schema import LayoutProblem, SolveResult
from roomlayout.solver import evaluate, solve, solve_detailed
from roomlayout.solver.__main__ import main as cli_main
from roomlayout.solver.diversity import interchangeable_groups, layout_distance, select_diverse
from roomlayout.solver.model import Model

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "problems"
# a fixed amount of work (no wall-clock dependence) so results must be bit-identical
FIXED_WORK = {"min_restarts": 6, "max_restarts": 6, "iterations": 300}


def _small_problem():
    return make_problem(
        [
            box("sofa", 1.8, 0.8, zones=[front_zone(1.8, 0.8, 0.4, "use")]),
            box("tv", 1.2, 0.4, 0.5),
            box("table", 0.8, 0.5, 0.4),
            box("chair", 0.6, 0.6),
        ],
        [
            {"type": "faces", "id": "f", "a": "sofa", "b": "tv", "hard": True},
            {"type": "against_wall", "id": "aw", "a": "tv", "hard": True},
            {"type": "near", "id": "n", "a": "table", "b": "sofa", "max_distance": 0.5},
            {"type": "faces", "id": "f2", "a": "chair", "b": "table", "weight": 0.5},
            {"type": "walkway_width", "id": "ww", "min_width": 0.8, "hard": True},
        ],
        room=box_room(
            4.5,
            3.5,
            openings=[
                {
                    "id": "door",
                    "type": "door",
                    "wall_id": "w_s",
                    "offset": 0.6,
                    "width": 0.8,
                    "height": 2.0,
                    "swing": {"into_room": True, "hinge": "start"},
                }
            ],
        ),
    )


def _strip_time(r: SolveResult) -> dict:
    d = r.model_dump()
    d["stats"].pop("seconds")
    return d


def test_deterministic_for_a_seed_and_independent_of_workers():
    p = _small_problem()
    a = solve(p, k=3, seed=5, time_budget_s=120, workers=1, **FIXED_WORK)
    b = solve(p, k=3, seed=5, time_budget_s=120, workers=1, **FIXED_WORK)
    c = solve(p, k=3, seed=5, time_budget_s=120, workers=2, **FIXED_WORK)
    assert _strip_time(a) == _strip_time(b) == _strip_time(c)
    d = solve(p, k=3, seed=6, time_budget_s=120, workers=1, **FIXED_WORK)
    assert _strip_time(a) != _strip_time(d)


def test_result_is_consistent_with_independent_evaluation():
    p = _small_problem()
    out = solve_detailed(p, k=3, seed=0, time_budget_s=60, workers=1, **FIXED_WORK)
    r = out.result
    assert r.layouts and r.layouts[0].feasible
    assert out.diagnostics.time_to_first_feasible_s is not None
    costs = [lay.cost for lay in r.layouts]
    feas = [lay.feasible for lay in r.layouts]
    assert feas == sorted(feas, reverse=True)
    assert costs[: sum(feas)] == sorted(costs[: sum(feas)])
    ids = {c.id for c in p.constraints}
    for lay in r.layouts:
        assert {pl.asset_id for pl in lay.placements} == {a.id for a in p.assets}
        assert ids <= {res.constraint_id for res in lay.results}
        if lay.feasible:
            assert all(res.satisfied for res in lay.results if res.hard)
        res, ok, cost = evaluate(p, {pl.asset_id: pl.pose for pl in lay.placements})
        assert ok == lay.feasible and cost == pytest.approx(lay.cost)
    SolveResult.model_validate_json(r.model_dump_json())  # round-trips


def test_pinned_assets_stay_at_initial_pose():
    p = make_problem(
        [
            box("piano", 1.5, 0.6, 1.2),
            box("builtin", 1.0, 0.4, movable=False),
            box("chair", 0.5, 0.5),
        ],
        [
            {"type": "fixed", "id": "fx", "a": "piano", "hard": True},
            {
                "type": "near",
                "id": "n",
                "a": "chair",
                "b": "piano",
                "max_distance": 0.3,
                "hard": True,
            },
        ],
        initial={
            "piano": {"x": 1.0, "y": 2.7, "yaw": math.pi},
            "builtin": {"x": 3.8, "y": 1.5, "yaw": math.pi / 2},
        },
    )
    r = solve(p, k=2, seed=0, time_budget_s=60, workers=1, **FIXED_WORK)
    for lay in r.layouts:
        poses = {pl.asset_id: pl.pose for pl in lay.placements}
        assert (poses["piano"].x, poses["piano"].y, poses["piano"].yaw) == (1.0, 2.7, math.pi)
        assert (poses["builtin"].x, poses["builtin"].y) == (3.8, 1.5)
    assert r.layouts[0].feasible


def test_infeasible_problem_explained():
    p = make_problem(
        [box("a", 1, 1), box("b", 1, 1)],
        [
            {"type": "near", "id": "close", "a": "a", "b": "b", "max_distance": 0.2, "hard": True},
            {
                "type": "distance_at_least",
                "id": "far",
                "a": "a",
                "b": "b",
                "min_distance": 1.0,
                "hard": True,
            },
        ],
    )
    r = solve(
        p, k=3, seed=0, time_budget_s=30, workers=1, min_restarts=4, max_restarts=4, iterations=200
    )
    assert r.layouts and not any(lay.feasible for lay in r.layouts)
    text = " ".join(r.notes)
    assert "no feasible layout" in text
    assert "far" in text and "close" in text


def test_area_overflow_noted():
    p = make_problem([box("a", 3.5, 2.5), box("b", 3.5, 2.5)])
    r = solve(
        p, k=1, seed=0, time_budget_s=30, workers=1, min_restarts=2, max_restarts=2, iterations=100
    )
    assert any("footprints total" in n for n in r.notes)


def test_layout_distance_ignores_swapping_identical_assets():
    p = make_problem([box("c1", 0.5, 0.5), box("c2", 0.5, 0.5), box("t", 1, 1)])
    groups = interchangeable_groups(Model(p))
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]
    a = [(1, 1, 0), (3, 1, 0), (2, 2, 0)]
    swapped = [(3, 1, 0), (1, 1, 0), (2, 2, 0)]
    assert layout_distance(a, swapped, groups) == pytest.approx(0)
    moved = [(1, 1, 0), (3, 1, 0), (2, 0.5, 0)]
    assert layout_distance(a, moved, groups) == pytest.approx(1.5 / 3)
    cands = [(a, 0.0, True), (swapped, 0.1, True), (moved, 0.2, True)]
    assert select_diverse(cands, k=3, groups=groups) == [0, 2]


def test_cli_writes_result_and_pngs(tmp_path):
    src = tmp_path / "p.json"
    src.write_text(_small_problem().model_dump_json())
    rc = cli_main(
        [str(src), "--out", str(tmp_path / "out"), "--k", "2", "--budget", "4", "--workers", "1"]
    )
    assert rc == 0
    result = SolveResult.model_validate_json((tmp_path / "out" / "result.json").read_text())
    assert len(result.layouts) >= 1
    for i in range(len(result.layouts)):
        png = tmp_path / "out" / f"layout_{i:02d}.png"
        assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    assert json.loads((tmp_path / "out" / "diagnostics.json").read_text())["restarts"] >= 1


def test_basic_example_solves_quickly():
    p = LayoutProblem.model_validate_json((EXAMPLES / "living_room_basic.json").read_text())
    out = solve_detailed(p, k=3, seed=0, time_budget_s=10, workers=1, min_restarts=2)
    assert out.result.layouts[0].feasible
