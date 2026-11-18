"""roomlayout.explain: measured, plain-English reasons per layout."""

from __future__ import annotations

import math
from pathlib import Path

from roomlayout.explain import explain_layout
from roomlayout.schema import Layout, LayoutProblem, Placement, Pose2D

EX = Path(__file__).resolve().parents[1] / "examples" / "problems"


def _problem() -> LayoutProblem:
    """living_room_basic with its own constraints, plus a soft wish we will break."""
    p = LayoutProblem.model_validate_json((EX / "living_room_basic.json").read_text())
    data = p.model_dump(mode="json")
    data["constraints"].append(
        {"type": "near", "id": "u_near", "a": "armchair", "b": "bookshelf", "max_distance": 0.1,
         "hard": False, "origin": "user", "rationale": "the armchair next to the bookshelf"}
    )  # fmt: skip
    return LayoutProblem.model_validate(data)


def _layout(p: LayoutProblem) -> Layout:
    from roomlayout.solver import solve

    res = solve(p, k=1, seed=0, time_budget_s=4)
    return res.layouts[0]


def test_lines_measure_and_rank():
    p = _problem()
    lay = _layout(p)
    lines = explain_layout(p, lay)
    assert lines and all(x.kind in ("why", "bent", "broken") for x in lines)
    # Ordered: broken, then bent, then why; user lines before rule-library lines per kind.
    ranks = [{"broken": 0, "bent": 1, "why": 2}[x.kind] for x in lines]
    assert ranks == sorted(ranks)
    faces = [x for x in lines if x.constraint_id and "faces" in x.constraint_id]
    assert faces and "° off" in faces[0].text and " m away" in faces[0].text
    assert any("reachable along paths at least" in x.text for x in lines)


def test_bent_soft_wish_is_reported_with_the_wish():
    p = _problem()
    lay = _layout(p)
    # Move the armchair far from the bookshelf by hand: the soft wish is now bent.
    poses = {pl.asset_id: pl.pose for pl in lay.placements}
    far = Pose2D(x=poses["bookshelf"].x + 2.5, y=poses["bookshelf"].y, yaw=0.0)
    from roomlayout.solver import evaluate

    poses["armchair"] = far
    results, feasible, cost = evaluate(p, poses)
    moved = Layout(
        id="moved", placements=[Placement(asset_id=k, pose=v) for k, v in poses.items()],
        cost=cost, feasible=feasible, results=results,
    )  # fmt: skip
    lines = explain_layout(p, moved)
    bent = [x for x in lines if x.constraint_id == "u_near"]
    assert bent and bent[0].kind == "bent" and bent[0].origin == "user"
    assert "Wanted: the armchair next to the bookshelf." in bent[0].text
    assert bent[0].assets == ("armchair", "bookshelf")
    assert not math.isnan(cost)
