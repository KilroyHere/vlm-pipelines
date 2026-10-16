import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from roomlayout.schema import Asset, Dimensions, LayoutProblem, Polygon2D, Pose2D
from roomlayout.schema.export import render

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples" / "problems").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_validate(path):
    LayoutProblem.model_validate_json(path.read_text())


def test_committed_json_schemas_are_current():
    for filename, text in render().items():
        committed = ROOT / "schemas" / filename
        assert committed.exists(), f"run: python -m roomlayout.schema.export ({filename} missing)"
        assert committed.read_text() == text, (
            f"stale {filename}: run python -m roomlayout.schema.export"
        )


def test_polygon_normalized_to_ccw_and_open():
    cw_closed = Polygon2D(points=[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    assert len(cw_closed.points) == 4
    assert cw_closed.area == pytest.approx(1.0)


def test_degenerate_polygon_rejected():
    with pytest.raises(ValidationError):
        Polygon2D(points=[(0, 0), (1, 1), (2, 2)])


def test_pose_front_and_transform():
    p = Pose2D(x=1, y=2, yaw=math.pi / 2)  # front (+Y local) now points to -X
    fx, fy = p.front
    assert (fx, fy) == pytest.approx((-1.0, 0.0))
    assert p.transform((0, 1)) == pytest.approx((0.0, 2.0))


def test_asset_default_footprint_is_bounding_rectangle():
    a = Asset(id="t", category="table", dimensions=Dimensions(width=2, depth=1, height=0.7))
    assert a.effective_footprint().area == pytest.approx(2.0)


def _example_dict():
    return json.loads((ROOT / "examples/problems/living_room_basic.json").read_text())


def test_unknown_asset_reference_rejected():
    data = _example_dict()
    data["constraints"].append(
        {"type": "near", "id": "bad", "a": "sofa", "b": "piano", "max_distance": 1}
    )
    with pytest.raises(ValidationError, match="unknown asset"):
        LayoutProblem.model_validate(data)


def test_fixed_requires_initial_pose():
    data = _example_dict()
    data["constraints"].append({"type": "fixed", "id": "pin", "a": "sofa"})
    with pytest.raises(ValidationError, match="initial pose"):
        LayoutProblem.model_validate(data)


def test_opening_must_fit_wall():
    data = _example_dict()
    data["room"]["openings"][0]["offset"] = 4.9
    with pytest.raises(ValidationError, match="does not fit"):
        LayoutProblem.model_validate(data)
