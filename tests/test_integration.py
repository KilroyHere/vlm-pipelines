"""Integration checkpoint 1 (docs/PROJECT_PLAN.md): the stages fit together.

A raw "generator" mesh (SAM 3D conventions: Y-up, arbitrary yaw/scale/offset) goes
through asset normalization, is written to disk, read back as an ``Asset`` JSON,
dropped into a ``LayoutProblem`` and solved. Only the public APIs and the files on
disk connect the stages, the same way the product will.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from roomlayout.assets import normalize, write_outputs
from roomlayout.schema import Asset, Dimensions, LayoutProblem
from roomlayout.solver import solve

ROOT = Path(__file__).resolve().parents[1]
SAM3D_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
SECTIONAL = Dimensions(width=2.6, depth=1.6, height=0.85)


def _box(x0, x1, y0, y1, z0, z1):
    box = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    box.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return box


def _raw_sectional() -> trimesh.Trimesh:
    """L-shaped sectional, front = +Y, then posed like generator output."""
    w, d = SECTIONAL.width / 2, SECTIONAL.depth / 2
    parts = [
        _box(-w, w, -d, -d + 0.9, 0.0, 0.45),  # main seat
        _box(-w, -w + 0.9, -d + 0.9, d, 0.0, 0.45),  # chaise
        _box(-w, w, -d, -d + 0.2, 0.45, 0.85),  # backrest
        _box(w - 0.2, w, -d + 0.2, -d + 0.9, 0.45, 0.65),  # arm
    ]
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(2.1, (0, 0, 1)))
    mesh.apply_scale(0.37)
    mesh.apply_translation((0.4, -1.2, 0.7))
    mesh.vertices = mesh.vertices @ SAM3D_ZUP_TO_YUP
    return mesh


@pytest.fixture(scope="module")
def normalized_asset(tmp_path_factory) -> tuple[Asset, Path]:
    out = tmp_path_factory.mktemp("asset")
    result = normalize(
        _raw_sectional(), category="sofa", dims=SECTIONAL, asset_id="sofa", label="sectional"
    )
    paths = write_outputs(result, out, render=False)
    return Asset.model_validate_json(Path(paths["asset"]).read_text()), out


def test_asset_round_trips_through_disk(normalized_asset):
    asset, out = normalized_asset
    assert (out / asset.mesh.uri).exists()
    dims = asset.dimensions
    assert (dims.width, dims.depth, dims.height) == pytest.approx((2.6, 1.6, 0.85), abs=1e-3)
    footprint = Polygon(asset.effective_footprint().points)
    assert footprint.area < 0.85 * dims.width * dims.depth  # the L survived


def test_generated_asset_solves_in_a_real_problem(normalized_asset):
    asset, _ = normalized_asset
    data = json.loads((ROOT / "examples/problems/living_room_basic.json").read_text())
    data["assets"] = [
        asset.model_dump(mode="json") if a["id"] == "sofa" else a for a in data["assets"]
    ]
    problem = LayoutProblem.model_validate(data)

    result = solve(problem, k=3, seed=0, time_budget_s=8.0)

    assert result.layouts and result.layouts[0].feasible, result.notes
    placed = {p.asset_id for p in result.layouts[0].placements}
    assert placed == {a.id for a in problem.assets}


def test_asset_category_table_matches_shared_taxonomy():
    from roomlayout.assets import categories
    from roomlayout.schema import taxonomy

    assert set(categories.ROOMPLAN_CATEGORIES) == set(taxonomy.ROOMPLAN_CATEGORIES)
    for key, profile in categories.PROFILES.items():
        assert taxonomy.resolve(key) is not None, key
        assert taxonomy.resolve(key)[0] == profile.roomplan, key
    for alias in categories.ALIASES:
        assert taxonomy.resolve(alias) is not None, alias


# --- Integration checkpoint 2: user words -> constraints -> layouts (M3c) -------------


def _minimal_living_room() -> LayoutProblem:
    data = json.loads((ROOT / "examples/problems/living_room_basic.json").read_text())
    data["constraints"] = []
    return LayoutProblem.model_validate(data)


def _vlm_answer(constraints, unexpressed=()):
    """A SYNTHETIC structured-output answer, served through the real SDK offline."""
    from roomlayout.constraints.vlm.offline import offline_client, synthetic_response

    output = {"constraints": list(constraints), "unexpressed": list(unexpressed)}
    return offline_client([synthetic_response(output)])


def test_user_words_to_feasible_sensible_layouts():
    from roomlayout.constraints import build_constraints
    from roomlayout.constraints.checks import check_layout

    text = "Sofa with its back to the north wall. Armchair next to the bookshelf. Cozy!"
    client = _vlm_answer(
        [
            {
                "type": "against_wall",
                "a": "sofa",
                "wall_id": "w_north",
                "max_gap": None,
                "hard": True,
                "restatement": "The sofa's back is against the north wall.",
                "source_text": "Sofa with its back to the north wall",
            },
            {
                "type": "near",
                "a": "armchair",
                "b": "bookshelf",
                "max_distance": 0.1,
                "hard": False,
                "restatement": "The armchair is right next to the bookshelf.",
                "source_text": "Armchair next to the bookshelf",
            },
        ],
        [
            {
                "text": "Cozy!",
                "kind": "out_of_vocabulary",
                "reason": "Coziness isn't something the layout rules can express yet.",
            }
        ],
    )

    built = build_constraints(_minimal_living_room(), text, client=client)

    assert built.parse.status == "ok", built.parse.message
    assert [u.text for u in built.parse.unexpressed] == ["Cozy!"]
    assert {c.origin for c in built.from_user} == {"vlm"}
    assert built.from_rules and {c.origin for c in built.from_rules} == {"rule_library"}
    wall_rule = next(c for c in built.from_user if c.type == "against_wall")

    result = solve(built.problem, k=3, seed=0, time_budget_s=10.0)

    top = result.layouts[0]
    assert top.feasible, result.notes
    by_id = {r.constraint_id: r for r in top.results}
    assert by_id[wall_rule.id].satisfied  # the user's hard request holds
    failed = [c for c in check_layout(built.problem, top) if not c.passed]
    assert not failed, failed  # and the room still makes sense


def test_user_intent_overrides_rule_library_defaults():
    from roomlayout.constraints import build_constraints

    client = _vlm_answer(
        [
            {
                "type": "faces",
                "a": "sofa",
                "b": "bookshelf",
                "max_angle_deg": None,
                "hard": False,
                "restatement": "The sofa faces the bookshelf.",
                "source_text": "sofa facing the bookshelf",
            }
        ]
    )
    built = build_constraints(_minimal_living_room(), "sofa facing the bookshelf", client=client)

    sofa_faces = [c for c in built.problem.constraints if c.type == "faces" and c.a == "sofa"]
    assert [c.origin for c in sofa_faces] == ["vlm"]  # no implied sofa->TV / sofa->table


def test_failed_parse_still_returns_a_rule_library_problem():
    from roomlayout.constraints import build_constraints
    from roomlayout.constraints.vlm.offline import offline_client, synthetic_response

    refused = offline_client([synthetic_response(None, stop_reason="refusal")])
    built = build_constraints(_minimal_living_room(), "anything", client=refused)

    assert built.parse.status == "refused"
    assert built.notes and not built.from_user
    assert built.from_rules
