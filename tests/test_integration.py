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
