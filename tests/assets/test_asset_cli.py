"""CLI end to end: raw GLB -> GLB + Asset JSON + report + renders, and the Asset drops
straight into a LayoutProblem."""

import json
import subprocess
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from roomlayout.assets import load_mesh
from roomlayout.assets.__main__ import main
from roomlayout.schema import Asset, LayoutProblem, Pose2D

from . import synthetic as S


@pytest.fixture(scope="module")
def sofa_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    synth = S.l_sectional()
    raw, _ = S.as_generator_output(synth, 9)
    raw_path = tmp / "raw_sofa.glb"
    raw.export(raw_path)
    out = tmp / "out"
    code = main(
        [
            str(raw_path),
            "--category",
            "sectional",
            "--dims",
            "2.8",
            "1.6",
            "0.85",
            "--id",
            "sofa1",
            "--dims-from",
            "listing",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return synth, out


def test_outputs_written(sofa_run):
    _, out = sofa_run
    names = {p.name for p in out.iterdir()}
    expected = {"sofa1.glb", "sofa1.asset.json", "sofa1.report.json", "sofa1.front_candidates.png"}
    assert expected <= names
    assert {f"sofa1.front_{c}.png" for c in "ABCD"} <= names
    with Image.open(out / "sofa1.front_candidates.png") as im:
        assert im.size[0] > 1000


def test_asset_json_validates_with_relative_mesh_uri(sofa_run):
    _, out = sofa_run
    asset = Asset.model_validate_json((out / "sofa1.asset.json").read_text())
    assert asset.category == "sofa"
    assert asset.mesh.uri == "sofa1.glb" and asset.mesh.format == "glb"
    assert (out / asset.mesh.uri).exists()
    assert asset.source.dimensions_from == "listing"
    assert asset.source.geometry_from == "sam3d"
    assert asset.front_confirmed is False
    assert [z.name for z in asset.clearance_zones] == ["legroom", "legroom_2"]


def test_glb_matches_asset_dimensions_in_our_frame(sofa_run):
    synth, out = sofa_run
    mesh = load_mesh(out / "sofa1.glb", up="y")
    assert mesh.extents == pytest.approx([2.8, 1.6, 0.85], abs=1e-3)
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    # Same geometry as the ground truth (front recovered), up to float32.
    from scipy.spatial import cKDTree

    assert cKDTree(synth.mesh.vertices).query(mesh.vertices)[0].max() < 1e-5


def test_report_contents(sofa_run):
    _, out = sofa_run
    rep = json.loads((out / "sofa1.report.json").read_text())
    assert rep["fit"]["scale"] and len(rep["fit"]["scale"]) == 3
    assert rep["fit"]["flagged"] is False
    assert rep["debris"]["removed_components"] == 2
    assert rep["footprint"]["vertex_count"] == 6
    front = rep["front"]
    assert front["chosen"] in "ABCD" and 0 <= front["confidence"] <= 1
    assert [h["label"] for h in front["hypotheses"]] == list("ABCD")
    assert all(h["render"] == f"sofa1.front_{h['label']}.png" for h in front["hypotheses"])
    assert front["contact_sheet"] == "sofa1.front_candidates.png"


def test_asset_drops_into_a_layout_problem(sofa_run):
    _, out = sofa_run
    asset = json.loads((out / "sofa1.asset.json").read_text())
    problem = LayoutProblem.model_validate(
        {
            "id": "one_sofa",
            "room": {
                "id": "r",
                "floor": {"points": [[0, 0], [5, 0], [5, 4], [0, 4]]},
                "ceiling_height": 2.6,
                "walls": [
                    {"id": "s", "start": [0, 0], "end": [5, 0], "height": 2.6},
                    {"id": "e", "start": [5, 0], "end": [5, 4], "height": 2.6},
                    {"id": "n", "start": [5, 4], "end": [0, 4], "height": 2.6},
                    {"id": "w", "start": [0, 4], "end": [0, 0], "height": 2.6},
                ],
            },
            "assets": [asset],
            "constraints": [{"type": "against_wall", "id": "c1", "a": "sofa1", "hard": True}],
            "initial": {"sofa1": {"x": 2.5, "y": 0.8, "yaw": 0.0}},
        }
    )
    a = problem.assets[0]
    # The footprint placed at the initial pose stays inside the 5 x 4 room.
    pose = Pose2D(x=2.5, y=0.8)
    pts = np.array([pose.transform(p) for p in a.effective_footprint().points])
    assert pts.min() >= 0 and pts[:, 0].max() <= 5 and pts[:, 1].max() <= 4


def test_cli_without_dims_warns_loudly(tmp_path, capsys):
    raw, _ = S.as_generator_output(S.chair(), 1)
    raw.export(tmp_path / "c.glb")
    code = main(
        [
            str(tmp_path / "c.glb"),
            "--category",
            "chair",
            "--out",
            str(tmp_path / "o"),
            "--no-render",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "DIMENSIONS NOT MEASURED" in err
    asset = Asset.model_validate_json((tmp_path / "o" / "c.asset.json").read_text())
    assert asset.source.dimensions_from == "mesh"


def test_cli_front_override_and_bad_input(tmp_path, capsys):
    raw, _ = S.as_generator_output(S.chair(), 1)
    raw.export(tmp_path / "c.glb")
    code = main(
        [
            str(tmp_path / "c.glb"),
            "--category",
            "chair",
            "--dims",
            "0.48",
            "0.52",
            "0.9",
            "--front",
            "B",
            "--out",
            str(tmp_path / "o"),
            "--no-render",
        ]
    )
    assert code == 0
    asset = Asset.model_validate_json((tmp_path / "o" / "c.asset.json").read_text())
    assert asset.front_confirmed is True
    assert main([str(tmp_path / "missing.glb"), "--category", "chair", "--out", str(tmp_path)]) == 2
    assert "error" in capsys.readouterr().err


def test_module_entry_point_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "roomlayout.assets", "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "--dims" in proc.stdout


def test_obj_input_with_z_up(tmp_path):
    synth = S.bookshelf()
    mesh = synth.mesh.copy()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.7, (0, 0, 1)))
    mesh.export(tmp_path / "shelf.obj")
    code = main(
        [
            str(tmp_path / "shelf.obj"),
            "--category",
            "bookshelf",
            "--up",
            "z",
            "--dims",
            "0.9",
            "0.35",
            "1.8",
            "--out",
            str(tmp_path / "o"),
            "--no-render",
        ]
    )
    assert code == 0
    asset = Asset.model_validate_json((tmp_path / "o" / "shelf.asset.json").read_text())
    assert asset.category == "storage"
    assert asset.dimensions.width == pytest.approx(0.9)
