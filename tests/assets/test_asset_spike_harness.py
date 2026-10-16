"""The CPU half of spikes/sam3d: input validation, normalization of (fake) SAM 3D outputs
via the CLI subprocess, per-item summaries and scoring. Generation itself needs a GPU."""

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import synthetic as S

SPIKE = Path(__file__).resolve().parents[2] / "spikes" / "sam3d"


def _item(data: Path, iid: str, meta: dict, mask: bool = True, size=(64, 48)):
    d = data / iid
    d.mkdir(parents=True)
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8)).save(d / "image.jpg")
    if mask:
        m = np.zeros((size[1], size[0]), np.uint8)
        m[10:40, 10:50] = 255
        Image.fromarray(m).save(d / "mask.png")
    (d / "meta.json").write_text(json.dumps(meta))


def _run(script: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SPIKE / script), *map(str, args)], capture_output=True, text=True
    )


def test_harness_cpu_path_and_scoring(tmp_path):
    data, out = tmp_path / "data", tmp_path / "outputs"
    sofa = S.l_sectional()
    _item(
        data,
        "sofa_01",
        {
            "category": "sectional",
            "dims": {"width": 2.8, "depth": 1.6, "height": 0.85},
            "condition": ["in_room"],
            "true_footprint": {"points": [list(p) for p in sofa.footprint.exterior.coords[:-1]]},
        },
    )
    _item(
        data,
        "chair_01",
        {
            "category": "dining_chair",
            "dims": {"width": 0.48, "depth": 0.52, "height": 0.9},
            "condition": ["product"],
            "true_footprint": "rect",
        },
    )
    _item(data, "broken_01", {"category": "sofa"}, mask=False)
    _item(data, "badmask_01", {"category": "sofa"})
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(data / "badmask_01" / "mask.png")

    # Fake SAM 3D outputs: synthetic meshes in SAM 3D's Y-up convention.
    for iid, synth in (("sofa_01", sofa), ("chair_01", S.chair())):
        (out / iid).mkdir(parents=True)
        S.as_generator_output(synth, 5)[0].export(out / iid / "sam3d.glb")

    dry = _run("run_sam3d.py", "--data", data, "--out", out, "--dry-run")
    assert dry.returncode == 1  # two invalid items
    assert "missing mask.png" in dry.stdout
    assert "mask size" in dry.stdout

    run = _run(
        "run_sam3d.py",
        "--data",
        data,
        "--out",
        out,
        "--skip-generation",
        "--roomlayout-python",
        sys.executable,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads((out / "summary.json").read_text())
    status = {i["item_id"]: i["status"] for i in summary["items"]}
    assert status == {
        "sofa_01": "ok",
        "chair_01": "ok",
        "broken_01": "invalid",
        "badmask_01": "invalid",
    }
    sofa_sum = json.loads((out / "sofa_01" / "summary.json").read_text())
    assert sofa_sum["footprint_vertices"] == 6
    assert (out / sofa_sum["asset_json"]).exists()
    assert (out / sofa_sum["contact_sheet"]).exists()

    first = _run("score.py", "--data", data, "--out", out)
    assert first.returncode == 0, first.stderr
    assert "INCOMPLETE" in first.stdout  # no ratings yet
    with open(out / "scores.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.update(shape_score="4", texture_score="3", front_correct_letter="A")
    with open(out / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    second = _run("score.py", "--data", data, "--out", out)
    assert second.returncode == 0, second.stderr
    results = (out / "results.md").read_text()
    # 2 of 4 items are invalid -> reliability 50 % -> G3 fails -> NO-GO
    assert "NO-GO" in results
    with open(out / "results.csv", newline="") as fh:
        res = {r["item_id"]: r for r in csv.DictReader(fh)}
    assert float(res["sofa_01"]["iou_best"]) > 0.97
    assert float(res["chair_01"]["iou_best"]) > 0.99


def test_harness_import_boundary():
    """ADR-0004: SAM 3D is imported only inside the generation code path, and the harness
    never imports roomlayout (it calls the CLI in a subprocess)."""
    tree = ast.parse((SPIKE / "run_sam3d.py").read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert {"inference", "sam3d_objects", "torch", "roomlayout"}.isdisjoint(top_level)
    every = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not any(m.startswith("roomlayout") for m in every)
