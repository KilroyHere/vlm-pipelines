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
            "dims_provenance": "seller_reported",
            "source_url": "https://example.com/listing/1",
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
    assert (sofa_sum["dims_provenance"], sofa_sum["dims_from"]) == ("", "manual")
    chair_sum = json.loads((out / "chair_01" / "summary.json").read_text())
    assert chair_sum["source_url"] == "https://example.com/listing/1"
    assert chair_sum["dims_provenance"] == "seller_reported"  # finer value kept
    assert chair_sum["dims_from"] == "listing"  # contract value passed to the CLI
    chair_asset = json.loads((out / chair_sum["asset_json"]).read_text())
    assert chair_asset["source"]["dimensions_from"] == "listing"
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


# ------------------------------------------------------------ provenance and scoring


def _scored_item(data, out, iid, meta, status="ok", gen_s=30.0, mismatch=0.05):
    d = data / iid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    entry = {"item_id": iid, "status": status, "generation_s": gen_s, "vram_peak_gb": 20.0}
    if status == "ok":
        w, dp = meta["dims"]["width"], meta["dims"]["depth"]
        pts = [[-w / 2, -dp / 2], [w / 2, -dp / 2], [w / 2, dp / 2], [-w / 2, dp / 2]]
        asset = {"dimensions": meta["dims"], "footprint": {"points": pts}}
        (out / iid).mkdir(parents=True)
        (out / iid / "asset.json").write_text(json.dumps(asset))
        entry.update(
            aspect_mismatch=mismatch,
            front_chosen="A",
            front_confidence=0.9,
            asset_json=f"{iid}/asset.json",
        )
    return entry


def _write_scores(out, ratings):
    cols = ["item_id", "shape_score", "texture_score", "front_correct_letter", "failure_tags"]
    with open(out / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for iid, (shape, tex, tags) in ratings.items():
            w.writerow(dict(zip(cols, [iid, shape, tex, "A", tags], strict=True)))


def test_score_one_decision_over_all_items_with_g2_coverage(tmp_path):
    data, out = tmp_path / "data", tmp_path / "outputs"
    out.mkdir()
    dims = {"width": 2.0, "depth": 0.9, "height": 0.8}
    own = {
        "category": "sofa",
        "dims": dims,
        "condition": ["in_room"],
        "dims_provenance": "tape_measured",
        "true_footprint": "rect",
    }
    web = {
        "category": "sofa",
        "dims": dims,
        "condition": ["in_room"],
        "dims_provenance": "seller_reported",
        "source_url": "https://example.com/x",
    }
    items = [_scored_item(data, out, f"own{i}", own) for i in range(2)]
    # Web items count like any other: no true footprint, and one failed generation.
    items += [_scored_item(data, out, f"web{i}", web) for i in range(7)]
    items.append(_scored_item(data, out, "web_fail", web, status="generation_failed", gen_s=None))
    (out / "summary.json").write_text(json.dumps({"environment": {}, "items": items}))
    ratings = {f"own{i}": ("4", "4", "") for i in range(2)}
    ratings |= {f"web{i}": ("4", "3", "fused_parts" if i == 0 else "") for i in range(7)}
    ratings["web_fail"] = ("", "", "other")
    _write_scores(out, ratings)

    res = _run("score.py", "--data", data, "--out", out)
    assert res.returncode == 0, res.stderr
    report = json.loads((out / "results.json").read_text())
    assert report["items"] == 10
    # G3 counts the failed web item: 9/10 = 90 % -> pass; G2 rests on 2 footprints only.
    assert report["criteria"]["G3 mesh produced + normalized"]["value"] == 0.9
    assert report["g2_items"] == 2
    assert report["criteria"]["G2 footprint IoU >= 0.80"]["value"] == 1.0
    assert report["decision"] == "GO"
    assert report["dims_provenance"] == {"seller_reported": 8, "tape_measured": 2}
    assert report["stats"]["failure_tags"] == {"fused_parts": 1, "other": 1}
    md = (out / "results.md").read_text()
    assert "G2 covered 2 of 10 items" in md
    assert "weak evidence" in md
    assert "| dims_provenance |" in md
    assert "| web0 | sofa | in_room | seller_reported | ok |" in md
    assert "dev" not in md.lower()  # one report, no separate sets


def test_prepare_writes_provenance_and_clean_url(tmp_path):
    photo = tmp_path / "p.jpg"
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(photo)
    url = "https://offerup.com/item/detail/abc?utm_source=share&utm_medium=x&ref=7"
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "sectional_03",
        "--category",
        "sectional",
        "--dims",
        "2.9",
        "1.7",
        "0.85",
        "--dims-provenance",
        "seller_reported",
        "--source-url",
        url,
    )
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "sectional_03" / "meta.json").read_text())
    assert "set" not in meta
    assert meta["dims_provenance"] == "seller_reported"
    assert meta["source_url"] == "https://offerup.com/item/detail/abc?ref=7"

    # --set no longer exists.
    r = _run(
        "make_mask.py", "prepare", photo, tmp_path / "x", "--category", "sofa", "--set", "dev_web"
    )
    assert r.returncode != 0

    # Dims without provenance are refused before anything is written.
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "sofa_02",
        "--category",
        "sofa",
        "--dims",
        "2",
        "1",
        "0.8",
    )
    assert r.returncode != 0 and "--dims-provenance" in r.stderr
    assert not (tmp_path / "sofa_02").exists()

    # Item without known dims (e.g. case #10): fine.
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "coffee_table_10",
        "--category",
        "coffee_table",
        "--source-url",
        "https://example.com/a",
    )
    assert r.returncode == 0, r.stderr
    assert "dims" not in json.loads((tmp_path / "coffee_table_10" / "meta.json").read_text())


def test_dry_run_checks_provenance(tmp_path):
    data = tmp_path / "data"
    _item(data, "x1", {"category": "sofa", "dims_provenance": "guess"})
    _item(data, "x2", {"category": "sofa", "dims_provenance": "manufacturer"})
    r = _run("run_sam3d.py", "--data", data, "--out", tmp_path / "o", "--dry-run")
    assert r.returncode == 1
    assert "'dims_provenance' must be one of" in r.stdout
    assert "manufacturer dims but no source_url" in r.stdout
    assert "1 valid, 1 invalid" in r.stdout
