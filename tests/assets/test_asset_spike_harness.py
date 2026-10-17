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
            "set": "dev_web",
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
    assert (sofa_sum["set"], sofa_sum["dims_provenance"], sofa_sum["dims_from"]) == (
        "go_no_go",
        "",
        "manual",
    )
    chair_sum = json.loads((out / "chair_01" / "summary.json").read_text())
    assert chair_sum["set"] == "dev_web"
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
    # go_no_go = sofa + 2 invalid items -> reliability 33 % -> G3 fails -> NO-GO
    # (the dev_web chair is not in the denominator)
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


# --------------------------------------------------------------------------- dev_web set


def _scored_item(data, out, iid, meta, status="ok", gen_s=30.0, mismatch=0.05, fp=None):
    d = data / iid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    entry = {"item_id": iid, "status": status, "generation_s": gen_s, "vram_peak_gb": 20.0}
    if status == "ok":
        w, dp = meta["dims"]["width"], meta["dims"]["depth"]
        pts = fp or [[-w / 2, -dp / 2], [w / 2, -dp / 2], [w / 2, dp / 2], [-w / 2, dp / 2]]
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
    with open(out / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "item_id",
                "shape_score",
                "texture_score",
                "front_correct_letter",
                "failure_tags",
            ],
        )
        w.writeheader()
        for iid, (shape, tex, tags) in ratings.items():
            w.writerow(
                {
                    "item_id": iid,
                    "shape_score": shape,
                    "texture_score": tex,
                    "front_correct_letter": "A",
                    "failure_tags": tags,
                }
            )


def test_score_keeps_dev_web_out_of_go_no_go(tmp_path):
    data, out = tmp_path / "data", tmp_path / "outputs"
    out.mkdir()
    dims = {"width": 2.0, "depth": 0.9, "height": 0.8}
    gng = {
        "set": "go_no_go",
        "category": "sofa",
        "dims": dims,
        "condition": ["in_room"],
        "dims_provenance": "tape_measured",
        "true_footprint": "rect",
    }
    dev = {
        "set": "dev_web",
        "category": "sofa",
        "dims": dims,
        "condition": ["in_room"],
        "dims_provenance": "seller_reported",
        "source_url": "https://example.com/x",
    }
    items = [_scored_item(data, out, f"g{i}", gng) for i in range(3)]
    # Dev items are terrible: one failed generation, huge mismatch, shape 1, no footprints.
    items.append(_scored_item(data, out, "d0", dev, status="generation_failed", gen_s=None))
    items += [_scored_item(data, out, f"d{i}", dev, gen_s=900.0, mismatch=0.6) for i in (1, 2)]
    # A legacy item without "set" counts as go_no_go.
    legacy = {k: v for k, v in gng.items() if k not in ("set", "dims_provenance")}
    items.append(_scored_item(data, out, "legacy", legacy))
    (out / "summary.json").write_text(json.dumps({"environment": {}, "items": items}))
    ratings = {f"g{i}": ("4", "4", "") for i in range(3)} | {"legacy": ("4", "3", "")}
    ratings |= {
        "d0": ("", "", "other"),
        "d1": ("1", "1", "fused_parts;floaters"),
        "d2": ("2", "2", "fused_parts"),
    }
    _write_scores(out, ratings)

    res = _run("score.py", "--data", data, "--out", out)
    assert res.returncode == 0, res.stderr
    report = json.loads((out / "results.json").read_text())
    # Dev failures (G3 33 %, 900 s runtime, 60 % mismatch, shape 1) don't touch the verdict.
    assert report["decision"] == "GO"
    assert report["decision_scope"] == "go_no_go items only"
    crit = report["go_no_go"]["criteria"]
    assert crit["G3 mesh produced + normalized"]["value"] == 1.0
    assert crit["G4 median generation time (s)"]["value"] == 30.0
    assert report["go_no_go"]["stats"]["n"] == 4
    st = report["dev_web"]["stats"]
    assert st["n"] == 3
    assert st["statuses"] == {"generation_failed": 1, "ok": 2}
    assert st["shape_ge3"] == 0.0 and st["shape_rated"] == 2
    assert st["failure_tags"] == {"fused_parts": 2, "floaters": 1, "other": 1}
    assert st["mismatch_median"] == 0.6
    assert st["gen_median_s"] == 900.0
    assert st["iou_n"] == 0  # no true footprints in the dev set: fine
    md = (out / "results.md").read_text()
    assert "computed over the **4 go_no_go items only**" in md
    assert "## Dev set (`dev_web`): NOT used for go/no-go" in md
    with open(out / "results.csv", newline="") as fh:
        sets = {r["item_id"]: r["set"] for r in csv.DictReader(fh)}
    assert sets["legacy"] == "go_no_go" and sets["d1"] == "dev_web"


def test_score_with_only_dev_items_is_incomplete(tmp_path):
    data, out = tmp_path / "data", tmp_path / "outputs"
    out.mkdir()
    dev = {
        "set": "dev_web",
        "category": "chair",
        "dims_provenance": "estimated",
        "dims": {"width": 0.5, "depth": 0.5, "height": 0.9},
    }
    items = [_scored_item(data, out, "d0", dev)]
    (out / "summary.json").write_text(json.dumps({"environment": {}, "items": items}))
    _write_scores(out, {"d0": ("5", "5", "")})
    res = _run("score.py", "--data", data, "--out", out)
    assert res.returncode == 0, res.stderr
    assert json.loads((out / "results.json").read_text())["decision"].startswith("INCOMPLETE")


def test_prepare_writes_set_provenance_and_clean_url(tmp_path):
    photo = tmp_path / "p.jpg"
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(photo)
    url = "https://offerup.com/item/detail/abc?utm_source=share&utm_medium=x&ref=7"
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "dev03",
        "--category",
        "sectional",
        "--dims",
        "2.9",
        "1.7",
        "0.85",
        "--dims-provenance",
        "seller_reported",
        "--set",
        "dev_web",
        "--source-url",
        url,
    )
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "dev03" / "meta.json").read_text())
    assert meta["set"] == "dev_web"
    assert meta["dims_provenance"] == "seller_reported"
    assert meta["source_url"] == "https://offerup.com/item/detail/abc?ref=7"

    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "own01",
        "--category",
        "armchair",
        "--dims",
        "0.8",
        "0.85",
        "0.9",
    )
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "own01" / "meta.json").read_text())
    assert (meta["set"], meta["dims_provenance"]) == ("go_no_go", "tape_measured")

    # A dev item with dims but no provenance is refused before anything is written.
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "dev04",
        "--category",
        "sofa",
        "--dims",
        "2",
        "1",
        "0.8",
        "--set",
        "dev_web",
    )
    assert r.returncode != 0
    assert not (tmp_path / "dev04").exists()

    # Dev item without dims (e.g. case #10): fine.
    r = _run(
        "make_mask.py",
        "prepare",
        photo,
        tmp_path / "dev10",
        "--category",
        "coffee_table",
        "--set",
        "dev_web",
        "--source-url",
        "https://example.com/a",
    )
    assert r.returncode == 0, r.stderr
    assert "dims" not in json.loads((tmp_path / "dev10" / "meta.json").read_text())


def test_dry_run_rejects_bad_set_and_provenance(tmp_path):
    data = tmp_path / "data"
    _item(data, "x1", {"category": "sofa", "set": "dev"})
    _item(data, "x2", {"category": "sofa", "dims_provenance": "guess"})
    _item(data, "x3", {"category": "sofa", "set": "dev_web", "dims_provenance": "estimated"})
    r = _run("run_sam3d.py", "--data", data, "--out", tmp_path / "o", "--dry-run")
    assert r.returncode == 1
    assert "'set' must be one of" in r.stdout
    assert "'dims_provenance' must be one of" in r.stdout
    assert "dev_web item without source_url" in r.stdout
    assert "1 valid, 2 invalid" in r.stdout
