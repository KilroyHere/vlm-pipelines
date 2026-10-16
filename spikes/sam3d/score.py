#!/usr/bin/env python
"""Score a SAM 3D spike run against EVAL.md and print the go/no-go verdict.

    python score.py --data data --out outputs                 # writes outputs/scores.csv template
    #  ... raters fill in outputs/scores.csv (shape/texture 1-5, front, failure tags) ...
    python score.py --data data --out outputs                 # -> outputs/results.md/.csv

Inputs:
- outputs/summary.json from run_sam3d.py
- data/<item>/meta.json "true_footprint": "rect" | "round" | {"points": [[x, y], ...]}
  (local frame: meters, origin at the bbox centre, +Y = front; see EVAL.md)
- outputs/scores.csv: manual ratings (template created on first run)

Runs on CPU in the roomlayout venv (needs shapely).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from shapely.affinity import rotate
from shapely.geometry import Polygon, box

# Go/no-go thresholds; keep in sync with EVAL.md section 5.
SHAPE_MIN_SCORE = 3
SHAPE_PASS_RATE_IN_ROOM = 0.70  # G1
IOU_MIN = 0.80
IOU_PASS_RATE = 0.80  # G2 (over items with a true footprint)
RELIABILITY_MIN = 0.90  # G3: mesh produced and normalized
RUNTIME_MEDIAN_MAX_S = 120.0  # G4
MISMATCH_MAX = 0.15
MISMATCH_PASS_RATE = 0.75  # G5
TEXTURE_MIN_SCORE = 3
TEXTURE_PASS_RATE = 0.60  # soft, reported only

SCORE_COLUMNS = [
    "item_id",
    "shape_score",
    "texture_score",
    "front_correct_letter",
    "failure_tags",
    "rater",
    "comments",
]
FAILURE_TAGS = {
    "missing_parts",
    "fused_parts",
    "wrong_topology",
    "hallucinated_back",
    "wrong_proportions",
    "not_upright",
    "floaters",
    "background_leak",
    "thin_parts_lost",
    "texture_wrong",
    "other",
}


def true_footprint(meta: dict) -> Polygon | None:
    spec = meta.get("true_footprint")
    dims = meta.get("dims")
    if spec is None:
        return None
    if spec in ("rect", "round"):
        if not dims:
            return None
        w, d = float(dims["width"]), float(dims["depth"])
        if spec == "rect":
            return box(-w / 2, -d / 2, w / 2, d / 2)
        t = [2 * math.pi * i / 128 for i in range(128)]
        return Polygon([(w / 2 * math.cos(a), d / 2 * math.sin(a)) for a in t])
    return Polygon(spec["points"])


def footprint_iou(asset: dict, truth: Polygon) -> tuple[float, float]:
    """(IoU with the asset's current front, best IoU over the 4 yaw hypotheses)."""
    fp = asset.get("footprint")
    if fp is None:
        w, d = asset["dimensions"]["width"], asset["dimensions"]["depth"]
        pred = box(-w / 2, -d / 2, w / 2, d / 2)
    else:
        pred = Polygon(fp["points"])
    ious = []
    for k in range(4):
        t = rotate(truth, 90 * k, origin=(0, 0))
        inter = pred.intersection(t).area
        union = pred.union(t).area
        ious.append(inter / union if union > 0 else 0.0)
    return ious[0], max(ious)


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="score a SAM 3D spike run")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scores", type=Path, help="default: OUT/scores.csv")
    a = p.parse_args(argv)
    summary = json.loads((a.out / "summary.json").read_text())
    items = summary["items"]
    scores_path = a.scores or a.out / "scores.csv"

    if not scores_path.exists():
        with open(scores_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
            w.writeheader()
            for it in items:
                w.writerow({"item_id": it["item_id"]})
        print(f"wrote template {scores_path}; fill it in (see EVAL.md) and re-run")
    with open(scores_path, newline="") as fh:
        manual = {r["item_id"]: r for r in csv.DictReader(fh)}

    rows = []
    for it in items:
        iid = it["item_id"]
        meta = json.loads((a.data / iid / "meta.json").read_text())
        m = manual.get(iid, {})
        cond = set(meta.get("condition", []))
        row = {
            "item_id": iid,
            "category": meta.get("category", ""),
            "condition": ",".join(sorted(cond)),
            "in_room": "product" not in cond,
            "status": it["status"],
            "generation_s": it.get("generation_s"),
            "vram_peak_gb": it.get("vram_peak_gb"),
            "aspect_mismatch": it.get("aspect_mismatch"),
            "front_auto": it.get("front_chosen", ""),
            "front_conf": it.get("front_confidence"),
            "front_true": (m.get("front_correct_letter") or "").strip().upper(),
            "shape": _num(m.get("shape_score")),
            "texture": _num(m.get("texture_score")),
            "failure_tags": m.get("failure_tags", ""),
            "iou_front": None,
            "iou_best": None,
        }
        unknown = {t.strip() for t in row["failure_tags"].split(";") if t.strip()} - FAILURE_TAGS
        if unknown:
            print(f"warning: {iid}: unknown failure tags {sorted(unknown)}", file=sys.stderr)
        truth = true_footprint(meta)
        if it["status"] == "ok" and truth is not None and it.get("asset_json"):
            asset = json.loads((a.out / it["asset_json"]).read_text())
            row["iou_front"], row["iou_best"] = (round(v, 4) for v in footprint_iou(asset, truth))
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    in_room_rated = [r for r in rows if r["in_room"] and r["shape"] is not None]
    with_iou = [r for r in ok if r["iou_best"] is not None]
    times = [r["generation_s"] for r in rows if r["generation_s"] is not None]
    mism = [r for r in ok if r["aspect_mismatch"] is not None]
    textured = [r for r in rows if r["texture"] is not None]
    fronts = [r for r in ok if r["front_true"]]

    g = {
        "G1 shape >= 3 on in-room items": (
            _rate([r["shape"] >= SHAPE_MIN_SCORE for r in in_room_rated]),
            SHAPE_PASS_RATE_IN_ROOM,
        ),
        "G2 footprint IoU >= 0.80": (
            _rate([r["iou_best"] >= IOU_MIN for r in with_iou]),
            IOU_PASS_RATE,
        ),
        "G3 mesh produced + normalized": (len(ok) / len(rows) if rows else None, RELIABILITY_MIN),
        "G4 median generation time (s)": (statistics.median(times) if times else None, None),
        "G5 aspect mismatch <= 15%": (
            _rate([r["aspect_mismatch"] <= MISMATCH_MAX for r in mism]),
            MISMATCH_PASS_RATE,
        ),
    }
    verdicts = {}
    for name, (val, thr) in g.items():
        if val is None:
            verdicts[name] = "NO DATA"
        elif name.startswith("G4"):
            verdicts[name] = "pass" if val <= RUNTIME_MEDIAN_MAX_S else "FAIL"
        else:
            verdicts[name] = "pass" if val >= thr else "FAIL"
    blocking = [k for k in verdicts if k[:2] in ("G1", "G2", "G3")]
    if any(verdicts[k] == "NO DATA" for k in blocking):
        decision = "INCOMPLETE (rate shapes and add true footprints)"
    elif any(verdicts[k] == "FAIL" for k in blocking):
        decision = "NO-GO"
        # EVAL.md: "GO for product photos only" when the product subset passes G1/G2/G3.
        prod = [r for r in rows if not r["in_room"]]
        prod_shape = _rate([r["shape"] >= SHAPE_MIN_SCORE for r in prod if r["shape"] is not None])
        prod_iou = _rate([r["iou_best"] >= IOU_MIN for r in prod if r["iou_best"] is not None])
        prod_ok = _rate([r["status"] == "ok" for r in prod])
        if (
            verdicts["G3 mesh produced + normalized"] == "pass"
            and prod_shape is not None
            and prod_shape >= SHAPE_PASS_RATE_IN_ROOM
            and (prod_iou is None or prod_iou >= IOU_PASS_RATE)
            and prod_ok is not None
            and prod_ok >= RELIABILITY_MIN
        ):
            decision = "GO for product photos only (in-room fails)"
    elif all(v == "pass" for v in verdicts.values()):
        decision = "GO"
    else:
        decision = "CONDITIONAL GO (G4/G5 need mitigation)"

    lines = ["# SAM 3D spike results", "", f"**Decision: {decision}**", ""]
    lines += ["| criterion | value | threshold | verdict |", "|---|---|---|---|"]
    for name, (val, thr) in g.items():
        if name.startswith("G4"):
            vs, ts = (f"{val:.1f}" if val is not None else "-"), f"<= {RUNTIME_MEDIAN_MAX_S:.0f}"
        else:
            vs, ts = (f"{val:.0%}" if val is not None else "-"), f">= {thr:.0%}"
        lines.append(f"| {name} | {vs} | {ts} | {verdicts[name]} |")
    tex = _rate([r["texture"] >= TEXTURE_MIN_SCORE for r in textured])
    front_acc = _rate([r["front_auto"] == r["front_true"] for r in fronts])
    lines += [
        "",
        f"Texture >= 3 (soft, target {TEXTURE_PASS_RATE:.0%}): "
        + (f"{tex:.0%}" if tex is not None else "-"),
        "Front heuristic accuracy (ours, not SAM 3D's): "
        + (f"{front_acc:.0%} of {len(fronts)}" if front_acc is not None else "-"),
        "",
        "| item | category | condition | status | gen s | VRAM GB | shape | texture | "
        "IoU (front / best) | mismatch | front auto/true (conf) | failures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v, fmt):
        return format(v, fmt) if isinstance(v, int | float) else "-"

    for r in rows:
        lines.append(
            f"| {r['item_id']} | {r['category']} | {r['condition']} | {r['status']} | "
            f"{f(r['generation_s'], '.1f')} | {f(r['vram_peak_gb'], '.1f')} | "
            f"{f(r['shape'], '.0f')} | {f(r['texture'], '.0f')} | "
            f"{f(r['iou_front'], '.2f')} / {f(r['iou_best'], '.2f')} | "
            f"{f(r['aspect_mismatch'], '.0%')} | {r['front_auto']}/{r['front_true'] or '?'} "
            f"({f(r['front_conf'], '.2f')}) | {r['failure_tags']} |"
        )
    (a.out / "results.md").write_text("\n".join(lines) + "\n")
    with open(a.out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines[:12]))
    print(f"\nfull table: {a.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
