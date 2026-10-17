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
- data/<item>/meta.json "dims_provenance" / "source_url" (informational; shown per item)

One report and one decision over all items. G2 is computed over the items that have a
true footprint, and the report says how many that was. Outputs: results.md, results.csv,
results.json.

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


def build_rows(items: list[dict], data: Path, out: Path, manual: dict) -> list[dict]:
    rows = []
    for it in items:
        iid = it["item_id"]
        meta = json.loads((data / iid / "meta.json").read_text())
        m = manual.get(iid, {})
        cond = set(meta.get("condition", []))
        row = {
            "item_id": iid,
            "category": meta.get("category", ""),
            "condition": ",".join(sorted(cond)),
            "in_room": "product" not in cond,
            "dims_provenance": meta.get("dims_provenance", ""),
            "source_url": meta.get("source_url", ""),
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
        unknown = set(_tags(row)) - FAILURE_TAGS
        if unknown:
            print(f"warning: {iid}: unknown failure tags {sorted(unknown)}", file=sys.stderr)
        truth = true_footprint(meta)  # web-sourced items often have none: IoU stays empty
        if it["status"] == "ok" and truth is not None and it.get("asset_json"):
            asset = json.loads((out / it["asset_json"]).read_text())
            row["iou_front"], row["iou_best"] = (round(v, 4) for v in footprint_iou(asset, truth))
        rows.append(row)
    return rows


def _tags(row: dict) -> list[str]:
    return [t.strip() for t in row["failure_tags"].split(";") if t.strip()]


def decide(rows: list[dict]) -> tuple[dict, dict, str]:
    """G1-G5 and the go/no-go decision over all items (EVAL.md section 5)."""
    ok = [r for r in rows if r["status"] == "ok"]
    in_room_rated = [r for r in rows if r["in_room"] and r["shape"] is not None]
    with_iou = [r for r in ok if r["iou_best"] is not None]
    times = [r["generation_s"] for r in rows if r["generation_s"] is not None]
    mism = [r for r in ok if r["aspect_mismatch"] is not None]
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
    if not rows:
        decision = "INCOMPLETE (no items)"
    elif any(verdicts[k] == "NO DATA" for k in blocking):
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
    return g, verdicts, decision


def describe(rows: list[dict]) -> dict:
    """Descriptive statistics over all items (complements the G1-G5 table)."""
    ok = [r for r in rows if r["status"] == "ok"]
    shapes = [r["shape"] for r in rows if r["shape"] is not None]
    textures = [r["texture"] for r in rows if r["texture"] is not None]
    mism = [r["aspect_mismatch"] for r in ok if r["aspect_mismatch"] is not None]
    times = [r["generation_s"] for r in rows if r["generation_s"] is not None]
    vram = [r["vram_peak_gb"] for r in rows if r["vram_peak_gb"] is not None]
    ious = [r["iou_best"] for r in ok if r["iou_best"] is not None]
    fronts = [r for r in ok if r["front_true"]]
    statuses: dict[str, int] = {}
    tags: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        for t in _tags(r):
            tags[t] = tags.get(t, 0) + 1
    return {
        "n": len(rows),
        "statuses": statuses,
        "shape_rated": len(shapes),
        "shape_ge3": _rate([v >= SHAPE_MIN_SCORE for v in shapes]),
        "shape_mean": statistics.mean(shapes) if shapes else None,
        "texture_rated": len(textures),
        "texture_ge3": _rate([v >= TEXTURE_MIN_SCORE for v in textures]),
        "failure_tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
        "mismatch_n": len(mism),
        "mismatch_median": statistics.median(mism) if mism else None,
        "mismatch_le15": _rate([v <= MISMATCH_MAX for v in mism]),
        "gen_n": len(times),
        "gen_median_s": statistics.median(times) if times else None,
        "vram_max_gb": max(vram) if vram else None,
        "iou_n": len(ious),
        "iou_ge80": _rate([v >= IOU_MIN for v in ious]),
        "front_n": len(fronts),
        "front_acc": _rate([r["front_auto"] == r["front_true"] for r in fronts]),
    }


def _f(v, fmt):
    return format(v, fmt) if isinstance(v, int | float) else "-"


def _stats_lines(st: dict) -> list[str]:
    tags = ", ".join(f"{k} {v}" for k, v in st["failure_tags"].items()) or "-"
    status = ", ".join(f"{k} {v}" for k, v in sorted(st["statuses"].items())) or "-"
    return [
        f"- items: {st['n']} ({status})",
        f"- shape >= 3: {_f(st['shape_ge3'], '.0%')} of {st['shape_rated']} rated "
        f"(mean {_f(st['shape_mean'], '.1f')})",
        f"- texture >= 3: {_f(st['texture_ge3'], '.0%')} of {st['texture_rated']} rated",
        f"- failure tags: {tags}",
        f"- aspect mismatch: median {_f(st['mismatch_median'], '.0%')}, <= 15 % for "
        f"{_f(st['mismatch_le15'], '.0%')} of {st['mismatch_n']}",
        f"- generation: median {_f(st['gen_median_s'], '.1f')} s over {st['gen_n']} items, "
        f"peak VRAM {_f(st['vram_max_gb'], '.1f')} GB",
        f"- footprint IoU >= 0.80: {_f(st['iou_ge80'], '.0%')} of {st['iou_n']} items "
        "with a true footprint",
        f"- front heuristic accuracy (ours): {_f(st['front_acc'], '.0%')} of {st['front_n']}",
    ]


def _item_table(rows: list[dict]) -> list[str]:
    lines = [
        "| item | category | condition | dims_provenance | status | gen s | VRAM GB | shape | texture | "
        "IoU (front / best) | mismatch | front auto/true (conf) | failures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['item_id']} | {r['category']} | {r['condition']} | "
            f"{r['dims_provenance'] or '-'} | {r['status']} | "
            f"{_f(r['generation_s'], '.1f')} | {_f(r['vram_peak_gb'], '.1f')} | "
            f"{_f(r['shape'], '.0f')} | {_f(r['texture'], '.0f')} | "
            f"{_f(r['iou_front'], '.2f')} / {_f(r['iou_best'], '.2f')} | "
            f"{_f(r['aspect_mismatch'], '.0%')} | {r['front_auto']}/{r['front_true'] or '?'} "
            f"({_f(r['front_conf'], '.2f')}) | {r['failure_tags']} |"
        )
    return lines


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

    rows = build_rows(items, a.data, a.out, manual)
    g, verdicts, decision = decide(rows)
    stats = describe(rows)
    g2_items = sum(1 for r in rows if r["iou_best"] is not None)
    provenance: dict[str, int] = {}
    for r in rows:
        key = r["dims_provenance"] or "unspecified"
        provenance[key] = provenance.get(key, 0) + 1

    lines = [
        "# SAM 3D spike results",
        "",
        f"**Decision: {decision}**",
        "",
        f"Computed over all {len(rows)} items. **G2 covered {g2_items} of {len(rows)} items** "
        "(those with a true footprint and a normalized mesh)"
        + (": few footprints, so G2 is weak evidence." if 0 < g2_items < 10 else "."),
        "Dims provenance: "
        + ", ".join(f"{k} {v}" for k, v in sorted(provenance.items(), key=lambda kv: -kv[1])),
        "",
        "| criterion | value | threshold | verdict |",
        "|---|---|---|---|",
    ]
    for name, (val, thr) in g.items():
        if name.startswith("G4"):
            vs, ts = _f(val, ".1f"), f"<= {RUNTIME_MEDIAN_MAX_S:.0f}"
        else:
            vs, ts = _f(val, ".0%"), f">= {thr:.0%}"
        label = f"{name} ({g2_items} items)" if name.startswith("G2") else name
        lines.append(f"| {label} | {vs} | {ts} | {verdicts[name]} |")
    lines += ["", f"Texture >= 3 is a soft target of {TEXTURE_PASS_RATE:.0%}.", ""]
    lines += _stats_lines(stats) + [""] + _item_table(rows)

    (a.out / "results.md").write_text("\n".join(lines) + "\n")
    with open(a.out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "items": len(rows),
                "g2_items": g2_items,
                "dims_provenance": provenance,
                "criteria": {k: {"value": v, "verdict": verdicts[k]} for k, (v, _) in g.items()},
                "stats": stats,
            },
            indent=2,
        )
    )
    print("\n".join(lines[:20]))
    print(f"\nfull report: {a.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
