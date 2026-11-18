"""Baseline: every public RoomPlan scan, emptied, furnished with listing-size boxes.

    python spikes/fit_baseline/run.py [--seconds 10] [--out out/fit_baseline]

Needs the scans: python datasets/roomplan_public/fetch.py. Rule library only (no request
text, no API). Writes one JSON row per (room, scenario) and a summary table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roomlayout.assets import box_assets, load_products
from roomlayout.ingest.roomplan import IngestError
from roomlayout.pipeline import empty_room, furnish, plan, problem_from_roomplan

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ROOT / "examples" / "products" / "ikea_2026-11.json"
MANIFEST = ROOT / "datasets" / "roomplan_public" / "manifest.json"
SCENARIOS = {
    "living": ["soderhamn_sectional", "billy_bookcase"],
    "dining": ["lisabo_table", "lisabo_chair"],
    "bedroom": ["malm_bed_queen", "micke_desk", "billy_bookcase"],
}


def floor_area(problem) -> float:
    pts = problem.room.floor.points
    return (
        abs(
            sum(
                x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True)
            )
        )
        / 2
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, default=ROOT / "data" / "roomplan_public")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "fit_baseline")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rooms = json.loads(MANIFEST.read_text())["rooms"]
    rows = []
    for r in rooms:
        try:
            scanned, _ = problem_from_roomplan(args.scans / f"{r['id']}.json")
        except IngestError as e:
            rows.append({"room": r["id"], "error": str(e)})
            continue
        room, removed = empty_room(scanned)
        for name, ids in SCENARIOS.items():
            assets = [a for s in load_products(PRODUCTS, ids) for a in box_assets(s)]
            problem = furnish(room, assets)
            t = time.time()
            built, result = plan(problem, None, k=3, seconds=args.seconds)
            feas = [lay for lay in result.layouts if lay.feasible]
            row = {
                "room": r["id"], "kind": r["kind"], "scenario": name,
                "floor_m2": round(floor_area(room), 1), "removed": len(removed),
                "fixed_left": len(room.assets), "feasible_layouts": len(feas),
                "seconds": round(time.time() - t, 1), "notes": result.notes,
            }  # fmt: skip
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ("room", "scenario", "floor_m2", "feasible_layouts")}
                )
            )
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
