#!/usr/bin/env python
"""Furnish a real scanned room with new furniture from product photos (ADR-0009 demo).

    python furnish_demo.py 47204607 --products ../sam3d/outputs --out out/furnish

1. The room: an ARKitScenes LiDAR capture (``to_problem.py``) emptied of its movable
   furniture with ``roomlayout.pipeline.empty_room``.
2. The furniture: SAM 3D assets generated from retailer product photos in the pilot
   (``spikes/sam3d``, manufacturer dimensions), added with ``furnish``.
3. The request, in words, with the VLM's structured answer hand-written in its output
   format (no API key here; see requests.json's note), then rules, solver and checks.

Variants answer "which sofa fits?" (the KIVIK sectional vs a 3-seat sofa) and, for small
rooms, "what would fit?" (the sofa set without the work spot).
Writes ``out/<room>/<variant>/``: ``layout_<k>.png``, ``scene3d_<k>.png``, ``report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_e2e import render_3d

from roomlayout.constraints.checks import check_layout
from roomlayout.constraints.vlm.offline import offline_client, synthetic_response
from roomlayout.pipeline import empty_room, furnish, load_asset, plan
from roomlayout.schema import LayoutProblem
from roomlayout.solver.render import render_layout

HERE = Path(__file__).resolve().parent
WORK_SPOT = " I need a small work spot: the desk against a wall with the chair at it."
REQUEST = (
    "Put the sofa against a wall with the coffee table in front of it, and the armchair "
    "by the coffee table so we can talk." + WORK_SPOT + " The dresser goes against a wall too."
)
COMMON = ["armchair_05", "coffee_table_10", "desk_13", "dining_chair_07", "dresser_16"]
# name -> (sofa, the rest); the last one answers "what would fit?" in a small room
VARIANTS = {
    "sectional": ("sectional_04", COMMON),
    "sofa": ("sofa_01", COMMON),
    "sofa_no_desk": ("sofa_01", ["armchair_05", "coffee_table_10", "dresser_16"]),
}


def vlm_answer(sofa: str, items: list[str]) -> dict:
    """What the VLM returns for REQUEST (hand-written in its output format)."""

    def c(kind, hard, restatement, source, **kw):
        base = {"type": kind, "hard": hard, "restatement": restatement, "source_text": source}
        return {**base, **kw}

    s1 = "Put the sofa against a wall"
    s2 = "with the coffee table in front of it"
    s3 = "the armchair by the coffee table so we can talk"
    s4 = "the desk against a wall with the chair at it"
    return {
        "constraints": [
            x
            for x in [
                c(
                    "against_wall",
                    True,
                    "The sofa's back is against a wall.",
                    s1,
                    a=sofa,
                    wall_id=None,
                    max_gap=None,
                ),
                c(
                    "faces",
                    False,
                    "The sofa faces the coffee table.",
                    s2,
                    a=sofa,
                    b="coffee_table_10",
                    max_angle_deg=None,
                ),
                c(
                    "near",
                    False,
                    "The coffee table is within 0.5 m of the sofa.",
                    s2,
                    a="coffee_table_10",
                    b=sofa,
                    max_distance=0.5,
                ),
                c(
                    "near",
                    False,
                    "The armchair is within 0.8 m of the coffee table.",
                    s3,
                    a="armchair_05",
                    b="coffee_table_10",
                    max_distance=0.8,
                ),
                c(
                    "faces",
                    False,
                    "The armchair faces the coffee table.",
                    s3,
                    a="armchair_05",
                    b="coffee_table_10",
                    max_angle_deg=None,
                ),
                c(
                    "against_wall",
                    False,
                    "The desk stands against a wall.",
                    s4,
                    a="desk_13",
                    wall_id=None,
                    max_gap=None,
                ),
                c(
                    "near",
                    False,
                    "The chair is pulled up to the desk.",
                    s4,
                    a="dining_chair_07",
                    b="desk_13",
                    max_distance=0.1,
                ),
                c(
                    "faces",
                    False,
                    "The chair faces the desk.",
                    s4,
                    a="dining_chair_07",
                    b="desk_13",
                    max_angle_deg=None,
                ),
                c(
                    "against_wall",
                    True,
                    "The dresser stands against a wall.",
                    "The dresser goes against a wall too",
                    a="dresser_16",
                    wall_id=None,
                    max_gap=None,
                ),
            ]
            if {x["a"], x.get("b", sofa)} <= {sofa, *items}
        ],  # fmt: skip
        "unexpressed": [],
    }


def products(root: Path, ids: list[str]):
    return [load_asset(next((root / i / "normalized").glob("*.asset.json"))) for i in ids]


def run(vid: str, root: Path, out: Path, seconds: float) -> dict:
    scanned = LayoutProblem.model_validate_json((HERE / "data" / vid / "problem.json").read_text())
    room, removed = empty_room(scanned)
    summary = {"room": vid, "removed": removed, "request": REQUEST, "variants": {}}
    for name, (sofa, items) in VARIANTS.items():
        d = out / vid / name
        d.mkdir(parents=True, exist_ok=True)
        problem = furnish(room, products(root, [sofa, *items]))
        text = REQUEST if "desk_13" in items else REQUEST.replace(WORK_SPOT, "")
        client = offline_client([synthetic_response(vlm_answer(sofa, items))])
        built, result = plan(problem, text, client=client, k=3, seconds=seconds)
        user = {c.id for c in built.from_user}
        lays = []
        for k, lay in enumerate(result.layouts, 1):
            render_layout(built.problem, lay, d / f"layout_{k}.png")
            render_3d(built.problem, lay, d / f"scene3d_{k}.png", f"{vid}, {name}: layout {k}")
            by = {r.constraint_id: r for r in lay.results}
            lays.append({
                "feasible": lay.feasible, "cost": round(lay.cost, 3),
                "request_met": f"{sum(by[c].satisfied for c in user)}/{len(user)}",
                "unmet": [c for c in user if not by[c].satisfied],
                "checks_failed": [f"{c.name}: {c.detail}" for c in check_layout(built.problem, lay)
                                  if not c.passed],
            })  # fmt: skip
        floor = built.problem.room.floor.points
        footprint = sum(a.effective_footprint().area for a in built.problem.assets)
        report = {
            "variant": name, "sofa": sofa, "layouts": lays, "solver_notes": result.notes,
            "furniture_m2": round(footprint, 2), "floor_m2": round(_area(floor), 2),
        }  # fmt: skip
        (d / "report.json").write_text(json.dumps(report, indent=1))
        summary["variants"][name] = report
        print(name, [(x["feasible"], x["request_met"], len(x["checks_failed"])) for x in lays])
    return summary


def _area(pts) -> float:
    return (
        abs(
            sum(
                x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True)
            )
        )
        / 2
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("rooms", nargs="+")
    p.add_argument("--products", type=Path, default=HERE.parent / "sam3d" / "outputs")
    p.add_argument("--out", type=Path, default=HERE / "out" / "furnish")
    p.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    for vid in a.rooms:
        s = run(vid, a.products, a.out, a.seconds)
        (a.out / vid / "summary.json").write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
