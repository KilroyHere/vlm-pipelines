#!/usr/bin/env python
"""End-to-end on an ARKitScenes room: scan + generated assets + request -> layouts.

    python run_e2e.py 42445441 --assets ../sam3d/outputs_arkit --out out

Stages, each through its public API only:

1. ``data/<id>/problem.json`` (``to_problem.py``): room outline, scanned furniture boxes,
   the real arrangement as ``initial``.
2. Assets: every object with a SAM 3D result (``<assets>/<id>__<asset>/normalized/
   *.asset.json``, from ``run_sam3d.py``) replaces its box asset: same id, scanned dims,
   mesh footprint, mesh. Objects without one stay boxes (the product's fallback).
3. Request: ``requests.json`` text -> VLM structured output (see the note in that file:
   hand-written in the VLM's format and served offline, no API key here) ->
   ``build_constraints`` (user constraints + rule library).
4. ``solve`` (k=3), then ``check_layout`` on each layout.
5. Comparison with the real room: the same constraints scored on the real arrangement
   (``evaluate``), per-object displacement and rotation, and against-the-wall agreement.

Writes ``out/<id>/``: ``problem_e2e.json``, ``result.json``, ``report.json``,
``real.png``, ``layout_<k>.png`` and ``scene3d_<real|layout_1>.png``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from roomlayout.constraints import build_constraints
from roomlayout.constraints.checks import check_layout
from roomlayout.constraints.vlm.offline import offline_client, synthetic_response
from roomlayout.schema import Asset, LayoutProblem
from roomlayout.schema.layout import Layout, Placement
from roomlayout.solver import evaluate, solve
from roomlayout.solver.render import render_layout

HERE = Path(__file__).resolve().parent


def swap_in_assets(
    problem: LayoutProblem, assets_dir: Path, vid: str
) -> tuple[LayoutProblem, dict]:
    data = problem.model_dump(mode="json")
    info = {}
    for i, a in enumerate(problem.assets):
        norm = assets_dir / f"{vid}__{a.id}" / "normalized"
        found = sorted(norm.glob("*.asset.json")) if norm.exists() else []
        if not found:
            info[a.id] = {"geometry": "box"}
            continue
        gen = Asset.model_validate_json(found[0].read_text())
        new = gen.model_copy(update={"id": a.id, "label": a.label, "movable": a.movable})
        if new.mesh is not None:  # mesh uri is relative to the asset JSON
            new = new.model_copy(
                update={
                    "mesh": new.mesh.model_copy(
                        update={"uri": str((found[0].parent / new.mesh.uri).resolve())}
                    )
                }
            )
        data["assets"][i] = new.model_dump(mode="json")
        report = json.loads(next(norm.glob("*.report.json")).read_text())
        info[a.id] = {
            "geometry": "sam3d",
            "front_confidence": gen.source.front_confidence,
            "aspect_mismatch": gen.source.aspect_mismatch,
            "footprint_bbox_fill": report.get("footprint", {}).get("bbox_fill")
            if isinstance(report.get("footprint"), dict)
            else None,
        }
    return LayoutProblem.model_validate(data), info


def real_layout(problem: LayoutProblem) -> Layout:
    results, feasible, cost = evaluate(problem)
    return Layout(
        id="real",
        placements=[Placement(asset_id=k, pose=p) for k, p in problem.initial.items()],
        cost=cost,
        feasible=feasible,
        results=results,
    )


def _against_any_wall(problem: LayoutProblem, a: Asset, pose) -> bool:
    from shapely.geometry import LineString, Point

    back = (
        pose.x + math.sin(pose.yaw) * a.dimensions.depth / 2,
        pose.y - math.cos(pose.yaw) * a.dimensions.depth / 2,
    )
    return (
        min(LineString([w.start, w.end]).distance(Point(back)) for w in problem.room.walls) < 0.15
    )


def compare(problem: LayoutProblem, layout: Layout) -> dict:
    by = {a.id: a for a in problem.assets}
    rows = {}
    for p in layout.placements:
        r = problem.initial[p.asset_id]
        d = math.hypot(p.pose.x - r.x, p.pose.y - r.y)
        rot = abs((math.degrees(p.pose.yaw - r.yaw) + 180) % 360 - 180)
        a = by[p.asset_id]
        rows[p.asset_id] = {
            "moved_m": round(d, 2),
            "rotated_deg": round(rot),
            "wall_real": _against_any_wall(problem, a, r),
            "wall_new": _against_any_wall(problem, a, p.pose),
        }
    n = len(rows)
    return {
        "per_asset": rows,
        "mean_moved_m": round(sum(r["moved_m"] for r in rows.values()) / n, 2),
        "stayed_within_0.3m": sum(r["moved_m"] < 0.3 for r in rows.values()),
        "wall_agreement": sum(r["wall_real"] == r["wall_new"] for r in rows.values()),
        "n": n,
    }


def render_3d(problem: LayoutProblem, layout: Layout, path: Path, title: str) -> None:
    """Perspective view: floor, low walls, every asset's mesh (or box) at its pose."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    floor = problem.room.floor.points
    ax.add_collection3d(Poly3DCollection([[(x, y, 0) for x, y in floor]], fc="#ece6dc", ec="none"))
    for w in problem.room.walls:
        (x0, y0), (x1, y1) = w.start, w.end
        ax.add_collection3d(
            Poly3DCollection(
                [[(x0, y0, 0), (x1, y1, 0), (x1, y1, 1.0), (x0, y0, 1.0)]],
                fc="#b9b2a7",
                ec="#8d867b",
                alpha=0.35,
                lw=0.3,
            )
        )
    poses = {p.asset_id: p.pose for p in layout.placements}
    for a in problem.assets:
        pose = poses[a.id]
        if a.mesh is not None and Path(a.mesh.uri).exists():
            m = trimesh.load(a.mesh.uri, force="mesh")
            if len(m.faces) > 4000:
                m = m.simplify_quadric_decimation(face_count=4000)
            cols = None
            vc = getattr(m.visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(m.vertices):
                cols = vc[m.faces].mean(1)[:, :3] / 255.0
        else:
            m = trimesh.creation.box(
                extents=(a.dimensions.width, a.dimensions.depth, a.dimensions.height)
            )
            m.apply_translation((0, 0, a.dimensions.height / 2))
            cols = None
        T = trimesh.transformations.rotation_matrix(pose.yaw, (0, 0, 1))
        T[:3, 3] = (pose.x, pose.y, a.elevation)
        m.apply_transform(T)
        tris = m.vertices[m.faces]
        shade = 0.55 + 0.45 * np.clip(m.face_normals @ np.array([0.3, -0.5, 0.8]), 0, 1)
        base = cols if cols is not None else np.tile([0.55, 0.62, 0.75], (len(tris), 1))
        ax.add_collection3d(
            Poly3DCollection(
                tris, facecolors=np.clip(base * shade[:, None], 0, 1), edgecolors="none"
            )
        )
        ax.text(
            pose.x, pose.y, a.elevation + a.dimensions.height + 0.15, a.id, fontsize=7, ha="center"
        )
    xs, ys = [p[0] for p in floor], [p[1] for p in floor]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs), min(xs) + span)
    ax.set_ylim(min(ys), min(ys) + span)
    ax.set_zlim(0, span)
    ax.view_init(elev=38, azim=-60)
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _share(results_by_id: dict, ids: set[str]) -> str:
    return f"{sum(results_by_id[c].satisfied for c in ids)}/{len(ids)}"


def run(vid: str, assets_dir: Path, out: Path, seconds: float) -> dict:
    out = out / vid
    out.mkdir(parents=True, exist_ok=True)
    problem = LayoutProblem.model_validate_json((HERE / "data" / vid / "problem.json").read_text())
    problem, asset_info = swap_in_assets(problem, assets_dir, vid)

    req = json.loads((HERE / "requests.json").read_text())[vid]
    client = offline_client([synthetic_response(req["vlm_answer"])])
    built = build_constraints(problem, req["text"], client=client)
    full = built.problem
    (out / "problem_e2e.json").write_text(full.model_dump_json(indent=1))

    real = real_layout(full)
    result = solve(full, k=3, seed=0, time_budget_s=seconds)
    (out / "result.json").write_text(result.model_dump_json(indent=1))

    render_layout(full, real, out / "real.png")
    render_3d(full, real, out / "scene3d_real.png", f"{vid}: real room (scanned poses)")
    layouts = []
    user_ids = {c.id for c in built.from_user}
    for k, lay in enumerate(result.layouts, 1):
        render_layout(full, lay, out / f"layout_{k}.png")
        checks = check_layout(full, lay)
        by = {r.constraint_id: r for r in lay.results}
        layouts.append(
            {
                "id": lay.id,
                "feasible": lay.feasible,
                "cost": round(lay.cost, 3),
                "user_constraints_satisfied": _share(by, user_ids),
                "checks_failed": [f"{c.name}: {c.detail}" for c in checks if not c.passed],
                "vs_real": compare(full, lay),
            }
        )
    if result.layouts:
        render_3d(full, result.layouts[0], out / "scene3d_layout_1.png", f"{vid}: layout 1")
    real_by = {r.constraint_id: r for r in real.results}
    report = {
        "video_id": vid,
        "request": req["text"],
        "parse_status": built.parse.status if built.parse else None,
        "unexpressed": [u.text for u in built.parse.unexpressed] if built.parse else [],
        "constraints": {"from_user": len(built.from_user), "from_rules": len(built.from_rules)},
        "assets": asset_info,
        "real": {
            "feasible": real.feasible,
            "cost": round(real.cost, 3),
            "hard_failed": [
                f"{r.constraint_id}: {r.detail}" for r in real.results if r.hard and not r.satisfied
            ],
            "user_constraints_satisfied": _share(real_by, user_ids),
            "checks_failed": [
                f"{c.name}: {c.detail}" for c in check_layout(full, real) if not c.passed
            ],
        },
        "layouts": layouts,
        "solver": {"seconds": round(result.stats.seconds, 1), "notes": result.notes},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("video_ids", nargs="+")
    p.add_argument("--assets", type=Path, default=HERE / "data" / "sam3d_outputs")
    p.add_argument("--out", type=Path, default=HERE / "out")
    p.add_argument("--seconds", type=float, default=20.0)
    a = p.parse_args()
    for vid in a.video_ids:
        r = run(vid, a.assets, a.out, a.seconds)
        print(json.dumps({k: r[k] for k in ("video_id", "real", "solver")}, indent=1))
        for lay in r["layouts"]:
            print(
                lay["id"],
                lay["feasible"],
                lay["cost"],
                lay["user_constraints_satisfied"],
                lay["vs_real"]["mean_moved_m"],
                lay["checks_failed"],
            )


if __name__ == "__main__":
    main()
