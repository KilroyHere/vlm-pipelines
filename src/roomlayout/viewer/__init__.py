"""Web viewer (ADR-0010): a static three.js page plus one ``scene.json`` per result.

    python -m roomlayout.viewer export PROBLEM.json RESULT.json --out DIR [--fit fit.json]
        [--meshes DIR_WITH_<asset_id>.glb]
    python -m roomlayout.viewer serve DIR [--port 8000]     # adds drag / lock / re-solve

``scene.json`` is the only interface between Python and the page. Coordinates stay in
our frame (meters, Z-up, yaw CCW, front = local +Y; ADR-0001). The page converts to
three.js's Y-up frame: (x, y, z)_ours -> (x, z, -y)_three, yaw -> rotation about +Y.
GLBs written by ``roomlayout.assets`` are already in glTF's Y-up convention, in the
asset's local frame, so they drop in with the same transform as the boxes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from roomlayout.constraints.vlm.summary import wall_names
from roomlayout.explain import explain_layout
from roomlayout.schema import Layout, LayoutProblem, SolveResult

STATIC = Path(__file__).resolve().parent / "static"
SCENE_VERSION = 1


def _layout_json(problem: LayoutProblem, layout: Layout) -> dict[str, Any]:
    return {
        "id": layout.id,
        "feasible": layout.feasible,
        "cost": round(layout.cost, 4),
        "placements": {
            p.asset_id: {"x": p.pose.x, "y": p.pose.y, "yaw": p.pose.yaw} for p in layout.placements
        },
        "lines": [
            {"kind": x.kind, "text": x.text, "origin": x.origin, "assets": list(x.assets)}
            for x in explain_layout(problem, layout)
        ],
    }


def scene_json(
    problem: LayoutProblem,
    result: SolveResult | None,
    *,
    fit: dict[str, Any] | None = None,
    meshes: dict[str, str] | None = None,
    request: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Everything the page draws, JSON-ready. ``meshes`` maps asset id -> GLB path
    relative to the page."""
    meshes = meshes or {}
    names = wall_names(problem)
    room = problem.room
    layouts = [] if result is None else [_layout_json(problem, lay) for lay in result.layouts]
    pinned = {aid: {"x": p.x, "y": p.y, "yaw": p.yaw} for aid, p in problem.initial.items()}
    return {
        "version": SCENE_VERSION,
        "title": title or problem.id,
        "request": request,
        "room": {
            "floor": [list(p) for p in room.floor.points],
            "ceiling_height": room.ceiling_height,
            "walls": [
                {
                    "id": w.id,
                    "name": names.get(w.id, w.id),
                    "start": list(w.start),
                    "end": list(w.end),
                    "height": w.height,
                    "thickness": w.thickness,
                }
                for w in room.walls
            ],  # fmt: skip
            "openings": [
                {
                    "id": o.id,
                    "type": o.type.value,
                    "wall_id": o.wall_id,
                    "offset": o.offset,
                    "width": o.width,
                    "height": o.height,
                    "sill": o.sill_height,
                    "swing": None if o.swing is None else o.swing.model_dump(mode="json"),
                }
                for o in room.openings
            ],  # fmt: skip
            "obstacles": [
                {
                    "id": ob.id,
                    "label": ob.label,
                    "height": ob.height,
                    "footprint": [list(p) for p in ob.footprint.points],
                }
                for ob in room.obstacles
            ],  # fmt: skip
        },
        "assets": [
            {
                "id": a.id,
                "label": a.label,
                "category": a.category,
                "subtype": a.subtype,
                "dims": a.dimensions.model_dump(),
                "elevation": a.elevation,
                "footprint": [list(p) for p in a.effective_footprint().points],
                "l_shaped": a.footprint is not None and len(a.footprint.points) == 6,
                "mount": a.mount.value,
                "movable": a.movable,
                "mesh": meshes.get(a.id),
            }
            for a in problem.assets
        ],  # fmt: skip
        "pinned": pinned,
        "layouts": layouts,
        "fit": fit,
    }


def export(
    problem: LayoutProblem,
    result: SolveResult | None,
    out_dir: str | Path,
    *,
    fit: dict[str, Any] | None = None,
    mesh_dir: str | Path | None = None,
    request: str | None = None,
    title: str | None = None,
) -> Path:
    """Write the page (index.html, viewer.js, viewer.css), ``scene.json``, the problem
    (for re-solving when served) and any ``<asset_id>.glb`` found in ``mesh_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in STATIC.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    meshes: dict[str, str] = {}
    if mesh_dir is not None:
        (out / "meshes").mkdir(exist_ok=True)
        for a in problem.assets:
            src = Path(mesh_dir) / f"{a.id}.glb"
            if src.exists():
                shutil.copy2(src, out / "meshes" / src.name)
                meshes[a.id] = f"meshes/{src.name}"
    scene = scene_json(problem, result, fit=fit, meshes=meshes, request=request, title=title)
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    (out / "problem.json").write_text(problem.model_dump_json(indent=1))
    return out / "index.html"
