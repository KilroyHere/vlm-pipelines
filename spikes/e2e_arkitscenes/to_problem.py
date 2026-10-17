#!/usr/bin/env python
"""ARKitScenes capture -> ``LayoutProblem`` whose ``initial`` poses are the real room.

    python to_problem.py data/45662987                 # writes data/45662987/problem.json + plot

This stands in for the RoomPlan importer (M4a) on scans we did not make ourselves:

- **Frame.** ARKitScenes boxes and mesh share one world frame, meters, Z-up (gravity
  aligned: every box's third axis is +Z). We rotate about Z so the walls are axis aligned
  (dominant direction of vertical mesh faces), shift so the floor is z = 0 and the room's
  min corner is near the origin. ``scene.json`` keeps that transform for the frame picker.
- **Room.** No walls are annotated, so the floor polygon is rebuilt from the mesh: floor
  vertices plus furniture footprints on a 5 cm grid, closed, holes filled; an opening of
  0.45 m cuts doorways so adjacent rooms seen through them fall away; the outline is kept
  rectilinear (mitre joins). Walls are its CCW edges. No doors/windows are annotated, so
  the room has **no openings** (door rules are not exercised).
- **Furniture.** Each annotated box becomes an ``Asset`` (box geometry, dims from the
  box, zones from the category profile) placed at its real pose. Tables get a subtype
  from size (low and within 2.5 m of a sofa: coffee table; small: side table), standing
  in for RoomPlan's TableType attribute. ARKitScenes boxes have no
  front, so the real yaw uses a heuristic, recorded per asset: back to the nearest wall
  when within 0.35 m, else sofas face the TV (or the nearest table), chairs face the
  nearest table. Fireplaces become obstacles. A TV standing on furniture is not an asset
  (the contract has no "stands on" relation): the furniture under it becomes the
  ``tv_stand``, which the rule library already treats as the TV when there is none.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from roomlayout.assets.categories import profile_for
from roomlayout.assets.zones import build_zones
from roomlayout.schema import Asset, Dimensions, LayoutProblem
from roomlayout.schema.asset import AssetSource
from roomlayout.schema.geometry import Polygon2D, Pose2D
from roomlayout.schema.room import Obstacle, Room, Wall

CELL = 0.05
# ARKitScenes label -> (RoomPlan category, subtype)
CATEGORY = {
    "sofa": ("sofa", None),
    "chair": ("chair", None),
    "stool": ("chair", "stool"),
    "table": ("table", None),
    "cabinet": ("storage", "cabinet"),
    "shelf": ("storage", "bookshelf"),
    "tv_monitor": ("television", None),
    "bed": ("bed", None),
}
OBSTACLES = {
    "fireplace",
    "stove",
    "oven",
    "sink",
    "refrigerator",
    "dishwasher",
    "washer",
    "toilet",
    "bathtub",
}
FRONT_FROM_WALL = {"sofa", "storage", "television", "bed", "chair"}


def load_boxes(scene: Path) -> list[dict]:
    vid = scene.name
    ann = json.loads((scene / f"{vid}_3dod_annotation.json").read_text())
    boxes = []
    for d in ann["data"]:
        obb = d["segments"]["obbAligned"]
        R = np.array(obb["normalizedAxes"]).reshape(3, 3)  # rows = box axes in world
        if abs(abs(R[2, 2]) - 1) > 1e-3:
            raise ValueError(f"{vid}: box {d['uid']} is not gravity aligned")
        boxes.append(
            {
                "uid": d["uid"],
                "label": d["label"].replace(" ", "_"),
                "center": np.array(obb["centroid"]),
                "size": np.array(obb["axesLengths"]),
                "axes": R,
            }
        )
    return boxes


def wall_angle(mesh: trimesh.Trimesh) -> float:
    """Dominant wall direction (radians, mod 90 deg) from near-vertical faces."""
    n, area = mesh.face_normals, mesh.area_faces
    vertical = np.abs(n[:, 2]) < 0.2
    ang = np.arctan2(n[vertical, 1], n[vertical, 0])
    w = area[vertical]
    c, s = (w * np.cos(4 * ang)).sum(), (w * np.sin(4 * ang)).sum()
    return math.atan2(s, c) / 4


def _grid_union(cells: np.ndarray) -> Polygon | MultiPolygon:
    """Union of occupied grid cells (integer ij array) as polygons, by row runs."""
    rects = []
    for j in np.unique(cells[:, 1]):
        xs = np.sort(np.unique(cells[cells[:, 1] == j, 0]))
        if xs.size == 0:
            continue
        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[xs[0], xs[breaks + 1]]
        ends = np.r_[xs[breaks], xs[-1]]
        for a, b in zip(starts, ends, strict=True):
            rects.append(box(a * CELL, j * CELL, (b + 1) * CELL, (j + 1) * CELL))
    return unary_union(rects)


def _largest(geom) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    return max(geom.geoms, key=lambda g: g.area)


def _mitre(g, d):
    return g.buffer(d, join_style="mitre", cap_style="square")


def floor_polygon(verts: np.ndarray, footprints: list[Polygon]) -> Polygon:
    floor = verts[np.abs(verts[:, 2]) < 0.08]
    cells = np.unique(np.floor(floor[:, :2] / CELL).astype(int), axis=0)
    region = unary_union([_grid_union(cells), *footprints])
    region = _mitre(_mitre(region, 0.15), -0.15)  # close cracks between scan strips
    region = Polygon(_largest(region).exterior)  # fill holes
    # cut doorways and hallways (< 1.4 m wide); keep the part holding the most furniture
    furn = unary_union(footprints)
    parts = _mitre(_mitre(region, -0.7), 0.7)
    parts = list(getattr(parts, "geoms", [parts]))
    core = max(parts, key=lambda g: (g.intersection(furn).area, g.area))
    room = _largest(region.intersection(_mitre(core, 0.15)))
    room = _largest(_mitre(_mitre(room, -0.10), 0.10))
    # furniture touching the room stays inside it (annotation boxes overhang walls a little)
    touching = [f for f in footprints if f.intersects(room)]
    room = _largest(_mitre(_mitre(unary_union([room, *touching]), 0.05), -0.05))
    room = Polygon(room.exterior).simplify(0.12)
    # 2 cm of slack: annotation boxes overhang the reconstructed walls by millimetres,
    # which would otherwise read as hard inside_room violations of the real layout
    room = _largest(_mitre(room, 0.02)).simplify(0.01)
    from shapely.geometry.polygon import orient

    return orient(room, 1.0)  # CCW


def walls_from(poly: Polygon, height: float) -> list[Wall]:
    pts = list(poly.exterior.coords)[:-1]
    return [
        Wall(
            id=f"w{i}",
            start=tuple(map(float, pts[i])),
            end=tuple(map(float, pts[(i + 1) % len(pts)])),
            height=height,
        )
        for i in range(len(pts))
    ]


def _dist_to_walls(p: np.ndarray, poly: Polygon) -> float:
    return poly.exterior.distance(Point(p))


def choose_front(b: dict, cat: str, poly: Polygon, others: list[dict]) -> tuple[np.ndarray, str]:
    """Unit 2D front direction for box ``b`` (heuristic; see module docstring)."""
    a0, a1 = b["axes2d"]
    l0, l1 = b["len2d"]
    c = b["c2d"]
    # candidate fronts: for elongated sofas/storage/TVs the front is a long side
    cands = [a0, -a0, a1, -a1]
    if cat in ("sofa", "storage", "television") and max(l0, l1) > 1.15 * min(l0, l1):
        short = a0 if l0 < l1 else a1
        cands = [short, -short]
    best, why = None, ""
    if cat in FRONT_FROM_WALL:
        scored = []
        for f in cands:
            half = l0 / 2 if abs(f @ a0) > 0.5 else l1 / 2
            back = c - f * (half + 0.35)
            scored.append(
                (poly.exterior.distance(Point(back)) if poly.contains(Point(back)) else 0.0, f)
            )
        # back point outside or on the wall -> that side is against a wall
        d_back = []
        for f in cands:
            half = l0 / 2 if abs(f @ a0) > 0.5 else l1 / 2
            d_back.append(poly.exterior.distance(Point(c - f * half)))
        i = int(np.argmin(d_back))
        if d_back[i] < 0.35:
            best, why = cands[i], f"back to wall ({d_back[i]:.2f} m)"
    if best is None:
        target_cats = {"sofa": ["television", "tv_stand", "table"], "chair": ["table", "sofa"]}.get(
            cat, []
        )
        for tc in target_cats:
            ts = [o for o in others if tc in (o["cat"], o["subtype"]) and o is not b]
            if ts:
                t = min(ts, key=lambda o: np.linalg.norm(o["c2d"] - c))
                v = t["c2d"] - c
                best = max(cands, key=lambda f: f @ v)
                why = f"faces nearest {tc}"
                break
    if best is None:
        best, why = cands[0], "arbitrary (no wall or target)"
    return best, why


def convert(scene: Path) -> dict:
    vid = scene.name
    mesh = trimesh.load(scene / f"{vid}_3dod_mesh.ply", process=False)
    boxes = load_boxes(scene)

    theta = wall_angle(mesh)
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v = mesh.vertices @ Rz.T
    zs = v[:, 2]
    hist, edges = np.histogram(zs[zs < np.percentile(zs, 30)], bins=200)
    floor_z = float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2)
    v[:, 2] -= floor_z
    ceiling = float(np.clip(np.percentile(v[:, 2], 99.5), 2.2, 3.6))

    # boxes into the aligned frame
    for b in boxes:
        b["center_w"] = b["center"] @ Rz.T - np.array([0, 0, floor_z])
        ax = b["axes"] @ Rz.T
        horiz = [i for i in range(3) if abs(ax[i, 2]) < 0.5]
        vert = [i for i in range(3) if abs(ax[i, 2]) >= 0.5][0]
        b["axes2d"] = [ax[i, :2] / np.linalg.norm(ax[i, :2]) for i in horiz]
        b["len2d"] = [float(b["size"][i]) for i in horiz]
        b["height"] = float(b["size"][vert])
        b["c2d"] = b["center_w"][:2]
        b["bottom"] = float(b["center_w"][2] - b["height"] / 2)
        hw = [b["axes2d"][k] * b["len2d"][k] / 2 for k in range(2)]
        corners = [
            b["c2d"] + sx * hw[0] + sy * hw[1] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        b["poly"] = Polygon(corners)
        b["cat"], b["subtype"] = CATEGORY.get(b["label"], (None, None))

    # shift: room min corner to the origin
    furn = [b["poly"] for b in boxes if b["cat"] or b["label"] in OBSTACLES]
    poly = floor_polygon(v, furn)
    minx, miny = poly.bounds[0], poly.bounds[1]
    shift = np.array([-minx, -miny])
    poly = affinity.translate(poly, *shift)
    v[:, :2] += shift
    for b in boxes:
        b["c2d"] = b["c2d"] + shift
        b["poly"] = affinity.translate(b["poly"], *shift)

    notes = []
    kept, obstacles = [], []
    for b in boxes:
        inside = poly.buffer(0.05).contains(b["poly"].centroid)
        if not inside:
            notes.append(f"dropped {b['label']} {b['uid']}: outside the room outline")
            continue
        if b["label"] in OBSTACLES:
            obstacles.append(b)
        elif b["cat"] is None:
            notes.append(f"dropped {b['label']} {b['uid']}: no category mapping")
        else:
            kept.append(b)

    # TV standing on furniture: drop it (no stacking relation in the contract)
    for b in [k for k in kept if k["cat"] == "television"]:
        under = [
            o
            for o in kept
            if o is not b
            and o["cat"] in ("storage", "table")
            and o["poly"].intersects(b["poly"])
            and abs(o["bottom"] + o["height"] - b["bottom"]) < 0.25
        ]
        if under and b["bottom"] > 0.2:
            kept.remove(b)
            b["on"] = under[0]
            under[0]["cat"], under[0]["subtype"] = "storage", "tv_stand"
            notes.append(
                f"TV {b['uid']} stands on {under[0]['label']} {under[0]['uid']}: "
                "that becomes the tv_stand (the TV rides on it)"
            )
        elif b["bottom"] > 0.6:
            b["mount"] = "wall"

    counts: dict[str, int] = {}
    assets, initial, gt = [], {}, {}
    for b in kept:
        cat = b["cat"]
        key = b["subtype"] or cat
        counts[key] = counts.get(key, 0) + 1
        aid = f"{key}_{counts[key]}"
        b["id"] = aid
    # Table type (RoomPlan reports TableType coffee/dining; ARKitScenes only "table"):
    # low and within 2.5 m of a sofa -> coffee_table, small -> side_table. Ids keep "table_n".
    sofas = [o for o in kept if o["cat"] == "sofa"]
    for b in kept:
        if b["cat"] != "table" or b["subtype"]:
            continue
        near_sofa = any(np.linalg.norm(o["c2d"] - b["c2d"]) < 2.5 for o in sofas)
        if b["height"] < 0.5 and near_sofa and min(b["len2d"]) > 0.4:
            b["subtype"] = "coffee_table"
        elif max(b["len2d"]) <= 0.65:
            b["subtype"] = "side_table"
    for b in kept:
        cat, aid = b["cat"], b["id"]
        front, why = choose_front(b, cat, poly, kept)
        lateral = np.array([front[1], -front[0]])  # local +X = front rotated -90 deg
        width = b["len2d"][0] if abs(lateral @ b["axes2d"][0]) > 0.5 else b["len2d"][1]
        depth = b["len2d"][1] if abs(lateral @ b["axes2d"][0]) > 0.5 else b["len2d"][0]
        yaw = math.atan2(-front[0], front[1])
        prof = profile_for(b["subtype"] or cat)
        fp = Polygon2D.rectangle(width, depth)
        mount = b.get("mount", prof.mount)
        asset = Asset(
            id=aid,
            category=cat,
            subtype=b["subtype"],
            label=f"{b['label']} ({b['uid']})",
            dimensions=Dimensions(
                width=round(width, 3), depth=round(depth, 3), height=round(b["height"], 3)
            ),
            clearance_zones=build_zones(fp, prof.zones),
            mount=mount,
            elevation=round(max(b["bottom"], 0.0), 3) if mount == "wall" else 0.0,
            source=AssetSource(dimensions_from="roomplan", geometry_from="box"),
        )
        assets.append(asset)
        initial[aid] = Pose2D(x=float(b["c2d"][0]), y=float(b["c2d"][1]), yaw=yaw)
        gt[aid] = {
            "uid": b["uid"],
            "label": b["label"],
            "front_rule": why,
            "center": b["center_w"].tolist(),
            "bottom": b["bottom"],
        }

    room = Room(
        id=f"arkitscenes_{vid}",
        floor=Polygon2D(points=[tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1]]),
        ceiling_height=round(ceiling, 2),
        walls=walls_from(poly, round(ceiling, 2)),
        obstacles=[
            Obstacle(
                id=f"{o['label']}_{i + 1}",
                label=o["label"],
                footprint=Polygon2D(
                    points=[tuple(map(float, p)) for p in list(o["poly"].exterior.coords)[:-1]]
                ),
                height=round(o["height"], 3),
            )
            for i, o in enumerate(obstacles)
        ],
    )
    problem = LayoutProblem(id=f"arkitscenes_{vid}", room=room, assets=assets, initial=initial)
    (scene / "problem.json").write_text(problem.model_dump_json(indent=1))
    meta = {
        "video_id": vid,
        "theta": theta,
        "floor_z": floor_z,
        "shift": shift.tolist(),
        "world_to_room": "p_room = Rz(-theta) @ p_world - [0,0,floor_z] + [shift, 0]",
        "assets": gt,
        "notes": notes,
        "tv_on_furniture": [
            {
                "uid": b["uid"],
                "on": b["on"]["id"],
                "center": b["center_w"].tolist(),
                "size": b["size"].tolist(),
            }
            for b in boxes
            if "on" in b
        ],
    }
    (scene / "scene.json").write_text(json.dumps(meta, indent=1))
    plot(scene, problem, v, notes)
    return meta


def real_layout(problem: LayoutProblem):
    """The scanned arrangement as a ``Layout``, scored by the solver's own evaluation."""
    from roomlayout.schema.layout import Layout, Placement
    from roomlayout.solver import evaluate

    results, feasible, cost = evaluate(problem)
    return Layout(
        id="real",
        placements=[Placement(asset_id=k, pose=p) for k, p in problem.initial.items()],
        cost=cost,
        feasible=feasible,
        results=results,
    )


def plot(scene: Path, problem: LayoutProblem, verts: np.ndarray, notes: list[str]) -> None:
    """Real layout (solver renderer) plus a check image: mesh points under the outline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from roomlayout.solver.render import render_layout

    render_layout(problem, real_layout(problem), scene / "real_layout.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    sel = verts[(verts[:, 2] > 0.05) & (verts[:, 2] < 2.0)][::7]
    ax.scatter(sel[:, 0], sel[:, 1], s=0.2, c=sel[:, 2], cmap="viridis", alpha=0.4)
    fl = problem.room.floor.points
    ax.plot([p[0] for p in [*fl, fl[0]]], [p[1] for p in [*fl, fl[0]]], "r-", lw=2)
    for a in problem.assets:
        pose = problem.initial[a.id]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        w, d = a.dimensions.width / 2, a.dimensions.depth / 2
        pts = [
            (pose.x + c * x - s * y, pose.y + s * x + c * y)
            for x, y in ((-w, -d), (w, -d), (w, d), (-w, d), (-w, -d))
        ]
        ax.plot(*zip(*pts, strict=True), "k-", lw=1)
        ax.arrow(pose.x, pose.y, -s * (d + 0.2), c * (d + 0.2), head_width=0.08, color="k")
        ax.text(pose.x, pose.y, a.id, fontsize=7, ha="center")
    ax.set_aspect("equal")
    ax.set_title(f"{scene.name}: mesh (0-2 m) + outline + boxes")
    fig.savefig(scene / "outline_check.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scenes", nargs="+", type=Path)
    a = p.parse_args()
    for scene in a.scenes:
        meta = convert(scene)
        print(scene.name, len(meta["assets"]), "assets;", *meta["notes"], sep="\n  ")


if __name__ == "__main__":
    main()
