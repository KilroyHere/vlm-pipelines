#!/usr/bin/env python
"""Pick the best video frame of every asset and write SAM 3D item folders.

    python pick_frames.py data/45662987 --items items      # -> items/45662987__sofa_1/...

This is what the product does with the user's capture: the scan gives each object's 3D
box, the video gives the camera poses, so the best view of each object can be chosen
geometrically and the projected box becomes the SAM 2 prompt (no text detector needed).

For each asset (its ARKitScenes box, in the world frame) and each 1920x1440 ``wide``
frame, using the nearest 10 Hz pose (``lowres_wide.traj``, world -> camera, OpenCV
camera axes) and the frame's own intrinsics:

- all 8 box corners in front of the camera (z > 0.3 m); at least half of the projected
  box inside the image, scored by the visible share cubed (ARKitScenes captures are
  close-up sweeps: big sofas are often never fully in one frame, so truncated views are
  allowed and flagged rather than dropped);
- size: the visible part covers 3-80% of the image (score peaks at ~25%);
- occlusion: share of the projected box covered by nearer annotated boxes (penalized);
- view: the camera sits on the object's front side (the ground-truth heuristic front),
  ideally 20-50 degrees off-axis (SAM 3D reconstructs better from a 3/4 view).

The chosen frame is rotated upright (metadata ``sky_direction``) and written with
``meta.json`` (category/subtype, dims from the scanned box, prompt box in the rotated
image). Masks are made next by ``mask_items.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from fetch_scene import fetch_frames
from PIL import Image
from shapely.geometry import MultiPoint, box
from to_problem import load_boxes

HERE = Path(__file__).resolve().parent
ROTATE = {
    "Up": None,
    "Left": Image.Transpose.ROTATE_270,
    "Down": Image.Transpose.ROTATE_180,
    "Right": Image.Transpose.ROTATE_90,
}


def rodrigues(r: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def load_traj(path: Path) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    ts, poses = [], []
    for line in path.read_text().split("\n"):
        tok = line.split()
        if len(tok) != 7:
            continue
        v = [float(x) for x in tok]
        ts.append(v[0])
        poses.append((rodrigues(np.array(v[1:4])), np.array(v[4:7])))
    return np.array(ts), poses


def corners(b: dict) -> np.ndarray:
    half = b["size"] / 2
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return b["center"] + (signs * half) @ b["axes"]  # rows of axes = box axes in world


def project(P: np.ndarray, R: np.ndarray, t: np.ndarray, K: list[float]):
    w, h, fx, fy, cx, cy = K
    X = P @ R.T + t
    z = X[:, 2]
    uv = np.c_[fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy]
    return uv, z, (w, h)


def front_world(b: dict, scene_meta: dict, initial_yaw: float) -> np.ndarray:
    """Asset front (from problem.json yaw, room frame) back in the world frame."""
    f_room = np.array([-math.sin(initial_yaw), math.cos(initial_yaw)])
    th = scene_meta["theta"]  # room = Rz(-theta) world  =>  world = Rz(theta) room
    c, s = math.cos(th), math.sin(th)
    return np.array([c * f_room[0] - s * f_room[1], s * f_room[0] + c * f_room[1], 0.0])


def score_view(b, others, R, t, K, front):
    uv, z, (w, h) = project(corners(b), R, t, K)
    if (z < 0.3).any():
        return None
    hull = MultiPoint([tuple(p) for p in uv]).convex_hull
    frame = box(0, 0, w, h)
    seen = hull.intersection(frame)
    vis = seen.area / hull.area  # share of the object inside the image
    frac = seen.area / (w * h)
    if vis < 0.5 or not 0.03 <= frac <= 0.8:
        return None
    size_s = math.exp(-((math.log(frac / 0.25)) ** 2) / 0.5)
    cam = -R.T @ t
    dist = np.linalg.norm(b["center"] - cam)
    occl = 0.0
    for o in others:
        ouv, oz, _ = project(corners(o), R, t, K)
        if (oz < 0.1).any() or np.linalg.norm(o["center"] - cam) >= dist:
            continue
        occl += seen.intersection(MultiPoint([tuple(p) for p in ouv]).convex_hull).area
    occl = min(occl / max(seen.area, 1e-9), 1.0)
    to_cam = cam - b["center"]
    to_cam[2] = 0
    ang = math.degrees(math.acos(np.clip(to_cam @ front / (np.linalg.norm(to_cam) + 1e-9), -1, 1)))
    view_s = 0.25 if ang > 75 else (1.0 if 15 <= ang <= 50 else 0.8)
    x0, y0, x1, y1 = seen.bounds
    return {
        "score": vis**3 * size_s * (1 - occl) ** 2 * view_s,
        "frac": frac,
        "occl": occl,
        "visible": vis,
        "angle": ang,
        "bbox": [x0, y0, x1, y1],
        "size": (w, h),
    }


def rotate_box(bb, size, sky):
    x0, y0, x1, y1 = bb
    w, h = size
    if sky in ("Up", None):
        return bb
    if sky == "Left":  # ROTATE_270 (clockwise 90): (x, y) -> (h - y, x)
        return [h - y1, x0, h - y0, x1]
    if sky == "Down":
        return [w - x1, h - y1, w - x0, h - y0]
    if sky == "Right":  # ROTATE_90 (counter-clockwise): (x, y) -> (y, w - x)
        return [y0, w - x1, y1, w - x0]
    raise ValueError(sky)


def sky_direction(video_id: str) -> str:
    meta = HERE / "data" / "metadata.csv"
    for row in csv.DictReader(open(meta)):
        if row["video_id"] == video_id:
            return row["sky_direction"]
    raise KeyError(video_id)


def pick(scene: Path, items: Path, top: int = 1) -> list[dict]:
    from roomlayout.schema import LayoutProblem

    vid = scene.name
    problem = LayoutProblem.model_validate_json((scene / "problem.json").read_text())
    smeta = json.loads((scene / "scene.json").read_text())
    index = json.loads((scene / "wide_index.json").read_text())
    boxes = {b["uid"]: b for b in load_boxes(scene)}
    ts, poses = load_traj(scene / "lowres_wide.traj")
    sky = sky_direction(vid)
    annotated = list(boxes.values())
    chosen = []
    for asset in problem.assets:
        b = boxes[smeta["assets"][asset.id]["uid"]]
        front = front_world(b, smeta, problem.initial[asset.id].yaw)
        others = [o for o in annotated if o is not b]
        best = []
        for stamp in index["frames"]:
            i = int(np.argmin(np.abs(ts - float(stamp))))
            if abs(ts[i] - float(stamp)) > 0.06 or stamp not in index["intrinsics"]:
                continue
            R, t = poses[i]
            s = score_view(b, others, R, t, index["intrinsics"][stamp], front)
            if s:
                best.append((s["score"], stamp, s))
        best.sort(key=lambda x: -x[0])
        if not best:
            print(f"{vid} {asset.id}: no frame shows at least half of it; stays a box")
            continue
        score, stamp, s = best[0]
        (png,) = fetch_frames(vid, scene, [stamp])
        item = items / f"{vid}__{asset.id}"
        item.mkdir(parents=True, exist_ok=True)
        im = Image.open(png).convert("RGB")
        if ROTATE[sky] is not None:
            im = im.transpose(ROTATE[sky])
        im.save(item / "image.png")
        bb = rotate_box(s["bbox"], s["size"], sky)
        meta = {
            "category": asset.subtype or asset.category,
            "subtype": asset.subtype,
            "dims": asset.dimensions.model_dump(),
            "dims_from": "roomplan",
            "dims_note": "ARKitScenes annotated box (stands in for a RoomPlan object box)",
            "condition": ["in_room", "video_frame"],
            "source": f"ARKitScenes raw/Validation/{vid} wide frame {stamp}",
            "frame": {
                "timestamp": stamp,
                "sky_direction": sky,
                "score": round(score, 3),
                "box_area_frac": round(s["frac"], 3),
                "occlusion": round(s["occl"], 3),
                "visible_fraction": round(s["visible"], 3),
                "truncated": s["visible"] < 0.9,
                "angle_from_front_deg": round(s["angle"], 1),
            },
            "prompt_box": [round(v, 1) for v in bb],
            "asset_id": asset.id,
            "video_id": vid,
        }
        (item / "meta.json").write_text(json.dumps(meta, indent=1))
        chosen.append(meta)
        print(
            f"{vid} {asset.id}: frame {stamp} score {score:.2f} vis {s['visible']:.2f} "
            f"area {s['frac']:.2f} "
            f"occl {s['occl']:.2f} angle {s['angle']:.0f}"
        )
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scenes", nargs="+", type=Path)
    p.add_argument("--items", type=Path, default=HERE / "data" / "items")
    a = p.parse_args()
    for scene in a.scenes:
        pick(scene, a.items)


if __name__ == "__main__":
    main()
