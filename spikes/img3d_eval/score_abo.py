"""Score image-to-3D outputs against ABO ground truth (CPU, after the pod run).

    python spikes/img3d_eval/score_abo.py --abo data/abo --results data/abo/results \
        [--methods sam3d,trellis] [--out data/abo/results/scores.json]

For every item and method:
1. ``roomlayout.assets.normalize`` the raw GLB (glTF Y-up), fitted to the item's true
   W x D x H, exactly as the product pipeline would with listing dims. Output (GLB with
   display cleanup + Asset JSON + report) goes to ``<results>/normalized/<method>/<id>/``.
2. Ground truth: ABO's GLB in our frame. glTF (X, Y, Z) -> ours (X, -Z, Y); ABO models
   face glTF +Z (checked on chairs/sofas/beds: the backrest is at -Z), which is our -Y,
   so it is turned 180 degrees to face +Y like our assets. Centred, base at z = 0.
3. Metrics (all on surface samples, distances relative to the true bbox diagonal):
   - ``aspect_mismatch``: the generator's raw proportions vs the true dims (normalize's
     report) - shape error *before* we force the size.
   - ``chamfer`` / ``fscore`` at the delivered orientation (what a user sees), and
     ``chamfer_best`` / ``fscore_best`` over the 4 yaws (shape quality alone; each yaw is
     re-fitted to the true box).
   - ``front_ok``: the delivered yaw is the best of the 4 (our front heuristic was right).
   - ``footprint_iou``: the Asset's footprint polygon vs the true top-down footprint
     (``extract_footprint`` on the ground truth, same options) - what the solver uses.
   - generation seconds and peak VRAM from the pod summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from roomlayout.assets import extract_footprint, normalize, write_outputs
from roomlayout.schema import Dimensions

N_SAMPLES = 20000
F_TAU = (0.02, 0.05)  # F-score thresholds, fraction of the true bbox diagonal
RAW_NAME = {"sam3d": "sam3d.glb", "trellis": "trellis.glb"}


def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path), force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    return mesh


def gt_in_our_frame(path: Path) -> trimesh.Trimesh:
    m = load_mesh(path).copy()
    v = np.asarray(m.vertices, float)
    ours = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])  # glTF Y-up -> Z-up
    ours[:, :2] *= -1  # face +Y (ABO models face glTF +Z = our -Y)
    lo, hi = ours.min(0), ours.max(0)
    ours -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    return trimesh.Trimesh(ours, m.faces, process=False)


def fit_box(mesh: trimesh.Trimesh, dims: np.ndarray) -> trimesh.Trimesh:
    v = np.asarray(mesh.vertices, float)
    lo, hi = v.min(0), v.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    out = (v - lo) / span * dims - [dims[0] / 2, dims[1] / 2, 0]
    return trimesh.Trimesh(out, mesh.faces, process=False)


def yawed(mesh: trimesh.Trimesh, k: int) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(k * math.pi / 2, [0, 0, 1]))
    return m


def sample(mesh: trimesh.Trimesh, seed: int = 0) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, N_SAMPLES, seed=seed)
    return np.asarray(pts)


def chamfer_f(a: np.ndarray, b: np.ndarray, diag: float) -> tuple[float, dict[str, float]]:
    da = cKDTree(b).query(a)[0] / diag  # pred -> gt (precision side)
    db = cKDTree(a).query(b)[0] / diag  # gt -> pred (recall side)
    cd = float((da.mean() + db.mean()) / 2)
    fs = {}
    for t in F_TAU:
        p, r = (da < t).mean(), (db < t).mean()
        fs[f"{t:g}"] = float(2 * p * r / (p + r)) if p + r else 0.0
    return cd, fs


def footprint_iou(asset_fp, gt_mesh: trimesh.Trimesh, dims: Dimensions) -> float | None:
    gt_fp = extract_footprint(gt_mesh, dims.width, dims.depth).polygon
    a, b = Polygon(asset_fp.points), Polygon(gt_fp.points)
    if not (a.is_valid and b.is_valid):
        a, b = a.buffer(0), b.buffer(0)
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union else None


def score_item(item: Path, raw: Path, gt: Path, method: str, out: Path) -> dict:
    meta = json.loads((item / "meta.json").read_text())
    d = meta["dims"]
    dims = Dimensions(width=d["width"], depth=d["depth"], height=d["height"])
    res = normalize(
        raw, category=meta["category"], dims=dims, asset_id=item.name,
        dimensions_from="listing", geometry_from="sam3d" if method == "sam3d" else "manual",
    )  # fmt: skip
    write_outputs(res, out, render=False)
    gt_mesh = gt_in_our_frame(gt)
    box = np.array([dims.width, dims.depth, dims.height])
    diag = float(np.linalg.norm(box))
    g = sample(gt_mesh, 1)
    per_yaw = []
    for k in range(4):
        m = res.mesh if k == 0 else fit_box(yawed(res.mesh, k), box)
        per_yaw.append(chamfer_f(sample(m), g, diag))
    best = min(range(4), key=lambda k: per_yaw[k][0])
    rep = res.report
    return {
        "item": item.name,
        "method": method,
        "category": meta["category"],
        "condition": meta.get("condition", []),
        "aspect_mismatch": rep.fit.aspect_mismatch,
        "chamfer": per_yaw[0][0],
        "fscore": per_yaw[0][1],
        "chamfer_best": per_yaw[best][0],
        "fscore_best": per_yaw[best][1],
        "front_ok": best == 0,
        "best_yaw_deg": best * 90,
        "front_confidence": rep.front.confidence,
        "footprint_iou": footprint_iou(res.asset.footprint, gt_mesh, dims),
        "warnings": rep.warnings,
        "raw_faces": rep.input_faces,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--abo", type=Path, default=Path("data/abo"))
    ap.add_argument("--results", type=Path, default=Path("data/abo/results"))
    ap.add_argument("--methods", default="sam3d,trellis")
    ap.add_argument("--items", help="comma-separated ids (default: all)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out_path = args.out or args.results / "scores.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["item"], r["method"]) for r in rows if "error" not in r}
    wanted = set(args.items.split(",")) if args.items else None
    for method in args.methods.split(","):
        timing = {}
        summ = args.results / method / "summary.json"
        if summ.exists():
            timing = json.loads(summ.read_text())
        for item in sorted((args.abo / "eval").iterdir()):
            if (wanted and item.name not in wanted) or (item.name, method) in done:
                continue
            raw = args.results / method / item.name / RAW_NAME[method]
            if not raw.exists():
                rows.append({"item": item.name, "method": method, "error": "no output"})
                continue
            try:
                row = score_item(item, raw, args.abo / "items" / item.name / "gt.glb", method,
                                 args.results / "normalized" / method / item.name)  # fmt: skip
            except Exception as e:  # report and keep going
                row = {"item": item.name, "method": method, "error": f"{type(e).__name__}: {e}"}
            t = timing.get(item.name, {})
            row["generation_s"] = t.get("generation_s")
            row["vram_peak_gb"] = t.get("vram_peak_gb")
            rows = [r for r in rows if (r["item"], r["method"]) != (item.name, method)] + [row]
            out_path.write_text(json.dumps(rows, indent=1))
            ok = "error" not in row
            print(item.name, method, f"F@5%={row['fscore']['0.05']:.2f}" if ok else row["error"])


if __name__ == "__main__":
    main()
