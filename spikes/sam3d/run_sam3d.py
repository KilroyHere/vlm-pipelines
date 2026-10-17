#!/usr/bin/env python
"""Batch-run SAM 3D Objects over a folder of items, then normalize with roomlayout.assets.

UNTESTED ON GPU. The generation stage follows the official README/demo.py of
facebookresearch/sam-3d-objects (commit f91db41, 2026-06-02) but has not been executed:
this repo's dev machines have no GPU. The CPU half (--skip-generation, --dry-run) is
covered by tests/assets/test_asset_spike_harness.py.

Layout:

    data/<item_id>/image.jpg|png   photo (EXIF rotation is applied on load)
    data/<item_id>/mask.png        object mask, same size as the image (non-zero = object)
    data/<item_id>/meta.json       {"category": "sofa", "dims": {"width": 2.1, "depth": 0.9,
                                    "height": 0.82}, "dims_provenance": "tape_measured",
                                    "source_url": "...",
                                    "condition": ["in_room", "occluded"], "notes": "..."}

    meta "dims_provenance": tape_measured | manufacturer | seller_reported | estimated is
    informational and maps to the contract's dims_from (DIMS_FROM below); "source_url" records
    where a web photo came from. A legacy "dims_from" is still accepted without provenance.

    outputs/<item_id>/sam3d.glb           raw SAM 3D mesh (glTF Y-up, vertex colours)
    outputs/<item_id>/splat.ply           Gaussian splat (--save-splat)
    outputs/<item_id>/pose.json           SAM 3D layout: rotation/translation/scale in camera
    outputs/<item_id>/normalized/...      roomlayout.assets outputs (GLB, Asset, report, PNGs)
    outputs/<item_id>/summary.json        per-item summary (status, timings, key metrics)
    outputs/summary.csv, summary.json     all items + environment info

Per ADR-0004, SAM 3D is imported only inside ``generate()``, and roomlayout runs in a
subprocess with its own interpreter (``--roomlayout-python``). SAM 3D's env (Python 3.11,
torch 2.5.1/cu121) and ours (Python 3.13) never share a process.

Usage (on the GPU host, inside ``mamba activate sam3d-objects``):

    python run_sam3d.py --data data --out outputs --sam3d-repo ~/sam-3d-objects \\
        --roomlayout-python ~/vlm-pipelines/.venv/bin/python

    python run_sam3d.py --data data --out outputs --dry-run           # validate inputs only
    python run_sam3d.py --data data --out outputs --skip-generation   # re-normalize (CPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_NAMES = ("image.jpg", "image.jpeg", "image.png")
RAW_MESH = "sam3d.glb"
# Finer dimension provenance -> AssetSource.dimensions_from in the contract.
DIMS_FROM = {
    "tape_measured": "manual",
    "manufacturer": "listing",
    "seller_reported": "listing",
    "estimated": "manual",
}


def dims_from(meta: dict) -> str:
    """Contract ``dims_from`` for an item: from ``dims_provenance``, else legacy field."""
    prov = meta.get("dims_provenance")
    if prov:
        return DIMS_FROM[prov]
    return meta.get("dims_from", "manual")


@dataclass
class Item:
    item_id: str
    dir: Path
    image: Path | None
    mask: Path | None
    meta: dict
    problems: list[str] = field(default_factory=list)

    @property
    def dims(self) -> tuple[float, float, float] | None:
        d = self.meta.get("dims")
        if not d:
            return None
        return float(d["width"]), float(d["depth"]), float(d["height"])


@dataclass
class Summary:
    item_id: str
    category: str = ""
    condition: str = ""
    source_url: str = ""
    dims_provenance: str = ""  # finer than the contract's dims_from; kept for evaluation
    dims_from: str = ""
    # pending | invalid | generated | generation_failed | ok | normalize_failed | skipped
    status: str = "pending"
    error: str = ""
    generation_s: float | None = None
    vram_peak_gb: float | None = None
    raw_faces: int | None = None
    raw_vertices: int | None = None
    normalize_s: float | None = None
    aspect_mismatch: float | None = None
    mismatch_flagged: bool | None = None
    front_chosen: str = ""
    front_confidence: float | None = None
    footprint_vertices: int | None = None
    footprint_bbox_fill: float | None = None
    debris_removed: int | None = None
    tilt_corrected_deg: float | None = None
    warnings: list[str] = field(default_factory=list)
    asset_json: str = ""
    contact_sheet: str = ""


# --------------------------------------------------------------------------- inputs


def discover(data_dir: Path, only: set[str] | None) -> list[Item]:
    items = []
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if only and d.name not in only:
            continue
        image = next((d / n for n in IMAGE_NAMES if (d / n).exists()), None)
        mask = d / "mask.png" if (d / "mask.png").exists() else None
        problems = []
        meta: dict = {}
        if image is None:
            problems.append("missing image.jpg/png")
        if mask is None:
            problems.append("missing mask.png")
        try:
            meta = json.loads((d / "meta.json").read_text())
        except FileNotFoundError:
            problems.append("missing meta.json")
        except json.JSONDecodeError as exc:
            problems.append(f"meta.json is not valid JSON: {exc}")
        if meta and not meta.get("category"):
            problems.append("meta.json needs 'category'")
        if meta.get("dims"):
            try:
                vals = [float(meta["dims"][k]) for k in ("width", "depth", "height")]
                if min(vals) <= 0 or max(vals) > 6:
                    problems.append(f"dims look wrong (meters?): {vals}")
            except (KeyError, TypeError, ValueError):
                problems.append("meta.json 'dims' needs width/depth/height in meters")
        prov = meta.get("dims_provenance")
        if prov is not None and prov not in DIMS_FROM:
            problems.append(f"meta.json 'dims_provenance' must be one of {tuple(DIMS_FROM)}")
        if meta.get("dims_from") not in (None, "manual", "listing", "roomplan"):
            problems.append("meta.json 'dims_from' must be manual, listing or roomplan")
        items.append(Item(d.name, d, image, mask, meta, problems))
    return items


def check_image_mask(item: Item) -> None:
    """Load both with PIL (EXIF-transposed) and verify sizes match. Raises ValueError."""
    from PIL import Image, ImageOps

    with Image.open(item.image) as im:
        size = ImageOps.exif_transpose(im).size
    with Image.open(item.mask) as m:
        msize = m.size
    if size != msize:
        raise ValueError(
            f"mask size {msize} != image size {size} (after EXIF rotation). Re-export the "
            "image upright or redraw the mask on the upright image (make_mask.py does both)"
        )


def load_rgb_and_mask(item: Item):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(item.image) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
    with Image.open(item.mask) as m:
        arr = np.array(m)
    mask = arr[..., -1] > 0 if arr.ndim == 3 else arr > 0  # same rule as SAM 3D load_mask
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {rgb.shape[:2]}")
    if mask.sum() < 100:
        raise ValueError(f"mask has only {int(mask.sum())} pixels")
    return rgb, mask


# --------------------------------------------------------------------------- SAM 3D


class Sam3D:
    """Thin wrapper around the official ``notebook/inference.py`` API (README quick start)."""

    def __init__(self, repo: Path, tag: str = "hf", compile_model: bool = False):
        if "CONDA_PREFIX" not in os.environ:
            # notebook/inference.py does os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
            raise RuntimeError("activate the sam3d-objects conda/mamba env first")
        config = repo / "checkpoints" / tag / "pipeline.yaml"
        if not config.exists():
            raise FileNotFoundError(f"{config} not found: download checkpoints (README step 2)")
        sys.path.insert(0, str(repo / "notebook"))
        os.chdir(repo)  # the demo is run from the repo root; keep relative paths working
        import torch
        from inference import Inference  # SAM 3D's notebook/inference.py; only imported here

        self.torch = torch
        self.inference = Inference(str(config), compile=compile_model)

    def run(self, rgb, mask, seed: int) -> tuple[dict, float, float | None]:
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = self.inference(rgb, mask, seed=seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        return output, dt, vram


def _to_list(x):
    try:
        return x.detach().cpu().tolist()
    except AttributeError:
        return x


def save_generation(output: dict, out_dir: Path, save_splat: bool) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = output.get("glb")
    if glb is None:
        raise RuntimeError("SAM 3D returned no mesh ('glb' is None)")
    # output["glb"] is a trimesh.Trimesh already rotated Z-up -> Y-up by SAM 3D's to_glb.
    glb.export(str(out_dir / RAW_MESH))
    if save_splat and output.get("gs") is not None:
        output["gs"].save_ply(str(out_dir / "splat.ply"))
    pose = {k: _to_list(output[k]) for k in ("rotation", "translation", "scale") if k in output}
    (out_dir / "pose.json").write_text(json.dumps(pose, indent=2))
    return len(glb.faces), len(glb.vertices)


def generate(items: list[Item], args, summaries: dict[str, Summary]) -> None:
    runnable = [i for i in items if summaries[i.item_id].status == "pending"]
    if not runnable:
        return
    print(f"[sam3d] loading model from {args.sam3d_repo} (tag {args.tag})", flush=True)
    t0 = time.perf_counter()
    model = Sam3D(Path(args.sam3d_repo).expanduser().resolve(), args.tag, args.compile)
    print(f"[sam3d] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    for item in runnable:
        s = summaries[item.item_id]
        out_dir = args.out / item.item_id
        try:
            rgb, mask = load_rgb_and_mask(item)
            output, dt, vram = model.run(rgb, mask, args.seed)
            s.generation_s = round(dt, 3)
            s.vram_peak_gb = round(vram, 2) if vram is not None else None
            s.raw_faces, s.raw_vertices = save_generation(output, out_dir, args.save_splat)
            s.status = "generated"
            print(f"[sam3d] {item.item_id}: {dt:.1f}s, {s.raw_faces} faces", flush=True)
            del output
        except Exception as exc:  # one bad item must not stop the batch
            s.status = "generation_failed"
            s.error = f"{type(exc).__name__}: {exc}"
            (out_dir).mkdir(parents=True, exist_ok=True)
            (out_dir / "generation_error.txt").write_text(traceback.format_exc())
            print(f"[sam3d] {item.item_id}: FAILED {s.error}", flush=True)
            if "out of memory" in str(exc).lower():
                model.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- normalize


def normalize(item: Item, args, s: Summary) -> None:
    out_dir = args.out / item.item_id
    raw = out_dir / RAW_MESH
    if not raw.exists():
        s.status = "skipped" if args.skip_generation else s.status
        s.error = s.error or f"no {RAW_MESH}"
        return
    norm_dir = out_dir / "normalized"
    cmd = [
        args.roomlayout_python,
        "-m",
        "roomlayout.assets",
        str(raw),
        "--category",
        item.meta["category"],
        "--id",
        item.item_id,
        "--up",
        "y",
        "--geometry-from",
        "sam3d",
        "--out",
        str(norm_dir),
    ]
    if item.dims is not None:
        cmd += ["--dims", *(f"{v:.4f}" for v in item.dims)]
        cmd += ["--dims-from", dims_from(item.meta)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.normalize_timeout)
    except subprocess.TimeoutExpired:
        s.status, s.error = "normalize_failed", f"timeout after {args.normalize_timeout}s"
        return
    s.normalize_s = round(time.perf_counter() - t0, 3)
    norm_dir.mkdir(parents=True, exist_ok=True)
    (norm_dir / "cli_stdout.txt").write_text(proc.stdout)
    (norm_dir / "cli_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        s.status = "normalize_failed"
        s.error = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        return
    report = json.loads((norm_dir / f"{item.item_id}.report.json").read_text())
    s.status = "ok"
    s.aspect_mismatch = report["fit"]["aspect_mismatch"]
    s.mismatch_flagged = report["fit"]["flagged"]
    s.front_chosen = report["front"]["chosen"]
    s.front_confidence = report["front"]["confidence"]
    s.footprint_vertices = report["footprint"]["vertex_count"]
    s.footprint_bbox_fill = report["footprint"]["bbox_fill"]
    s.debris_removed = report["debris"]["removed_components"]
    s.tilt_corrected_deg = report.get("tilt_corrected_deg")
    s.warnings = report["warnings"]
    s.asset_json = str((norm_dir / f"{item.item_id}.asset.json").relative_to(args.out))
    sheet = report["front"].get("contact_sheet")
    s.contact_sheet = str((norm_dir / sheet).relative_to(args.out)) if sheet else ""


# --------------------------------------------------------------------------- reporting


def environment(args) -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if shutil.which("nvidia-smi"):
        q = "--query-gpu=name,memory.total,driver_version"
        r = subprocess.run(
            ["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True
        )
        env["gpu"] = r.stdout.strip()
    repo = Path(args.sam3d_repo).expanduser() if args.sam3d_repo else None
    if repo and (repo / ".git").exists():
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        env["sam3d_commit"] = r.stdout.strip()
    return env


def write_summaries(summaries: dict[str, Summary], items: list[Item], args, env: dict) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        s = summaries[item.item_id]
        d = asdict(s)
        d["dims"] = item.meta.get("dims")
        d["notes"] = item.meta.get("notes", "")
        item_dir = args.out / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "summary.json").write_text(json.dumps(d, indent=2))
        rows.append(d)
    (args.out / "summary.json").write_text(
        json.dumps({"environment": env, "items": rows}, indent=2)
    )
    cols = [f for f in Summary.__dataclass_fields__ if f != "warnings"] + ["n_warnings"]
    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_warnings": len(r["warnings"])})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sam3d-repo", default=os.environ.get("SAM3D_REPO", "~/sam-3d-objects"))
    p.add_argument("--tag", default="hf", help="checkpoint folder under checkpoints/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", help="Inference(compile=True)")
    p.add_argument("--save-splat", action="store_true")
    p.add_argument("--items", help="comma-separated item ids (default: all)")
    p.add_argument("--roomlayout-python", default=sys.executable)
    p.add_argument("--normalize-timeout", type=int, default=300)
    p.add_argument("--skip-generation", action="store_true", help="only (re)normalize")
    p.add_argument("--skip-normalize", action="store_true", help="only generate")
    p.add_argument("--dry-run", action="store_true", help="validate inputs, run nothing")
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    args.data = args.data.resolve()

    only = set(args.items.split(",")) if args.items else None
    items = discover(args.data, only)
    if not items:
        print(f"no items under {args.data}", file=sys.stderr)
        return 2
    summaries: dict[str, Summary] = {}
    for item in items:
        s = Summary(
            item.item_id,
            category=item.meta.get("category", ""),
            condition=",".join(item.meta.get("condition", [])),
            source_url=item.meta.get("source_url", ""),
            dims_provenance=item.meta.get("dims_provenance", ""),
        )
        if not item.problems:
            try:
                check_image_mask(item)
            except Exception as exc:
                item.problems.append(str(exc))
        if item.problems:
            s.status, s.error = "invalid", "; ".join(item.problems)
        else:
            s.dims_from = dims_from(item.meta) if item.dims is not None else "mesh"
        summaries[item.item_id] = s
        flag = "OK " if not item.problems else "BAD"
        dims = item.dims or "NO DIMS"
        print(
            f"[{flag}] {item.item_id:24s} {s.category:14s} {dims} "
            f"{'; '.join(item.problems)}"
        )
        if not item.problems and item.dims is None:
            print(f"      {item.item_id}: no measured dims; the asset will be flagged 'mesh'")
        if s.dims_provenance in ("manufacturer", "seller_reported") and not s.source_url:
            print(f"      {item.item_id}: {s.dims_provenance} dims but no source_url (record it)")

    if args.dry_run:
        bad = sum(1 for i in items if i.problems)
        print(f"dry run: {len(items) - bad} valid, {bad} invalid")
        return 1 if bad else 0

    env = environment(args)
    if not args.skip_generation:
        generate(items, args, summaries)
    if not args.skip_normalize:
        for item in items:
            s = summaries[item.item_id]
            if s.status in ("invalid", "generation_failed"):
                continue
            normalize(item, args, s)
            print(f"[norm ] {item.item_id}: {s.status} {s.error}", flush=True)
    write_summaries(summaries, items, args, env)
    counts: dict[str, int] = {}
    for s in summaries.values():
        counts[s.status] = counts.get(s.status, 0) + 1
    print("done:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"summary: {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
