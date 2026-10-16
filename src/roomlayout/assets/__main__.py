"""CLI: normalize a raw generator mesh into a solver-ready Asset.

    python -m roomlayout.assets RAW_MESH --category sofa --dims W D H --out DIR
        [--up y|z] [--id ID] [--label TEXT] [--dims-from roomplan|listing|manual]
        [--geometry-from sam3d|object_capture|catalog|manual|box] [--mount floor|wall]
        [--front A|B|C|D] [--mismatch-tol 0.15] [--max-footprint-vertices 32]
        [--no-clearance] [--no-render]

Writes DIR/<id>.glb, <id>.asset.json, <id>.report.json and front-candidate PNGs.
Exit codes: 0 ok (warnings go to stderr), 2 bad input.
"""

from __future__ import annotations

import argparse
import sys

from roomlayout.schema import Dimensions, Mount

from .footprint import FootprintOptions
from .meshio import MeshLoadError
from .pipeline import NormalizeOptions, normalize, write_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m roomlayout.assets",
        description="Normalize a raw mesh (glb/gltf/obj/ply) into a roomlayout Asset.",
    )
    p.add_argument("mesh", help="raw mesh file (e.g. SAM 3D Objects output .glb)")
    p.add_argument("--category", required=True, help="sofa, sectional, chair, bed, desk, ...")
    p.add_argument(
        "--dims",
        nargs=3,
        type=float,
        metavar=("W", "D", "H"),
        help="measured width (side-to-side, seen from the front), depth, height in meters. "
        "Strongly recommended: without it the generator's (non-metric) scale is kept",
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--up", choices=["y", "z"], default="y", help="source up axis (default: y)")
    p.add_argument("--id", dest="asset_id", help="asset id (default: mesh file stem)")
    p.add_argument("--label", help="human-readable label (default: from category)")
    p.add_argument("--dims-from", choices=["roomplan", "listing", "manual"], default="manual")
    p.add_argument(
        "--geometry-from",
        choices=["sam3d", "object_capture", "catalog", "manual", "box"],
        default="sam3d",
    )
    p.add_argument("--mount", choices=[m.value for m in Mount])
    p.add_argument(
        "--front",
        choices=["A", "B", "C", "D"],
        help="confirmed front from the candidate renders of a previous run (sets "
        "front_confirmed=true)",
    )
    p.add_argument("--mismatch-tol", type=float, default=0.15)
    p.add_argument("--max-footprint-vertices", type=int, default=32)
    p.add_argument("--footprint-tol", type=float, default=0.015, help="meters")
    p.add_argument("--no-clearance", action="store_true", help="emit no clearance zones")
    p.add_argument("--no-render", action="store_true", help="skip the front-candidate PNGs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dims = None
    if args.dims is not None:
        if min(args.dims) <= 0:
            print("error: --dims must be positive meters", file=sys.stderr)
            return 2
        dims = Dimensions(width=args.dims[0], depth=args.dims[1], height=args.dims[2])
    options = NormalizeOptions(
        up=args.up,
        mismatch_tolerance=args.mismatch_tol,
        front_override=args.front,
        clearance=not args.no_clearance,
        footprint=FootprintOptions(
            tolerance=args.footprint_tol, max_vertices=args.max_footprint_vertices
        ),
    )
    try:
        result = normalize(
            args.mesh,
            category=args.category,
            dims=dims,
            asset_id=args.asset_id,
            label=args.label,
            dimensions_from=args.dims_from,
            geometry_from=args.geometry_from,
            mount=Mount(args.mount) if args.mount else None,
            options=options,
        )
    except (MeshLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = write_outputs(result, args.out, render=not args.no_render)

    rep = result.report
    for w in rep.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    d = result.asset.dimensions
    print(f"asset   {paths['asset']}")
    print(f"mesh    {paths['glb']}")
    print(f"report  {paths['report']}")
    if "front_sheet" in paths:
        print(f"front   {paths['front_sheet']}")
    print(
        f"dims    {d.width:.3f} x {d.depth:.3f} x {d.height:.3f} m "
        f"(from {rep.fit.dimensions_from}); aspect mismatch {rep.fit.aspect_mismatch:.1%}"
        + (" FLAGGED" if rep.fit.flagged else "")
    )
    print(
        f"front   {rep.front.chosen} (yaw {rep.front.chosen_yaw_deg} deg, "
        f"{rep.front.method}, confidence {rep.front.confidence:.2f})"
    )
    print(
        f"footprint {rep.footprint.vertex_count} vertices, {rep.footprint.area_m2:.3f} m2 "
        f"({rep.footprint.bbox_fill:.0%} of bbox), zones: {', '.join(rep.clearance_zones) or '-'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
