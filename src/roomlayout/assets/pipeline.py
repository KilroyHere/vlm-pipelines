"""Raw generator mesh -> solver-ready ``Asset`` (+ GLB, report and front renders).

    load -> Z-up -> debris removal -> principal-axis alignment -> front choice
         -> fit to measured dims -> footprint -> clearance zones -> Asset

See docs/assets.md for the rationale behind each step.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import trimesh

from roomlayout.schema import Asset, AssetSource, Dimensions, MeshRef, Mount

from . import categories
from .canonical import align_principal_axes, extents, level, recenter
from .cleanup import remove_debris
from .fit import FitReport, fit_to_dimensions, up_check
from .footprint import FootprintOptions, extract_footprint
from .frames import UpAxis
from .front import LABELS, FrontDecision, apply_front, choose_front
from .meshio import export_glb, load_mesh, to_ours
from .report import (
    DebrisSection,
    FitSection,
    FootprintSection,
    FrontHypothesisOut,
    FrontSection,
    NormalizationReport,
    UpCheckSection,
)
from .zones import build_zones

DimensionsFrom = Literal["roomplan", "listing", "manual"]
GeometryFrom = Literal["box", "sam3d", "catalog", "object_capture", "manual"]

UNMEASURED_WARNING = (
    "DIMENSIONS NOT MEASURED: the asset keeps the generator's scale, which is NOT metric "
    "for single-image generators like SAM 3D. Layouts built on it can be confidently wrong. "
    "Provide --dims W D H (RoomPlan box, product listing or tape measure)."
)
LOW_CONFIDENCE = 0.5


@dataclass
class NormalizeOptions:
    up: UpAxis = "y"
    mismatch_tolerance: float = 0.15
    debris_area_fraction: float = 0.005
    debris_size_fraction: float = 0.10
    max_tilt_deg: float = 8.0
    footprint: FootprintOptions = field(default_factory=FootprintOptions)
    front_override: str | None = None  # "A".."D": a human/VLM pick, sets front_confirmed
    clearance: bool = True


@dataclass
class NormalizeResult:
    asset: Asset
    mesh: trimesh.Trimesh  # final, in the asset's local frame
    aligned_mesh: trimesh.Trimesh  # before the front choice; the renders show this
    decision: FrontDecision
    report: NormalizationReport


def default_asset_id(path: str | Path) -> str:
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "asset"


def normalize(
    source: str | Path | trimesh.Trimesh,
    *,
    category: str,
    dims: Dimensions | None,
    asset_id: str | None = None,
    label: str | None = None,
    dimensions_from: DimensionsFrom = "manual",
    geometry_from: GeometryFrom = "sam3d",
    mount: Mount | None = None,
    options: NormalizeOptions | None = None,
) -> NormalizeResult:
    """Normalize a raw mesh (path, or an in-memory mesh in the source convention)."""
    opt = options or NormalizeOptions()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        timings[name] = round(now - t0, 4)
        t0 = now

    if isinstance(source, trimesh.Trimesh):
        raw = to_ours(source, opt.up)
        source_name = "<in-memory>"
    else:
        raw = load_mesh(source, up=opt.up)
        source_name = str(source)
    asset_id = asset_id or (default_asset_id(source) if source_name != "<in-memory>" else "asset")
    lap("load")

    profile = categories.lookup(category)
    if profile is None:
        profile = categories.profile_for(category)
        warnings.append(
            f"unknown category {category!r}: no clearance zones, generic front rule. "
            f"RoomPlan categories: {', '.join(categories.ROOMPLAN_CATEGORIES)}"
        )

    debris = remove_debris(raw, opt.debris_area_fraction, opt.debris_size_fraction)
    if debris.removed_area_fraction > 0.05:
        warnings.append(
            f"removed {debris.removed_components} floating components "
            f"({debris.removed_area_fraction:.1%} of surface area); check the mesh"
        )
    lap("debris")

    leveled, tilt = level(debris.mesh, opt.max_tilt_deg)
    if math.isnan(tilt):
        warnings.append(
            f"no support plane within {opt.max_tilt_deg:.0f} deg of horizontal; mesh not "
            "levelled. Is the up axis right?"
        )
    aligned, principal_yaw = align_principal_axes(leveled)
    aligned = recenter(aligned)
    lap("align")

    override = None
    if opt.front_override is not None:
        letter = opt.front_override.strip().upper()
        if letter not in LABELS:
            raise ValueError(f"front override must be one of {LABELS}, got {letter!r}")
        override = LABELS.index(letter)
    dims_wd = (dims.width, dims.depth) if dims is not None else None
    decision = choose_front(aligned, profile.front, profile.front_rule, dims_wd, override)
    oriented = recenter(apply_front(aligned, decision.chosen))
    if decision.method == "heuristic" and decision.confidence < LOW_CONFIDENCE:
        warnings.append(
            f"front guess {decision.chosen_hypothesis.label} has low confidence "
            f"({decision.confidence:.2f}); confirm it from the front-candidate renders"
        )
    if decision.dims_conflict and decision.method == "heuristic":
        warnings.append(
            f"measured width/depth may be swapped: the geometry points to front "
            f"{LABELS[decision.geometry_choice]}, but W x D = {dims.width} x {dims.depth} only "
            f"fits front {decision.chosen_hypothesis.label}. Width is side-to-side seen from "
            "the front; check the measurement or confirm the front with --front"
        )
    lap("front")

    oriented_ext = extents(oriented)
    if dims is not None:
        fitted, fit = fit_to_dimensions(oriented, dims, opt.mismatch_tolerance)
        dims_source: str = dimensions_from
        if fit.flagged:
            warnings.append(
                f"generator proportions disagree with the measured box: aspect mismatch "
                f"{fit.aspect_mismatch:.0%} > {fit.tolerance:.0%} (scale x/y/z = "
                + "/".join(f"{s:.3f}" for s in fit.scale)
                + "). The asset uses the measured size; the shape may look distorted"
            )
        uc = up_check(oriented_ext, dims, opt.mismatch_tolerance)
        if uc.suspicious:
            warnings.append(
                f"up axis looks wrong: treating the mesh's {uc.best_alternative_axis} axis as "
                f"vertical would cut the mismatch from {uc.current_mismatch:.0%} to "
                f"{uc.best_alternative_mismatch:.0%}. Check --up or the generator output"
            )
    else:
        fitted = oriented
        dims = Dimensions(
            width=float(oriented_ext[0]),
            depth=float(oriented_ext[1]),
            height=float(oriented_ext[2]),
        )
        dims_source = "mesh"
        fit = FitReport(
            mesh_extents=tuple(float(v) for v in oriented_ext),
            target=tuple(float(v) for v in oriented_ext),
            scale=(1.0, 1.0, 1.0),
            aspect_mismatch=0.0,
            per_axis_mismatch=(0.0, 0.0, 0.0),
            tolerance=opt.mismatch_tolerance,
            flagged=False,
            uniform_scale=1.0,
            uniform_residual_m=(0.0, 0.0, 0.0),
        )
        uc = None
        warnings.insert(0, UNMEASURED_WARNING)
    lap("fit")

    fp = extract_footprint(fitted, dims.width, dims.depth, opt.footprint)
    if fp.rule == "convex_hull_of_disjoint_parts":
        warnings.append(
            "footprint had several disjoint parts; using their convex hull (conservative)"
        )
    lap("footprint")

    zones = build_zones(fp.polygon, profile.zones) if opt.clearance else []
    lap("zones")

    asset = Asset(
        id=asset_id,
        category=profile.roomplan,
        label=label if label is not None else profile.label,
        dimensions=dims,
        footprint=fp.polygon,
        clearance_zones=zones,
        mount=mount if mount is not None else profile.mount,
        movable=profile.movable,
        mesh=None,
        source=AssetSource(dimensions_from=dims_source, geometry_from=geometry_from),
        front_confirmed=decision.method == "override",
    )

    chosen = decision.chosen_hypothesis
    report = NormalizationReport(
        asset_id=asset_id,
        source_mesh=source_name,
        category_input=category,
        category=profile.roomplan,
        fine_type=profile.key,
        input_faces=len(raw.faces),
        output_faces=len(fitted.faces),
        principal_yaw_deg=round(math.degrees(principal_yaw), 4),
        tilt_corrected_deg=None if math.isnan(tilt) else round(tilt, 4),
        debris=DebrisSection(
            components=debris.components,
            removed_components=debris.removed_components,
            removed_faces=debris.removed_faces,
            removed_area_fraction=round(debris.removed_area_fraction, 6),
        ),
        front=FrontSection(
            method=decision.method,
            rule=decision.rule,
            chosen=chosen.label,
            chosen_yaw_deg=chosen.yaw_deg,
            confidence=decision.confidence,
            confirmed=decision.method == "override",
            geometry_choice=(
                LABELS[decision.geometry_choice] if decision.geometry_choice is not None else None
            ),
            dims_conflict=decision.dims_conflict,
            hypotheses=[
                FrontHypothesisOut(
                    label=h.label,
                    yaw_deg=h.yaw_deg,
                    source_side=h.source_side,
                    score=h.score,
                    features=h.features,
                )
                for h in decision.hypotheses
            ],
        ),
        fit=FitSection(dimensions_from=dims_source, **_rounded(fit.__dict__)),
        up_check=UpCheckSection(
            source_up=opt.up,
            current_mismatch=round(uc.current_mismatch, 6) if uc else 0.0,
            suspicious=uc.suspicious if uc else False,
            best_alternative_axis=uc.best_alternative_axis if uc else None,
            best_alternative_mismatch=(
                round(uc.best_alternative_mismatch, 6)
                if uc and uc.best_alternative_mismatch is not None
                else None
            ),
        ),
        footprint=FootprintSection(
            vertex_count=fp.vertex_count,
            area_m2=round(fp.polygon.area, 6),
            bbox_fill=round(fp.bbox_fill, 6),
            rule=fp.rule,
            parts_found=fp.parts_found,
            parts_dropped=fp.parts_dropped,
            raster_resolution_m=round(fp.resolution, 6),
            simplify_tolerance_m=round(fp.tolerance_used, 6),
            max_deviation_m=round(fp.hausdorff, 6),
        ),
        clearance_zones=[z.name for z in zones],
        warnings=warnings,
        timings_s=timings,
    )
    return NormalizeResult(asset, fitted, aligned, decision, report)


def _rounded(d: dict) -> dict:
    def r(v):
        if isinstance(v, float):
            return round(v, 6)
        if isinstance(v, tuple):
            return tuple(r(x) for x in v)
        return v

    return {k: r(v) for k, v in d.items()}


def write_outputs(result: NormalizeResult, out_dir: str | Path, render: bool = True) -> dict:
    """Write ``<id>.glb``, ``<id>.asset.json``, ``<id>.report.json`` and front renders.

    ``Asset.mesh.uri`` is relative to ``out_dir``. Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aid = result.asset.id
    t = time.perf_counter()

    glb = export_glb(result.mesh, out_dir / f"{aid}.glb")
    result.asset = result.asset.model_copy(update={"mesh": MeshRef(uri=glb.name, format="glb")})
    paths: dict[str, Path] = {"glb": glb}

    if render:
        from .render import render_front_candidates  # matplotlib import only when needed

        renders = render_front_candidates(
            result.aligned_mesh, result.decision, out_dir, aid, category=result.report.fine_type
        )
        for h in result.report.front.hypotheses:
            h.render = renders[h.label].name
        result.report.front.contact_sheet = renders["sheet"].name
        paths.update({f"front_{k}": v for k, v in renders.items()})
    result.report.timings_s["write"] = round(time.perf_counter() - t, 4)

    asset_path = out_dir / f"{aid}.asset.json"
    report_path = out_dir / f"{aid}.report.json"
    paths["asset"] = asset_path
    paths["report"] = report_path
    result.report.outputs = {k: p.name for k, p in paths.items()}
    asset_path.write_text(result.asset.model_dump_json(indent=2) + "\n")
    report_path.write_text(result.report.model_dump_json(indent=2) + "\n")
    return paths
