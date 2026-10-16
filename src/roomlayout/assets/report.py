"""Sidecar report written next to each normalized asset (``<id>.report.json``).

This is WS-B's own format, not part of the shared contract. Keep it additive.
"""

from __future__ import annotations

from pydantic import BaseModel

REPORT_VERSION = "1"


class DebrisSection(BaseModel):
    components: int
    removed_components: int
    removed_faces: int
    removed_area_fraction: float


class FrontHypothesisOut(BaseModel):
    label: str
    yaw_deg: int
    source_side: str
    score: float
    features: dict[str, float]
    render: str | None = None


class FrontSection(BaseModel):
    method: str  # heuristic | override
    rule: str
    chosen: str
    chosen_yaw_deg: int
    confidence: float
    confirmed: bool
    geometry_choice: str | None = None  # best side ignoring the measured W/D
    dims_conflict: bool = False
    hypotheses: list[FrontHypothesisOut]
    contact_sheet: str | None = None


class FitSection(BaseModel):
    dimensions_from: str
    mesh_extents: tuple[float, float, float]
    target: tuple[float, float, float]
    scale: tuple[float, float, float]
    aspect_mismatch: float
    per_axis_mismatch: tuple[float, float, float]
    tolerance: float
    flagged: bool
    uniform_scale: float
    uniform_residual_m: tuple[float, float, float]


class UpCheckSection(BaseModel):
    source_up: str
    current_mismatch: float
    suspicious: bool
    best_alternative_axis: str | None = None
    best_alternative_mismatch: float | None = None


class FootprintSection(BaseModel):
    vertex_count: int
    area_m2: float
    bbox_fill: float
    rule: str
    parts_found: int
    parts_dropped: int
    raster_resolution_m: float
    simplify_tolerance_m: float
    max_deviation_m: float


class NormalizationReport(BaseModel):
    report_version: str = REPORT_VERSION
    asset_id: str
    source_mesh: str
    category_input: str
    category: str
    fine_type: str
    input_faces: int
    output_faces: int
    principal_yaw_deg: float
    tilt_corrected_deg: float | None  # None: no support plane found, not levelled
    debris: DebrisSection
    front: FrontSection
    fit: FitSection
    up_check: UpCheckSection
    footprint: FootprintSection
    clearance_zones: list[str]
    warnings: list[str]
    timings_s: dict[str, float]
    outputs: dict[str, str] = {}
