"""End-to-end normalization on synthetic 'generator output' (random yaw/scale/offset,
SAM 3D's Y-up convention, floating debris)."""

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

from roomlayout.assets import NormalizeOptions, normalize
from roomlayout.schema import Dimensions

from . import synthetic as S

SEEDS = [0, 1, 2, 3, 4, 5]


def _run(synth, seed, **kw):
    raw, info = S.as_generator_output(synth, seed)
    return normalize(raw, category=synth.category, dims=synth.dims, asset_id=synth.name, **kw), info


def _max_vertex_distance(result_mesh, truth_mesh) -> float:
    return float(cKDTree(truth_mesh.vertices).query(result_mesh.vertices)[0].max())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", ["l_sectional", "chair", "bookshelf"])
def test_canonical_frame_and_front_recovered_exactly(name, seed):
    synth = S.ALL[name]()
    result, _ = _run(synth, seed)
    # Front heuristic right + fit to the true dims => the mesh lands on the ground truth.
    assert _max_vertex_distance(result.mesh, synth.mesh) < 1e-6
    assert result.report.front.confidence >= 0.8
    assert result.asset.front_confirmed is False


@pytest.mark.parametrize("seed", SEEDS)
def test_round_table_upright_and_fitted(seed):
    synth = S.round_table()
    result, _ = _run(synth, seed)
    lo, hi = result.mesh.bounds
    assert hi - lo == pytest.approx([1.2, 1.2, 0.75], abs=1e-3)
    # Upright: the 4 cm disc top is at the top, the pedestal is thin.
    top = result.mesh.vertices[result.mesh.vertices[:, 2] > 0.70]
    assert np.ptp(top[:, 0]) == pytest.approx(1.2, abs=2e-3)
    centres = result.mesh.triangles_center
    mid = centres[(centres[:, 2] > 0.2) & (centres[:, 2] < 0.6)]
    assert len(mid) > 0 and np.ptp(mid[:, 0]) < 0.2
    # Rotationally symmetric: geometry can't pick a front, and the report says so.
    assert result.report.front.confidence < 0.2
    assert any("low confidence" in w for w in result.report.warnings)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", list(S.ALL))
def test_dims_debris_and_footprint(name, seed):
    synth = S.ALL[name]()
    result, info = _run(synth, seed)
    a = result.asset
    d = synth.dims

    # Dimensions fitted within 1 mm (measured dims win).
    ext = result.mesh.extents
    assert ext == pytest.approx([d.width, d.depth, d.height], abs=1e-3)
    assert (a.dimensions.width, a.dimensions.depth, a.dimensions.height) == pytest.approx(
        (d.width, d.depth, d.height), abs=1e-9
    )
    lo, hi = result.mesh.bounds
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0, abs=1e-9)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(0, abs=1e-9)
    assert lo[2] == pytest.approx(0, abs=1e-9)

    # Debris gone.
    assert result.report.debris.removed_components == info["debris"]

    # Footprint: inside the bbox, capped, close to the true projection.
    fp = Polygon(a.footprint.points)
    bbox = box(-d.width / 2, -d.depth / 2, d.width / 2, d.depth / 2)
    assert fp.is_valid
    assert fp.difference(bbox).area < 1e-12
    assert all(abs(x) <= d.width / 2 and abs(y) <= d.depth / 2 for x, y in a.footprint.points)
    assert 3 <= len(a.footprint.points) <= 32
    assert a.footprint.area > 0  # CCW
    assert fp.area == pytest.approx(synth.footprint.area, rel=0.03)
    assert fp.symmetric_difference(synth.footprint).area / synth.footprint.area < 0.03

    # Aspect mismatch ~0 when the measured dims are the true ones (a 64-gon table at an
    # arbitrary yaw is a hair narrower than its diameter, hence not exactly 0).
    assert result.report.fit.aspect_mismatch < 0.005
    assert not result.report.fit.flagged


def test_l_sofa_keeps_its_l_shape():
    synth = S.l_sectional()
    result, _ = _run(synth, 11)
    fp = Polygon(result.asset.footprint.points)
    bbox_area = synth.dims.width * synth.dims.depth
    assert fp.area / bbox_area == pytest.approx(3.08 / 4.48, abs=0.02)
    assert len(result.asset.footprint.points) == 6
    # The notch in front of the main seats (left-front) is free.
    assert not fp.contains(box(-1.3, 0.2, 0.5, 0.75))
    assert fp.contains(box(0.65, 0.15, 1.35, 0.75))  # chaise


def test_chair_footprint_is_seat_outline_not_legs():
    synth = S.chair()
    result, _ = _run(synth, 3)
    assert result.report.footprint.rule == "rectangle"
    assert len(result.asset.footprint.points) == 4


def test_front_heuristic_puts_backrest_at_the_back():
    for name in ("l_sectional", "chair"):
        synth = S.ALL[name]()
        for seed in SEEDS:
            result, _ = _run(synth, seed)
            v = result.mesh.vertices
            high = v[v[:, 2] > 0.8 * synth.dims.height]
            assert high[:, 1].mean() < 0, (name, seed)  # backrest at -Y; front is +Y


def test_mismatch_flagged_when_proportions_disagree():
    synth = S.l_sectional()
    raw, _ = S.as_generator_output(synth, 2)
    wrong = Dimensions(width=2.8, depth=1.6 * 1.3, height=0.85)
    result = normalize(raw, category="sofa", dims=wrong)
    fit = result.report.fit
    assert fit.flagged
    assert fit.aspect_mismatch == pytest.approx(1.3 ** (2 / 3) - 1, rel=1e-3)
    # Per-axis factors: y stretched 1.3x relative to x and z.
    sx, sy, sz = fit.scale
    assert sy / sx == pytest.approx(1.3, rel=1e-6)
    assert sz / sx == pytest.approx(1.0, rel=1e-6)
    assert any("proportions disagree" in w for w in result.report.warnings)
    # Measured dims still win.
    assert result.mesh.extents == pytest.approx([2.8, 2.08, 0.85], abs=1e-3)


def test_mismatch_tolerance_is_configurable():
    synth = S.l_sectional()
    raw, _ = S.as_generator_output(synth, 2)
    dims = Dimensions(width=2.8, depth=1.6 * 1.1, height=0.85)  # ~6.6 % mismatch
    assert not normalize(raw, category="sofa", dims=dims).report.fit.flagged
    strict = NormalizeOptions(mismatch_tolerance=0.05)
    assert normalize(raw, category="sofa", dims=dims, options=strict).report.fit.flagged


def test_unknown_dimensions_keep_mesh_scale_and_warn_loudly():
    synth = S.chair()
    raw, info = S.as_generator_output(synth, 4)
    result = normalize(raw, category="chair", dims=None)
    a = result.asset
    assert a.source.dimensions_from == "mesh"
    assert result.report.fit.dimensions_from == "mesh"
    assert result.report.warnings[0].startswith("DIMENSIONS NOT MEASURED")
    s = info["scale"]
    assert (a.dimensions.width, a.dimensions.depth, a.dimensions.height) == pytest.approx(
        (0.48 * s, 0.52 * s, 0.9 * s), rel=1e-6
    )


def test_wrong_up_axis_is_detected():
    synth = S.bookshelf()
    raw, _ = S.as_generator_output(synth, 1)  # Y-up file...
    result = normalize(raw, category="bookshelf", dims=synth.dims, options=NormalizeOptions(up="z"))
    assert result.report.up_check.suspicious
    assert any("up axis looks wrong" in w for w in result.report.warnings)


def test_front_override_marks_confirmed():
    synth = S.chair()
    raw, _ = S.as_generator_output(synth, 0)
    auto = normalize(raw, category="chair", dims=synth.dims)
    other = "ABCD"[(auto.decision.chosen + 2) % 4]
    forced = normalize(
        raw, category="chair", dims=synth.dims, options=NormalizeOptions(front_override=other)
    )
    assert forced.asset.front_confirmed is True
    assert forced.report.front.method == "override"
    assert forced.report.front.chosen == other
    v = forced.mesh.vertices
    assert v[v[:, 2] > 0.8][:, 1].mean() > 0  # backrest now at the front, as instructed


def test_unknown_category_warns_and_has_no_zones():
    synth = S.chair()
    raw, _ = S.as_generator_output(synth, 0)
    result = normalize(raw, category="piano", dims=synth.dims)
    assert result.asset.category == "piano"
    assert result.asset.clearance_zones == []
    assert any("unknown category" in w for w in result.report.warnings)


def test_category_maps_to_roomplan_and_label():
    synth = S.l_sectional()
    raw, _ = S.as_generator_output(synth, 0)
    result = normalize(raw, category="Sectional", dims=synth.dims)
    assert result.asset.category == "sofa"
    assert result.asset.label == "sectional sofa"
    assert result.report.fine_type == "sectional"


def _noisy_tilted(synth, seed, noise=0.004, tilt_deg=2.5, max_edge=0.06):
    """Denser, noisy, slightly tilted copy, closer to what a generator emits."""
    m = synth.mesh
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge)
    rng = np.random.default_rng(seed)
    noisy = trimesh.Trimesh(v + rng.normal(0, noise, v.shape), f, process=False)
    axis = (np.cos(seed), np.sin(seed), 0.0)
    noisy.apply_transform(trimesh.transformations.rotation_matrix(np.radians(tilt_deg), axis))
    return S.Synthetic(synth.name, synth.category, noisy, synth.dims, synth.footprint)


def test_tilted_generator_output_is_levelled():
    synth = S.bookshelf()
    tilted = _noisy_tilted(synth, 0, noise=0.0, tilt_deg=3.0)
    raw, _ = S.as_generator_output(tilted, 0)
    result = normalize(raw, category="bookshelf", dims=synth.dims)
    assert result.report.tilt_corrected_deg == pytest.approx(3.0, abs=0.05)
    assert result.report.fit.aspect_mismatch < 0.01
    assert Polygon(result.asset.footprint.points).area == pytest.approx(0.315, rel=0.02)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["l_sectional", "chair"])
def test_front_heuristic_survives_noise_and_tilt(name, seed):
    synth = S.ALL[name]()
    raw, _ = S.as_generator_output(_noisy_tilted(synth, seed), seed)
    result = normalize(raw, category=synth.category, dims=synth.dims)
    v = result.mesh.vertices
    assert v[v[:, 2] > 0.8 * synth.dims.height][:, 1].mean() < 0
    assert Polygon(result.asset.footprint.points).area == pytest.approx(
        synth.footprint.area, rel=0.10
    )


def test_swapped_width_and_depth_is_flagged():
    synth = S.l_sectional()
    raw, _ = S.as_generator_output(synth, 3)
    swapped = Dimensions(width=1.6, depth=2.8, height=0.85)
    result = normalize(raw, category="sofa", dims=swapped)
    assert result.report.front.dims_conflict
    assert any("swapped" in w for w in result.report.warnings)
    ok = normalize(raw, category="sofa", dims=synth.dims)
    assert not ok.report.front.dims_conflict
    assert not any("swapped" in w for w in ok.report.warnings)
