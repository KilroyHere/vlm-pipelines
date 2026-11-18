"""Viewer-only mesh cleanup (roomlayout.assets.display)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from roomlayout.assets.display import DisplayOptions, display_cleanup


def noisy_blob(seed: int = 0) -> trimesh.Trimesh:
    """A dense, bumpy box-ish blob standing on z=0 with a ragged base and a hole."""
    rng = np.random.default_rng(seed)
    m = trimesh.creation.icosphere(subdivisions=5)  # 20k faces
    m.apply_scale([0.5, 0.4, 0.45])
    m.vertices += rng.normal(0, 0.004, m.vertices.shape)
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    # Ragged base: the lowest vertices sit a few mm above/below each other.
    low = m.vertices[:, 2] < 0.004
    m.vertices[low, 2] = rng.uniform(0, 0.004, low.sum())
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    # A hole: drop a few faces near the top.
    top = np.argsort(-m.triangles_center[:, 2])[:6]
    m.update_faces(np.setdiff1d(np.arange(len(m.faces)), top))
    m.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(m.vertices), 1))
    return m


def roughness(m: trimesh.Trimesh) -> float:
    """Mean angle between adjacent face normals (radians): lower = smoother."""
    return float(np.mean(m.face_adjacency_angles))


def test_bbox_kept_base_flat_hole_filled_smoother():
    m = noisy_blob()
    out, rep = display_cleanup(m)
    assert out.bounds == pytest.approx(m.bounds, abs=1e-9)
    base = out.vertices[:, 2] < 0.45 * 0.01 + 1e-12
    assert rep.base_vertices == base.sum() > 10
    assert np.all(out.vertices[base, 2] == pytest.approx(0.0))
    assert rep.holes_filled
    assert roughness(out) < 0.7 * roughness(m)
    assert out.vertex_normals.shape == out.vertices.shape
    assert len(out.visual.vertex_colors) == len(out.vertices)
    assert rep.max_shift_m < 0.03  # light touch: a few cm at most on a 0.9 m piece


def test_low_poly_meshes_are_not_smoothed():
    box = trimesh.creation.box([1.0, 0.5, 0.8])
    box.vertices[:, 2] -= box.vertices[:, 2].min()
    out, _ = display_cleanup(box)
    assert sorted(map(tuple, np.round(out.vertices, 9))) == sorted(
        map(tuple, np.round(box.vertices, 9))
    )


def test_smoothing_can_be_turned_off():
    m = noisy_blob(1)
    out, _ = display_cleanup(m, DisplayOptions(smooth_iterations=0, fill_holes=False))
    assert roughness(out) == pytest.approx(roughness(m), rel=0.05)
