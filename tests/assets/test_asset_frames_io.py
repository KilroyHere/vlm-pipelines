import numpy as np
import pytest
import trimesh

from roomlayout.assets import MeshLoadError, export_glb, load_mesh
from roomlayout.assets.frames import GLTF_TO_OURS, OURS_TO_GLTF, source_to_ours, yaw_matrix

from .synthetic import l_sectional


def test_gltf_conversion_is_a_proper_rotation_and_its_own_inverse():
    assert np.linalg.det(OURS_TO_GLTF) == pytest.approx(1.0)
    assert np.allclose(OURS_TO_GLTF @ OURS_TO_GLTF, np.eye(3))
    assert np.allclose(GLTF_TO_OURS @ OURS_TO_GLTF, np.eye(3))


def test_gltf_conversion_maps_up_and_front_per_spec():
    # ours: +Z up, +Y front, +X right  ->  glTF: +Y up, +Z front, -X right
    assert np.allclose(OURS_TO_GLTF @ [0, 0, 1], [0, 1, 0])
    assert np.allclose(OURS_TO_GLTF @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(OURS_TO_GLTF @ [1, 0, 0], [-1, 0, 0])


def test_yaw_matrix_quarter_turns_are_exact():
    assert np.array_equal(yaw_matrix(1) @ [1.0, 0, 0], [0, 1.0, 0])
    assert np.array_equal(yaw_matrix(4), np.eye(3))


def test_unknown_up_axis_rejected():
    with pytest.raises(ValueError):
        source_to_ours("x")


def test_glb_round_trip_preserves_geometry_in_our_frame(tmp_path):
    mesh = l_sectional().mesh
    path = export_glb(mesh, tmp_path / "sofa.glb")
    back = load_mesh(path, up="y")
    assert len(back.faces) == len(mesh.faces)
    # GLB stores float32; compare per vertex (order is preserved by trimesh).
    assert np.allclose(back.vertices, mesh.vertices, atol=1e-6)
    assert np.allclose(back.bounds, mesh.bounds, atol=1e-6)


def test_exported_glb_is_y_up_and_faces_plus_z(tmp_path):
    mesh = l_sectional().mesh  # 2.8 wide (X), 1.6 deep (Y), 0.85 tall (Z); chaise at +Y
    raw = trimesh.load(export_glb(mesh, tmp_path / "s.glb"), force="mesh", process=False)
    ext = raw.extents
    assert ext == pytest.approx([2.8, 0.85, 1.6], abs=1e-6)  # glTF: height on Y
    assert raw.bounds[0][1] == pytest.approx(0.0, abs=1e-6)  # bottom on the floor
    # The chaise sticks out to the front: in glTF, front is +Z.
    chaise = raw.vertices[raw.vertices[:, 2] > 0.5]
    assert len(chaise) > 0


def test_scene_with_node_transforms_is_flattened(tmp_path):
    a = trimesh.creation.box((1, 1, 1))
    b = trimesh.creation.box((1, 1, 1))
    scene = trimesh.Scene()
    scene.add_geometry(a, node_name="a")
    scene.add_geometry(
        b, node_name="b", transform=trimesh.transformations.translation_matrix([3, 0, 0])
    )
    path = tmp_path / "scene.glb"
    scene.export(path)
    mesh = load_mesh(path, up="z")
    assert len(mesh.faces) == 24
    assert mesh.bounds[1][0] == pytest.approx(3.5, abs=1e-6)


@pytest.mark.parametrize("ext", ["obj", "ply", "stl"])
def test_other_formats_load(tmp_path, ext):
    mesh = trimesh.creation.box((1.0, 2.0, 3.0))
    path = tmp_path / f"box.{ext}"
    mesh.export(path)
    loaded = load_mesh(path, up="z")
    assert loaded.extents == pytest.approx([1, 2, 3], abs=1e-5)
    as_y = load_mesh(path, up="y")
    assert as_y.extents == pytest.approx([1, 3, 2], abs=1e-5)


def test_missing_or_bad_file(tmp_path):
    with pytest.raises(MeshLoadError):
        load_mesh(tmp_path / "nope.glb")
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb")
    with pytest.raises(MeshLoadError):
        load_mesh(bad)
