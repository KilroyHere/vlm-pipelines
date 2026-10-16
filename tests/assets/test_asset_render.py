"""Front-candidate renders: each side view is a true elevation of that side."""

import numpy as np
import pytest
from PIL import Image

from roomlayout.assets import normalize, write_outputs

from . import synthetic as S

TITLE_ROWS = 60  # per-side PNGs are 512 px; the title sits in the top ~40 px


def _object_aspect(png) -> float:
    """Width / height of the non-white pixel bounding box below the title."""
    with Image.open(png) as im:
        a = np.asarray(im.convert("L"))[TITLE_ROWS:]
    rows, cols = np.nonzero(a < 245)
    return (cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1)


@pytest.mark.parametrize(
    ("name", "category", "seed"),
    [("l_sectional", "sofa", 7), ("bookshelf", "bookshelf", 2), ("chair", "chair", 4)],
)
def test_side_views_are_elevations(tmp_path, name, category, seed):
    synth = S.ALL[name]()
    raw, _ = S.as_generator_output(synth, seed)
    result = normalize(raw, category=category, dims=synth.dims, asset_id=name)
    paths = write_outputs(result, tmp_path)
    # Renders show the aligned (pre-front) mesh; hypothesis A/C look along Y, B/D along X.
    ex, ey, ez = result.aligned_mesh.extents
    for k, label in enumerate("ABCD"):
        horizontal = ex if k % 2 == 0 else ey
        expected = horizontal / ez
        got = _object_aspect(paths[f"front_{label}"])
        assert got == pytest.approx(expected, rel=0.05), (label, got, expected)
    assert paths["front_sheet"].exists()
