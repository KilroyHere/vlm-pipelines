"""spikes/img3d_eval helpers that run on CPU (white-background masks)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes" / "img3d_eval"))
from white_mask import keep_components, white_background_mask  # noqa: E402


def test_white_inside_the_object_stays_object_and_specks_go():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[20:80, 20:80] = (120, 80, 40)  # wooden cabinet
    img[40:60, 40:60] = 255  # white drawer front, enclosed by wood
    img[5, 5] = (0, 0, 0)  # a speck on the background
    obj = keep_components(white_background_mask(img), largest_only=False)
    assert obj[50, 50] and obj[25, 25] and not obj[5, 5] and not obj[95, 95]


def test_largest_only_keeps_one_of_two_chairs():
    img = np.full((100, 200, 3), 255, np.uint8)
    img[20:90, 10:80] = 30
    img[30:90, 120:180] = 30
    obj = keep_components(white_background_mask(img), largest_only=True)
    assert obj[50, 40] and not obj[50, 150]
