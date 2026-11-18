"""Masks for catalogue photos on a white background (CPU, no model).

The background is the near-white region connected to the image border (flood fill), so
white parts *inside* the object (a white drawer front surrounded by wood) stay object.
Small specks are removed; with ``--largest`` only the biggest component is kept (a photo
showing two identical chairs). Writes ``mask.png`` next to ``image.jpg`` and a
``mask_check.png`` overlay.

    python spikes/img3d_eval/white_mask.py data/abo/eval/* [--largest chair_02] [--tol 18]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def white_background_mask(rgb: np.ndarray, tol: int = 18) -> np.ndarray:
    near_white = (255 - rgb.astype(int)).max(axis=2) <= tol
    labels, _ = ndimage.label(near_white)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, border[border > 0])
    obj = ~background
    obj = ndimage.binary_opening(obj, iterations=2)
    obj = ndimage.binary_fill_holes(obj)
    return obj


def keep_components(obj: np.ndarray, largest_only: bool, min_frac: float = 0.02) -> np.ndarray:
    labels, n = ndimage.label(obj)
    if n == 0:
        return obj
    sizes = ndimage.sum(obj, labels, range(1, n + 1))
    if largest_only:
        return labels == (int(np.argmax(sizes)) + 1)
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_frac * sizes.max()]
    return np.isin(labels, keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("items", nargs="+", type=Path)
    ap.add_argument("--largest", nargs="*", default=[], help="item ids: keep only the biggest part")
    ap.add_argument("--tol", type=int, default=18)
    args = ap.parse_args()
    for d in args.items:
        img = d / "image.jpg"
        rgb = np.asarray(Image.open(img).convert("RGB"))
        obj = keep_components(white_background_mask(rgb, args.tol), d.name in args.largest)
        Image.fromarray((obj * 255).astype(np.uint8)).save(d / "mask.png")
        over = rgb.copy()
        over[~obj] = (over[~obj] * 0.25 + np.array([255, 0, 120]) * 0.75).astype(np.uint8)
        Image.fromarray(over).save(d / "mask_check.png")
        print(f"{d.name}: object {obj.mean():.1%} of the image")


if __name__ == "__main__":
    main()
