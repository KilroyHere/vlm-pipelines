#!/usr/bin/env python
"""Minimal mask tooling for the SAM 3D spike (run on a laptop, before renting the GPU).

    # Prepare an item folder: EXIF-rotate the photo upright, cap its size, write meta.json
    python make_mask.py prepare PHOTO.jpg data/sofa_01 --category sofa --dims 2.1 0.92 0.83 \\
        --dims-provenance tape_measured --condition in_room occluded --notes "throw pillows"

    # Item from a web listing: record where the photo and the dims came from
    python make_mask.py prepare listing.jpg data/sectional_03 --category sectional \\
        --dims 2.9 1.7 0.85 --dims-provenance seller_reported \\
        --source-url "https://offerup.com/item/detail/..."

    # Draw the mask: click polygon corners around the object, Enter to finish.
    # Several polygons are allowed (union). Backspace undoes a point, Esc clears.
    python make_mask.py polygon data/sofa_01

    # Or prompt SAM 2 with a box (needs `pip install sam2` and torch; UNTESTED here)
    python make_mask.py sam2 data/sofa_01 --box 120 340 1650 1180

    # Or take the alpha channel of a same-size cutout PNG
    python make_mask.py alpha data/sofa_01 cutout.png

    # QA: overlay every mask on its image -> data/<item>/mask_check.png
    python make_mask.py check data

Only needs numpy, Pillow and matplotlib (plus sam2 for the sam2 mode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw, ImageOps

MAX_SIDE = 2048  # SAM 3D downsamples the pointmap above ~1200 px anyway


def _image_path(item: Path) -> Path:
    for name in ("image.jpg", "image.jpeg", "image.png"):
        if (item / name).exists():
            return item / name
    raise SystemExit(f"no image.jpg/png in {item}")


def _load_rgb(item: Path) -> np.ndarray:
    with Image.open(_image_path(item)) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def _save_mask(item: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(item / "mask.png")
    print(f"wrote {item / 'mask.png'} ({int((mask > 0).sum())} px)")


def strip_tracking(url: str) -> str:
    """Drop utm_* (and similar) tracking parameters from a URL."""
    parts = urlsplit(url.strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


def build_meta(a) -> dict:
    meta = {
        "category": a.category,
        "condition": a.condition or [],
        "notes": a.notes or "",
    }
    if a.source_url:
        meta["source_url"] = strip_tracking(a.source_url)
    provenance = a.dims_provenance
    if a.dims:
        meta["dims"] = dict(zip(("width", "depth", "height"), a.dims, strict=True))
        if provenance is None:
            # Items come from tape measures and web listings alike: say which.
            raise SystemExit("--dims needs --dims-provenance (where did W/D/H come from?)")
    if provenance is not None:
        meta["dims_provenance"] = provenance
    if provenance in ("manufacturer", "seller_reported") and not a.source_url:
        print("warning: listing dims without --source-url; record where the photo came from")
    return meta


def cmd_prepare(a) -> None:
    meta = build_meta(a)  # validate before writing anything
    item = Path(a.item)
    item.mkdir(parents=True, exist_ok=True)
    with Image.open(a.photo) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(item / "image.jpg", quality=95)  # no EXIF orientation left to confuse anyone
    (item / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"prepared {item}: image {im.size}, meta {meta}")


def cmd_polygon(a) -> None:
    import matplotlib.pyplot as plt

    item = Path(a.item)
    rgb = _load_rgb(item)
    polys: list[list[tuple[float, float]]] = [[]]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)
    ax.set_title("click corners; Enter = close polygon (Enter twice = save); Backspace = undo")
    (line,) = ax.plot([], [], "r.-")
    done = {"save": False}

    def redraw():
        pts = polys[-1]
        line.set_data([p[0] for p in pts], [p[1] for p in pts])
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is ax and ev.button == 1:
            polys[-1].append((ev.xdata, ev.ydata))
            redraw()

    def on_key(ev):
        if ev.key == "backspace" and polys[-1]:
            polys[-1].pop()
        elif ev.key == "escape":
            polys[-1].clear()
        elif ev.key == "enter":
            if len(polys[-1]) >= 3:
                pts = polys[-1] + [polys[-1][0]]
                ax.fill([p[0] for p in pts], [p[1] for p in pts], alpha=0.3, color="red")
                polys.append([])
            elif not polys[-1] and len(polys) > 1:
                done["save"] = True
                plt.close(fig)
                return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    closed = [p for p in polys if len(p) >= 3]
    if not closed:
        raise SystemExit("no polygon drawn; nothing saved")
    mask = Image.new("L", (rgb.shape[1], rgb.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for p in closed:
        draw.polygon(p, fill=255)
    _save_mask(item, np.array(mask))


def cmd_sam2(a) -> None:
    """Box-prompted SAM 2 (API as in the facebookresearch/sam2 README). UNTESTED here."""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    item = Path(a.item)
    rgb = _load_rgb(item)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained(a.model, device=device)
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            box=np.array(a.box, dtype=float), multimask_output=False
        )
    print(f"SAM 2 score {float(np.ravel(scores)[0]):.3f}")
    _save_mask(item, np.asarray(masks[0]) > 0)


def cmd_alpha(a) -> None:
    item = Path(a.item)
    rgb = _load_rgb(item)
    with Image.open(a.cutout) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGBA":
            raise SystemExit("cutout has no alpha channel")
        alpha = np.array(im)[..., 3]
    if alpha.shape != rgb.shape[:2]:
        raise SystemExit(
            f"cutout size {alpha.shape[::-1]} != image size {rgb.shape[1::-1]}; "
            "cropped cutouts (e.g. iOS 'Copy Subject') can't be registered, use polygon"
        )
    _save_mask(item, alpha > 127)


def cmd_check(a) -> None:
    root = Path(a.data)
    for item in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (item / "mask.png").exists():
            print(f"{item.name}: no mask")
            continue
        rgb = _load_rgb(item).astype(float)
        with Image.open(item / "mask.png") as m:
            mask = np.array(m)
        mask = (mask[..., -1] if mask.ndim == 3 else mask) > 0
        if mask.shape != rgb.shape[:2]:
            print(f"{item.name}: SIZE MISMATCH mask {mask.shape} image {rgb.shape[:2]}")
            continue
        out = rgb * 0.35
        out[mask] = rgb[mask] * 0.6 + np.array([255, 0, 0]) * 0.4
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(item / "mask_check.png")
        print(f"{item.name}: {mask.mean():.1%} of the image -> mask_check.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="mask tooling for the SAM 3D spike")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare", help="create data/<item>/ with an upright image + meta.json")
    s.add_argument("photo")
    s.add_argument("item")
    s.add_argument("--category", required=True)
    s.add_argument("--dims", nargs=3, type=float, metavar=("W", "D", "H"))
    s.add_argument(
        "--dims-provenance",
        choices=["tape_measured", "manufacturer", "seller_reported", "estimated"],
        help="where W/D/H came from (required with --dims; informational provenance)",
    )
    s.add_argument("--source-url", help="where the photo came from (tracking params stripped)")
    s.add_argument("--condition", nargs="*", help="e.g. product in_room occluded low_light clutter")
    s.add_argument("--notes")
    s = sub.add_parser("polygon", help="draw the mask by clicking polygon corners")
    s.add_argument("item")
    s = sub.add_parser("sam2", help="box-prompted SAM 2 (needs sam2 + torch)")
    s.add_argument("item")
    s.add_argument("--box", nargs=4, type=float, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    s.add_argument("--model", default="facebook/sam2-hiera-large")
    s = sub.add_parser("alpha", help="mask from the alpha channel of a same-size cutout")
    s.add_argument("item")
    s.add_argument("cutout")
    s = sub.add_parser("check", help="write mask_check.png overlays for every item")
    s.add_argument("data")
    a = p.parse_args(argv)
    {"prepare": cmd_prepare, "polygon": cmd_polygon, "sam2": cmd_sam2, "alpha": cmd_alpha}.get(
        a.cmd, cmd_check
    )(a)


if __name__ == "__main__":
    main(sys.argv[1:])
