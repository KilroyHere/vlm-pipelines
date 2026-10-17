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

    # Automatic (the default path): text prompt -> OWLv2 box -> SAM 2 mask, no clicking.
    # Needs torch + transformers (a separate venv is fine); runs on CPU (slow) or CUDA.
    python make_mask.py auto data/sofa_01 data/desk_13      # or: auto data/*/
    python make_mask.py auto data/tv_stand_18 --prompt "tv stand,media console"
    python make_mask.py auto data/sofa_01 --box 120 340 1650 1180   # skip detection

    # QA: overlay every mask on its image -> data/<item>/mask_check.png
    python make_mask.py check data

Only needs numpy, Pillow and matplotlib (plus sam2 for the sam2 mode and torch +
transformers for auto; those are imported inside their subcommands).
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


def _read_meta(item: Path) -> dict:
    path = item / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _write_meta(item: Path, meta: dict) -> None:
    (item / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _save_mask(item: Path, mask: np.ndarray, source: str, extra: dict | None = None) -> None:
    """Write mask.png and record how it was made (score.py attributes failures by it)."""
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(item / "mask.png")
    print(f"wrote {item / 'mask.png'} ({int((mask > 0).sum())} px)")
    meta = _read_meta(item)
    if meta:
        meta["mask_source"] = source
        if extra is not None:
            meta["auto_mask"] = extra
        else:
            meta.pop("auto_mask", None)  # a hand fix replaces the automatic mask
        _write_meta(item, meta)


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


def prepare_item(photo: Path | str, item: Path | str, meta: dict) -> tuple[int, int]:
    """Upright (EXIF), size-capped image.jpg + meta.json. Returns the image size."""
    item = Path(item)
    item.mkdir(parents=True, exist_ok=True)
    with Image.open(photo) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(item / "image.jpg", quality=95)  # no EXIF orientation left to confuse anyone
    for stale in ("mask.png", "mask_check.png"):  # a new photo invalidates the old mask
        (item / stale).unlink(missing_ok=True)
    _write_meta(item, meta)
    return im.size


def cmd_prepare(a) -> None:
    meta = build_meta(a)  # validate before writing anything
    size = prepare_item(a.photo, a.item, meta)
    print(f"prepared {a.item}: image {size}, meta {meta}")


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
    _save_mask(item, np.array(mask), "manual_polygon")


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
    _save_mask(item, np.asarray(masks[0]) > 0, "manual_sam2_box")


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
    _save_mask(item, alpha > 127, "manual_alpha")


def write_overlay(item: Path) -> float | None:
    """mask_check.png: mask in red over the dimmed image, plus the detector box (green) for
    automatic masks. Returns the mask's image fraction (None if missing / mismatched)."""
    if not (item / "mask.png").exists():
        print(f"{item.name}: no mask")
        return None
    rgb = _load_rgb(item).astype(float)
    with Image.open(item / "mask.png") as m:
        mask = np.array(m)
    mask = (mask[..., -1] if mask.ndim == 3 else mask) > 0
    if mask.shape != rgb.shape[:2]:
        print(f"{item.name}: SIZE MISMATCH mask {mask.shape} image {rgb.shape[:2]}")
        return None
    out = rgb * 0.35
    out[mask] = rgb[mask] * 0.6 + np.array([255, 0, 0]) * 0.4
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    box = (_read_meta(item).get("auto_mask") or {}).get("detector_box")
    if box:
        ImageDraw.Draw(img).rectangle(box, outline=(0, 255, 0), width=max(2, img.width // 400))
    img.save(item / "mask_check.png")
    print(f"{item.name}: {mask.mean():.1%} of the image -> mask_check.png")
    return float(mask.mean())


def cmd_check(a) -> None:
    root = Path(a.data)
    for item in sorted(p for p in root.iterdir() if p.is_dir()):
        write_overlay(item)


def cmd_auto(a) -> None:
    """Text prompt -> OWLv2 box -> SAM 2 mask for each item folder (models loaded once)."""
    import automask as am  # torch/transformers load lazily inside the model classes

    items = [Path(p) for p in a.items]
    for item in items:
        _image_path(item)  # fail early on a missing image
        if not _read_meta(item).get("category") and not a.prompt and not a.box:
            raise SystemExit(f"{item}: meta.json needs 'category' (or pass --prompt / --box)")
    if a.box and len(items) > 1:
        raise SystemExit("--box applies to one item")
    detector = None if a.box else am.OwlDetector(a.detector, a.device)
    segmenter = am.Sam2Segmenter(a.sam_model, a.device)
    failed = []
    for item in items:
        meta = _read_meta(item)
        rgb = _load_rgb(item)
        image = Image.fromarray(rgb)
        size = image.size
        targets, distractors = am.queries_for(
            meta.get("category", ""), a.prompt or meta.get("prompt"), meta.get("subtype")
        )
        info: dict = {
            "prompt": targets,
            "distractors": distractors,
            "sam_model": segmenter.name,
            "device": segmenter.device,
        }
        if a.box:
            det = am.Detection(tuple(a.box), 1.0, "manual box")
            info.update(box_source="manual", detector=None)
        else:
            dets = detector.detect(image, targets, distractors)
            det = am.select_box(dets, size, min_score=a.min_score, rel_score=a.rel_score)
            kept = am.filter_boxes(dets, size, a.min_score)
            info.update(
                box_source="detector",
                detector=detector.name,
                selection_rule=f"largest of boxes with score >= {a.rel_score} x best "
                f"(min {a.min_score}, area {am.MIN_FRAC:.0%}-{am.MAX_FRAC:.0%}, NMS)",
                candidates=[d.to_json() for d in kept[:5]],
            )
            if det is None:
                print(f"{item.name}: no detection for {targets}; use --prompt or --box")
                failed.append(item.name)
                continue
        mask, sam_score = segmenter.segment(image, det.box)
        info.update(
            detector_box=[round(v, 1) for v in det.box],
            detector_score=round(det.score, 4),
            detector_label=det.label,
            box_area_frac=round(am.box_fraction(det, size), 4),
            sam_score=round(sam_score, 4),
            mask_area_frac=round(float(mask.mean()), 4),
        )
        _save_mask(item, mask, "auto", info)
        print(
            f"{item.name}: box {info['detector_box']} '{det.label}' score {det.score:.3f}, "
            f"SAM {sam_score:.3f}, mask {mask.mean():.1%}"
        )
        write_overlay(item)
    if failed:
        raise SystemExit(f"no mask for: {', '.join(failed)}")


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
    s = sub.add_parser("auto", help="automatic mask: OWLv2 text->box, SAM 2 box->mask")
    s.add_argument("items", nargs="+", help="item folders (data/<item>)")
    s.add_argument("--prompt", help="comma-separated detector queries (default: from category)")
    s.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                   help="skip detection and prompt SAM 2 with this box (one item)")
    s.add_argument("--device", default="auto", help="auto | cpu | cuda")
    s.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    s.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    s.add_argument("--min-score", type=float, default=0.10)
    s.add_argument("--rel-score", type=float, default=0.6)
    a = p.parse_args(argv)
    cmds = {"prepare": cmd_prepare, "polygon": cmd_polygon, "sam2": cmd_sam2,
            "alpha": cmd_alpha, "auto": cmd_auto, "check": cmd_check}
    cmds[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
