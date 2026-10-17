#!/usr/bin/env python
"""Tables and contact sheets from ``run_e2e.py`` reports (for RESULTS.md).

python summarize.py out/sam3d out/boxes --sheet out/sheet_<id>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def table(run_dir: Path) -> str:
    rows = [
        "| room | assets (SAM 3D / box) | real: user reqs met | real: checks failed | "
        "layout 1: feasible | layout 1: user reqs met | layout 1: checks failed | "
        "layouts passing all checks | mean move vs real (m) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rep in sorted(run_dir.glob("*/report.json")):
        r = json.loads(rep.read_text())
        geo = [a["geometry"] for a in r["assets"].values()]
        top = r["layouts"][0] if r["layouts"] else None
        passing = sum(not lay["checks_failed"] for lay in r["layouts"])
        rows.append(
            f"| {r['video_id']} | {geo.count('sam3d')} / {geo.count('box')} "
            f"| {r['real']['user_constraints_satisfied']} | {len(r['real']['checks_failed'])} "
            f"| {top['feasible'] if top else '-'} "
            f"| {top['user_constraints_satisfied'] if top else '-'} "
            f"| {len(top['checks_failed']) if top else '-'} | {passing}/{len(r['layouts'])} "
            f"| {top['vs_real']['mean_moved_m'] if top else '-'} |"
        )
    return "\n".join(rows)


def sheet(run_dir: Path, vid: str, out: Path) -> None:
    """real (2D) | layout 1 (2D) on top, real (3D) | layout 1 (3D) below."""
    d = run_dir / vid
    tiles = [d / "real.png", d / "layout_1.png", d / "scene3d_real.png", d / "scene3d_layout_1.png"]
    ims = [Image.open(p).convert("RGB") for p in tiles if p.exists()]
    ims = [
        im.crop((0, 0, int(im.width * 0.62), im.height)) if i < 2 else im
        for i, im in enumerate(ims)
    ]
    w = 700
    ims = [im.resize((w, int(w * im.height / im.width))) for im in ims]
    h0 = max(im.height for im in ims[:2])
    h1 = max((im.height for im in ims[2:]), default=0)
    canvas = Image.new("RGB", (2 * w, h0 + h1), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, ((i % 2) * w, 0 if i < 2 else h0))
    canvas.save(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--sheet", help="also write contact sheets; '<id>' is replaced")
    a = p.parse_args()
    for run in a.runs:
        print(f"### {run}\n\n{table(run)}\n")
        if a.sheet:
            for rep in sorted(run.glob("*/report.json")):
                vid = rep.parent.name
                sheet(run, vid, Path(a.sheet.replace("<id>", f"{run.name}_{vid}")))


if __name__ == "__main__":
    main()
