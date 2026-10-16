"""Top-down PNG of a layout (matplotlib, Agg backend).

Shows the floor, walls (drawn outside the floor by their thickness), openings (doors
with swing arcs, windows, passages), obstacles, keep-clear rectangles, footprints
labelled with a front arrow, clearance zones (hatched), the walkway free space when the
problem has a walkway constraint, and the violated constraints in a side panel.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from roomlayout.schema import KeepClear, Layout, LayoutProblem, OpeningType, WalkwayWidth  # noqa: E402

from . import rules  # noqa: E402
from .model import Model  # noqa: E402

_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def _poly_patches(geom):
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        yield list(p.exterior.coords)


def render_layout(
    problem: LayoutProblem, layout: Layout, path: str | Path, model: Model | None = None
) -> Path:
    model = model or Model(problem)
    room = model.room
    poses = {p.asset_id: (p.pose.x, p.pose.y, p.pose.yaw) for p in layout.placements}
    geoms = [model.geom(t.idx, poses[t.id]) for t in model.templates]

    fig = plt.figure(figsize=(11, 7.5), dpi=110)
    ax = fig.add_axes((0.07, 0.08, 0.58, 0.83))
    side = fig.add_axes((0.67, 0.05, 0.31, 0.86))
    side.axis("off")

    # floor + walkway free space
    ax.add_patch(
        MplPolygon(list(room.floor.exterior.coords), closed=True, fc="#f6f3ee", ec="none", zorder=0)
    )
    walk = [c for c in problem.constraints if isinstance(c, WalkwayWidth)]
    if walk:
        free = rules.walkway(room, model.templates, geoms, walk[0].min_width).free
        for coords in _poly_patches(free):
            ax.add_patch(
                MplPolygon(coords, closed=True, fc="#d8eed8", ec="none", alpha=0.7, zorder=0.5)
            )

    # walls (thickness drawn outward), with gaps at doors/passages
    for w, wall in zip(room.walls, problem.room.walls, strict=True):
        th = wall.thickness
        cuts = sorted(
            (o.t0, o.t1) for o in w.openings if o.type in (OpeningType.door, OpeningType.opening)
        )
        pos = 0.0
        for s0, s1 in [*cuts, (w.length, w.length)]:
            if s0 > pos:
                pts = [w.point(pos, 0), w.point(s0, 0), w.point(s0, -th), w.point(pos, -th)]
                ax.add_patch(MplPolygon(pts, closed=True, fc="#444444", ec="#444444", zorder=3))
            pos = max(pos, s1)
        for o in w.openings:
            if o.type == OpeningType.window:
                pts = [w.point(o.t0, 0), w.point(o.t1, 0), w.point(o.t1, -th), w.point(o.t0, -th)]
                ax.add_patch(
                    MplPolygon(pts, closed=True, fc="#cfe8ff", ec="#3b82c4", lw=1.2, zorder=4)
                )
                mx, my = w.point((o.t0 + o.t1) / 2, -th / 2)
                ax.text(
                    mx,
                    my,
                    o.id,
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="#1d4f80",
                    zorder=5,
                    rotation=math.degrees(math.atan2(w.uy, w.ux)) % 180,
                )
            elif o.swing is not None:
                coords = list(o.swing.exterior.coords)
                ax.add_patch(
                    MplPolygon(coords, closed=True, fc="#fff3c4", ec="none", alpha=0.8, zorder=1)
                )
                arc = coords[1:-1]
                ax.plot(
                    [p[0] for p in arc],
                    [p[1] for p in arc],
                    color="#b8860b",
                    lw=1,
                    ls="--",
                    zorder=4,
                )
                ax.plot(
                    [coords[0][0], arc[-1][0]],
                    [coords[0][1], arc[-1][1]],
                    color="#8b5a00",
                    lw=2,
                    zorder=4,
                )  # open door leaf
            if o.type != OpeningType.window:
                mx, my = w.point((o.t0 + o.t1) / 2, -th - 0.12)
                ax.text(mx, my, o.id, fontsize=6, ha="center", va="center", color="#8b5a00")

    # keep-clear rectangles
    for c in problem.constraints:
        if isinstance(c, KeepClear):
            rect = room.keep_clear_rect(c.opening_id, c.distance)
            ax.add_patch(
                MplPolygon(
                    list(rect.exterior.coords),
                    closed=True,
                    fill=False,
                    ec="#c0392b",
                    ls=":",
                    lw=1,
                    zorder=2,
                )
            )

    # obstacles
    for ob in room.obstacles:
        ax.add_patch(
            MplPolygon(
                list(ob.poly.exterior.coords),
                closed=True,
                fc="#999999",
                ec="#333333",
                hatch="xx",
                zorder=3,
            )
        )
        c = ob.poly.centroid
        ax.text(
            c.x,
            c.y,
            ob.label,
            fontsize=6,
            ha="center",
            va="center",
            zorder=6,
            bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 0.5},
        )

    # assets
    for t, g in zip(model.templates, geoms, strict=True):
        color = _PALETTE[t.idx % len(_PALETTE)]
        for z in g.zones:
            ax.add_patch(
                MplPolygon(
                    list(z.exterior.coords),
                    closed=True,
                    fill=False,
                    ec=color,
                    hatch="///",
                    lw=0.6,
                    alpha=0.6,
                    zorder=2,
                )
            )
        ax.add_patch(
            MplPolygon(
                list(g.fp.exterior.coords),
                closed=True,
                fc=color,
                alpha=0.85,
                ec="black",
                lw=1,
                zorder=5,
            )
        )
        (b0x, b0y), (b1x, b1y) = g.back
        ax.plot([b0x, b1x], [b0y, b1y], color="black", lw=2.5, zorder=6)  # back edge
        depth = t.ymax - t.ymin
        alen = max(0.15, min(0.4, 0.45 * depth))
        ax.annotate(
            "",
            xy=(g.x + g.fx * alen, g.y + g.fy * alen),
            xytext=(g.x, g.y),
            arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.5},
            zorder=7,
        )
        ax.text(
            g.x,
            g.y - 0.0,
            t.id,
            fontsize=7,
            ha="center",
            va="top",
            zorder=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.3},
        )

    x0, y0, x1, y1 = room.bounds
    pad = 0.4
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    status = "feasible" if layout.feasible else "INFEASIBLE"
    fig.suptitle(
        f"{layout.id}  ·  {status}  ·  cost {layout.cost:.3f}",
        fontsize=12,
        color="black" if layout.feasible else "#b00000",
    )

    # side panel: violated constraints, then legend
    lines = []
    violated = [r for r in layout.results if not r.satisfied]
    if violated:
        lines.append(("Violated constraints:", "black", "bold"))
        for r in violated:
            kind = "HARD" if r.hard else "soft"
            lines.append(
                (
                    f"[{kind}] {r.constraint_id}  ({r.violation:.3g})",
                    "#b00000" if r.hard else "#a05a00",
                    "normal",
                )
            )
            if r.detail:
                for part in r.detail.split("; ")[:3]:
                    lines.append((f"    {part[:52]}", "#555555", "normal"))
    else:
        lines.append(("All constraints satisfied.", "#1a7f37", "bold"))
    lines.append(("", "black", "normal"))
    lines.append(("Legend:", "black", "bold"))
    for txt in (
        "filled box = footprint, white arrow = front,",
        "  thick black line = back edge",
        "hatched outline = clearance zone",
        "yellow quarter disc = door swing",
        "red dotted = keep-clear area",
        "green = walkway free space (centre locus",
        "  of a walker of the walkway width)",
        "grey cross-hatch = obstacle, blue = window",
    ):
        lines.append((txt, "#333333", "normal"))
    y = 1.0
    for txt, color, weight in lines:
        side.text(
            0.0,
            y,
            txt,
            fontsize=7.5,
            color=color,
            weight=weight,
            va="top",
            transform=side.transAxes,
            family="monospace",
        )
        y -= 0.032
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path
