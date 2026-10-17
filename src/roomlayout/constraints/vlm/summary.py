"""Compact, deterministic text rendering of a LayoutProblem for the prompt.

The model needs enough to ground "the couch", "my desk", "the big window", "the wall
with the door": asset ids with type/label/size, openings with the wall they sit on,
walls with a compass name and length, obstacles, and constraints already in the
problem. Compass names come from our frame (+Y = north, ADR-0001); the app labels the
plan the same way.

Same problem in -> same bytes out (no dict/set iteration order, fixed number format).
"""

from __future__ import annotations

import math

from roomlayout.schema import Asset, LayoutProblem, Opening, Wall, referenced_assets

_COMPASS = ("east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast")


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _pt(p: tuple[float, float]) -> str:
    return f"({_fmt(p[0])}, {_fmt(p[1])})"


def _compass(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return _COMPASS[int(((angle + 22.5) % 360.0) // 45.0)]


def interior_normal(wall: Wall) -> tuple[float, float]:
    """Unit normal pointing into the room (interior is left of start -> end)."""
    dx, dy = wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]
    n = math.hypot(dx, dy)
    return (-dy / n, dx / n)


def wall_side(wall: Wall) -> str:
    """Which side of the room the wall is on: the wall facing north is the south wall."""
    nx, ny = interior_normal(wall)
    return _compass(-nx, -ny)


def wall_names(problem: LayoutProblem) -> dict[str, str]:
    """Human name per wall id, e.g. "south wall" (numbered when several share a side)."""
    sides = {w.id: wall_side(w) for w in problem.room.walls}
    counts: dict[str, int] = {}
    for s in sides.values():
        counts[s] = counts.get(s, 0) + 1
    seen: dict[str, int] = {}
    names: dict[str, str] = {}
    for w in problem.room.walls:
        s = sides[w.id]
        if counts[s] > 1:
            seen[s] = seen.get(s, 0) + 1
            names[w.id] = f"{s} wall #{seen[s]}"
        else:
            names[w.id] = f"{s} wall"
    return names


def _opposite_walls(problem: LayoutProblem, wall: Wall) -> list[str]:
    nx, ny = interior_normal(wall)
    out = []
    for other in problem.room.walls:
        if other.id == wall.id:
            continue
        ox, oy = interior_normal(other)
        if nx * ox + ny * oy < -0.99:
            out.append(other.id)
    return out


def _opening_centre(wall: Wall, o: Opening) -> tuple[float, float]:
    t = o.offset / wall.length
    return (
        wall.start[0] + t * (wall.end[0] - wall.start[0]),
        wall.start[1] + t * (wall.end[1] - wall.start[1]),
    )


def _asset_line(a: Asset, problem: LayoutProblem) -> str:
    kind = a.category if a.subtype is None else f"{a.category}/{a.subtype}"
    d = a.dimensions
    parts = [
        f"- {a.id}: {kind}",
        f'label "{a.label}"' if a.label else "no label",
        f"{_fmt(d.width)} W x {_fmt(d.depth)} D x {_fmt(d.height)} H m",
    ]
    if a.mount.value == "wall":
        parts.append(f"wall-mounted at {_fmt(a.elevation)} m")
    elif a.elevation > 0:
        parts.append(f"raised {_fmt(a.elevation)} m")
    if not a.movable:
        parts.append("immovable (built-in)")
    elif a.id in problem.initial:
        parts.append("has a current position (can be kept where it is)")
    return ", ".join(parts)


def _constraint_line(c) -> str:  # noqa: ANN001 - any schema Constraint
    args: list[str] = list(referenced_assets(c))
    for key in ("wall_id", "opening_id"):
        v = getattr(c, key, None)
        if v is not None:
            args.append(v)
    for key in ("max_distance", "min_distance", "distance", "min_width", "max_angle_deg"):
        v = getattr(c, key, None)
        if v is not None:
            args.append(f"{key}={v:g}")
    strength = "hard" if c.hard else "soft"
    return f"- {c.type}({', '.join(args)}) {strength}, from {c.origin}"


def summarize_problem(problem: LayoutProblem) -> str:
    """Render ``problem`` as the compact text block the model grounds references in."""
    room = problem.room
    xs = [p[0] for p in room.floor.points]
    ys = [p[1] for p in room.floor.points]
    names = wall_names(problem)
    openings_by_wall: dict[str, list[str]] = {}
    for o in room.openings:
        openings_by_wall.setdefault(o.wall_id, []).append(o.id)

    lines = [
        f"Room {room.id}: floor area {_fmt(room.floor.area)} m2, bounding box "
        f"{_fmt(max(xs) - min(xs))} m (west-east) x {_fmt(max(ys) - min(ys))} m (south-north), "
        f"{len(room.floor.points)} corners, ceiling {_fmt(room.ceiling_height)} m.",
        "Coordinates in meters, x = east, y = north.",
        "",
        "Walls:",
    ]
    walls = {w.id: w for w in room.walls}
    for w in room.walls:
        extra = []
        if w.id in openings_by_wall:
            extra.append("has " + ", ".join(openings_by_wall[w.id]))
        opp = _opposite_walls(problem, w)
        if opp:
            extra.append("opposite " + ", ".join(opp))
        tail = f"; {'; '.join(extra)}" if extra else ""
        lines.append(
            f"- {w.id}: {names[w.id]}, {_fmt(w.length)} m, from {_pt(w.start)} to {_pt(w.end)}"
            f"{tail}"
        )

    lines += ["", "Openings:"]
    if not room.openings:
        lines.append("- none")
    for o in room.openings:
        w = walls[o.wall_id]
        desc = f"- {o.id}: {o.type.value} in {o.wall_id} ({names[o.wall_id]}), "
        desc += f"{_fmt(o.width)} m wide, centre at {_pt(_opening_centre(w, o))}"
        if o.type.value == "window":
            desc += f", sill {_fmt(o.sill_height)} m, top {_fmt(o.sill_height + o.height)} m"
        lines.append(desc)

    if room.obstacles:
        lines += ["", "Fixed obstacles (not furniture; cannot be constrained):"]
        for ob in room.obstacles:
            ox = sum(p[0] for p in ob.footprint.points) / len(ob.footprint.points)
            oy = sum(p[1] for p in ob.footprint.points) / len(ob.footprint.points)
            lines.append(f'- {ob.id}: "{ob.label}" near {_pt((ox, oy))}')

    lines += ["", "Furniture (asset ids):"]
    lines += [_asset_line(a, problem) for a in problem.assets]

    lines += ["", "Constraints already in the problem:"]
    if problem.constraints:
        lines += [_constraint_line(c) for c in problem.constraints]
    else:
        lines.append("- none")
    return "\n".join(lines)
