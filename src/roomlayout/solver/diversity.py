"""Layout distance and greedy diverse top-k selection.

Distance between two layouts = mean over *movable* assets of
``|delta position| + YAW_WEIGHT * |delta yaw| / 90deg`` (meters), where identical,
interchangeable assets (same category, footprint, zones, mount and height, e.g. a set
of dining chairs) are matched to each other with the cheapest assignment, so swapping
two identical chairs is not a "different" layout.

Selection: candidates sorted best-first (feasible before infeasible, then cost); a
candidate is kept if its distance to every kept layout is >= MIN_LAYOUT_DISTANCE.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from .geometry import HALF_PI, wrap_angle
from .model import Model, Pose

YAW_WEIGHT = 0.5  # meters-equivalent of a 90 deg turn
MIN_LAYOUT_DISTANCE = 0.3  # meters (mean per movable asset)
_MAX_EXACT_GROUP = 4  # exhaustive matching up to 4! permutations; greedy beyond


def _pose_distance(p: Pose, q: Pose) -> float:
    return (
        math.hypot(p[0] - q[0], p[1] - q[1]) + YAW_WEIGHT * abs(wrap_angle(p[2] - q[2])) / HALF_PI
    )


def interchangeable_groups(model: Model) -> list[list[int]]:
    by_sig: dict[tuple, list[int]] = {}
    for i in model.movable:
        by_sig.setdefault(model.templates[i].signature, []).append(i)
    return list(by_sig.values())


def layout_distance(a: Sequence[Pose], b: Sequence[Pose], groups: list[list[int]]) -> float:
    total, count = 0.0, 0
    for g in groups:
        count += len(g)
        if len(g) == 1:
            total += _pose_distance(a[g[0]], b[g[0]])
            continue
        cost = [[_pose_distance(a[i], b[j]) for j in g] for i in g]
        if len(g) <= _MAX_EXACT_GROUP:
            total += min(
                sum(cost[r][c] for r, c in enumerate(perm))
                for perm in itertools.permutations(range(len(g)))
            )
        else:  # greedy matching
            free = set(range(len(g)))
            for r in range(len(g)):
                c = min(free, key=lambda c, r=r: cost[r][c])
                free.remove(c)
                total += cost[r][c]
    return total / count if count else 0.0


def select_diverse(
    candidates: Sequence[tuple[Sequence[Pose], float, bool]],
    k: int,
    groups: list[list[int]],
    min_distance: float = MIN_LAYOUT_DISTANCE,
) -> list[int]:
    """Greedy diverse top-k. ``candidates`` = (poses, cost, feasible); returns indices."""
    order = sorted(range(len(candidates)), key=lambda i: (not candidates[i][2], candidates[i][1]))
    kept: list[int] = []
    for i in order:
        if len(kept) >= k:
            break
        if all(
            layout_distance(candidates[i][0], candidates[j][0], groups) >= min_distance
            for j in kept
        ):
            kept.append(i)
    return kept
