"""Front orientation: four yaw hypotheses and a category-aware heuristic default.

Input is a mesh already aligned so its footprint rectangle is axis-aligned and centred
(``canonical``). Hypothesis ``k`` (labels A-D) turns the mesh by ``k * 90`` degrees CCW,
which brings the aligned frame's side ``+Y, +X, -Y, -X`` (for k = 0..3) to the front
(local +Y, ADR-0001).

Per hypothesis we measure, from surface samples:

- ``low``: mean top-surface height of the half *away* from the candidate side minus that
  of the half *toward* it, over the height. Positive when the candidate side is lower.
  This is the seat side of a sofa/chair (the backrest is at the back) and the foot of a
  bed.
- ``recess``: looking at the candidate side from outside, how far in (as a fraction of
  the depth) the first surface is, averaged over the side's silhouette. High for open
  shelves, knee space, a seat in front of a backrest; ~0 for a flat back panel.
- ``long``: length of the candidate side over the longest side (1 for a long side).
- ``dims``: ``-|log(w/d) - log(W/D)|`` where w, d are the mesh's width/depth under this
  hypothesis and W, D the measured ones. 0 when proportions agree. Only used when
  dimensions are known.

The score is the weighted sum, with weights per category (``categories.FrontWeights``).
Confidence is ``clip(margin / 0.25, 0, 1)``, where the margin is the gap between the best
and second-best scores. For tables, TVs and anything else with a 180-degree symmetry it is
~0 by construction, which is correct: geometry can't tell those fronts apart, and a
VLM/user pick from the renders is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .categories import FrontWeights
from .frames import homogeneous, yaw_matrix

LABELS = ("A", "B", "C", "D")
SOURCE_SIDES = ("+Y", "+X", "-Y", "-X")
CONFIDENCE_MARGIN = 0.25
GRID = 24


@dataclass
class FrontHypothesis:
    label: str
    quarter_turns: int
    yaw_deg: int
    source_side: str
    score: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class FrontDecision:
    hypotheses: list[FrontHypothesis]
    chosen: int  # index into hypotheses == quarter_turns
    confidence: float
    method: str  # "heuristic" | "override"
    rule: str
    # Geometry alone (without the measured W/D term) confidently prefers a side 90 degrees
    # away from the chosen one: the measured width and depth may be swapped.
    geometry_choice: int | None = None
    dims_conflict: bool = False

    @property
    def chosen_hypothesis(self) -> FrontHypothesis:
        return self.hypotheses[self.chosen]


def candidate_direction(k: int) -> np.ndarray:
    """Aligned-frame direction of the side that hypothesis ``k`` turns to the front."""
    return yaw_matrix(-k)[:2, :2] @ np.array([0.0, 1.0])


def apply_front(mesh: trimesh.Trimesh, k: int) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(homogeneous(yaw_matrix(k)))
    return out


def _grid_max(a: np.ndarray, b: np.ndarray, value: np.ndarray, n: int = GRID):
    """Max of ``value`` over an n x n grid in (a, b). Returns (grid, a_centres, b_centres)."""
    a0, a1 = a.min(), a.max()
    b0, b1 = b.min(), b.max()
    ia = np.clip(((a - a0) / max(a1 - a0, 1e-9) * n).astype(int), 0, n - 1)
    ib = np.clip(((b - b0) / max(b1 - b0, 1e-9) * n).astype(int), 0, n - 1)
    grid = np.full((n, n), -np.inf)
    np.maximum.at(grid, (ia, ib), value)
    ac = a0 + (np.arange(n) + 0.5) * (a1 - a0) / n
    bc = b0 + (np.arange(n) + 0.5) * (b1 - b0) / n
    return grid, ac, bc


def side_features(
    points: np.ndarray, k: int, dims_wd: tuple[float, float] | None
) -> dict[str, float]:
    d = candidate_direction(k)
    perp = np.array([-d[1], d[0]])
    t = points[:, :2] @ d  # toward the candidate side
    u = points[:, :2] @ perp  # along the candidate side
    z = points[:, 2]
    lt = float(np.ptp(t)) or 1e-9
    lu = float(np.ptp(u)) or 1e-9
    h = float(np.ptp(z)) or 1e-9

    # Top-surface height map over the footprint.
    top, uc, tc = _grid_max(u, t, z)
    occupied = np.isfinite(top)
    t_mid = 0.5 * (t.min() + t.max())
    near = occupied & (tc[None, :] > t_mid)
    far = occupied & (tc[None, :] < t_mid)
    low = 0.0
    if near.any() and far.any():
        low = float((top[far].mean() - top[near].mean()) / h)

    # Depth map seen from outside the candidate side.
    first, _, _ = _grid_max(u, z, t)
    occ = np.isfinite(first)
    recess = float(np.mean((t.max() - first[occ]) / lt)) if occ.any() else 0.0

    feats = {"low": low, "recess": recess, "long": lu / max(lu, lt)}
    if dims_wd is not None:
        w, dd = dims_wd
        feats["dims"] = -abs(math.log(lu / lt) - math.log(w / dd))
    return feats


def choose_front(
    mesh: trimesh.Trimesh,
    weights: FrontWeights,
    rule: str,
    dims_wd: tuple[float, float] | None = None,
    override: int | None = None,
    samples: int = 20000,
    seed: int = 0,
) -> FrontDecision:
    """Score the four hypotheses on an aligned, centred mesh and pick one."""
    points, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    points = np.vstack([points, np.asarray(mesh.vertices)])
    hyps = []
    for k in range(4):
        f = side_features(points, k, dims_wd)
        score = (
            weights.low * f["low"]
            + weights.recess * f["recess"]
            + weights.long * f["long"]
            + weights.dims * f.get("dims", 0.0)
        )
        hyps.append(
            FrontHypothesis(
                label=LABELS[k],
                quarter_turns=k,
                yaw_deg=90 * k,
                source_side=SOURCE_SIDES[k],
                score=round(float(score), 6),
                features={key: round(val, 6) for key, val in f.items()},
            )
        )
    order = sorted(range(4), key=lambda i: hyps[i].score, reverse=True)
    margin = hyps[order[0]].score - hyps[order[1]].score
    confidence = float(np.clip(margin / CONFIDENCE_MARGIN, 0.0, 1.0))

    # Same ranking without the dims term. If geometry is confident on its own and puts the
    # front on the other axis, the dims term overruled it: flag a possible W/D swap.
    geo = [h.score - weights.dims * h.features.get("dims", 0.0) for h in hyps]
    geo_order = sorted(range(4), key=lambda i: geo[i], reverse=True)
    geo_conf = (geo[geo_order[0]] - geo[geo_order[1]]) / CONFIDENCE_MARGIN
    chosen = override % 4 if override is not None else order[0]
    conflict = bool(
        dims_wd is not None
        and (weights.low > 0 or weights.recess > 0)
        and geo_conf >= 0.5
        and geo_order[0] % 2 != chosen % 2
    )
    if override is not None:
        return FrontDecision(hyps, chosen, 1.0, "override", rule, geo_order[0], conflict)
    return FrontDecision(
        hyps, chosen, round(confidence, 4), "heuristic", rule, geo_order[0], conflict
    )
