"""Builders for solver tests: tiny problems and an evaluator returning results by id."""

from __future__ import annotations

import math

from roomlayout.schema import LayoutProblem, Pose2D
from roomlayout.solver import evaluate

PI = math.pi
# yaw at which the asset's front faces +y / -y / +x / -x
FACE_N, FACE_S, FACE_E, FACE_W = 0.0, PI, -PI / 2, PI / 2


def rect(x0, y0, x1, y1):
    return {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def box_room(w=4.0, d=3.0, openings=(), obstacles=()):
    """Rectangle [0,w]x[0,d]; walls w_s (y=0), w_e (x=w), w_n (y=d), w_w (x=0), CCW."""
    return {
        "id": "room",
        "floor": {"points": [[0, 0], [w, 0], [w, d], [0, d]]},
        "ceiling_height": 2.5,
        "walls": [
            {"id": "w_s", "start": [0, 0], "end": [w, 0], "height": 2.5},
            {"id": "w_e", "start": [w, 0], "end": [w, d], "height": 2.5},
            {"id": "w_n", "start": [w, d], "end": [0, d], "height": 2.5},
            {"id": "w_w", "start": [0, d], "end": [0, 0], "height": 2.5},
        ],
        "openings": list(openings),
        "obstacles": list(obstacles),
    }


def box(id, w, d, h=0.8, zones=(), **kw):
    a = {
        "id": id,
        "category": kw.pop("category", "thing"),
        "dimensions": {"width": w, "depth": d, "height": h},
    }
    if zones:
        a["clearance_zones"] = list(zones)
    a.update(kw)
    return a


def front_zone(w, d, depth, purpose="access", name="front"):
    return {"name": name, "purpose": purpose, "polygon": rect(-w / 2, d / 2, w / 2, d / 2 + depth)}


def make_problem(assets, constraints=(), room=None, initial=None, id="t"):
    data = {
        "id": id,
        "room": room or box_room(),
        "assets": list(assets),
        "constraints": list(constraints),
    }
    if initial:
        data["initial"] = initial
    return LayoutProblem.model_validate(data)


def results(problem, poses):
    """Evaluate hand-placed poses {id: (x, y, yaw)} -> ({constraint_id: result}, feasible)."""
    ps = {k: Pose2D(x=v[0], y=v[1], yaw=v[2]) for k, v in poses.items()}
    res, feasible, _ = evaluate(problem, ps)
    return {r.constraint_id: r for r in res}, feasible
