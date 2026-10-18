"""Composition of the stages (ADR-0008): scan -> problem -> constraints -> layouts.

Stage packages import ``roomlayout.schema`` only; this module is the one place that wires
them together, through their public APIs.

    from roomlayout.pipeline import problem_from_roomplan, plan
    problem, ingest = problem_from_roomplan("Room.json")
    built, result = plan(problem, "Sofa facing the TV, keep the window clear")

CLI (the user's own-room test)::

    python -m roomlayout.pipeline Room.json --out out/myroom \\
        [--request "Sofa facing the TV"] [--room 0] [--seconds 20]

The request is parsed by the live VLM (needs ``ANTHROPIC_API_KEY``); without it, only
the rule library applies. Writes ``problem.json``, ``ingest_report.json``,
``result.json``, ``current.png`` (the scanned arrangement) and ``layout_<k>.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roomlayout.assets import with_default_zones
from roomlayout.constraints import BuildResult, build_constraints
from roomlayout.ingest.roomplan import IngestError, IngestResult, load
from roomlayout.schema import LayoutProblem, SolveResult
from roomlayout.solver import solve


def problem_from_roomplan(
    path: str | Path, *, room: int | None = None, problem_id: str | None = None
) -> tuple[LayoutProblem, IngestResult]:
    """RoomPlan JSON -> problem with the categories' default clearance zones."""
    ingest = load(path, room=room, problem_id=problem_id)
    data = ingest.problem.model_dump(mode="json")
    data["assets"] = [with_default_zones(a).model_dump(mode="json") for a in ingest.problem.assets]
    return LayoutProblem.model_validate(data), ingest


def plan(
    problem: LayoutProblem,
    text: str | None = None,
    *,
    client: Any = None,
    k: int = 3,
    seed: int = 0,
    seconds: float = 20.0,
) -> tuple[BuildResult, SolveResult]:
    """User request + rule library -> constraints -> ``k`` diverse layouts."""
    built = build_constraints(problem, text, client=client)
    return built, solve(built.problem, k=k, seed=seed, time_budget_s=seconds)


def _current_layout(problem: LayoutProblem):
    from roomlayout.schema import Layout, Placement
    from roomlayout.solver import evaluate

    results, feasible, cost = evaluate(problem)
    placements = [Placement(asset_id=a, pose=p) for a, p in problem.initial.items()]
    return Layout(id="current", placements=placements, cost=cost, feasible=feasible,
                  results=results)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RoomPlan scan + request -> layouts")
    p.add_argument("capture", type=Path, help="RoomPlan CapturedRoom JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--request", help="what the user wants, in their words (needs an API key)")
    p.add_argument("--room", type=int, help="room index in a merged CapturedStructure")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    try:
        problem, ingest = problem_from_roomplan(a.capture, room=a.room)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ingest_report.json").write_text(json.dumps(ingest.report(), indent=1))
    for w in ingest.warnings:
        print(f"scan warning: {w}", file=sys.stderr)

    built, result = plan(problem, a.request, k=a.k, seed=a.seed, seconds=a.seconds)
    (a.out / "problem.json").write_text(built.problem.model_dump_json(indent=1))
    (a.out / "result.json").write_text(result.model_dump_json(indent=1))
    if built.parse is not None:
        for r in built.parse.restatements:
            print(f"understood: {r}")
        for u in built.parse.unexpressed:
            print(f"not expressible: {u.text} ({u.reason})")
    for n in built.notes:
        print(f"note: {n}")

    from roomlayout.solver.render import render_layout

    render_layout(built.problem, _current_layout(built.problem), a.out / "current.png")
    for i, lay in enumerate(result.layouts, 1):
        render_layout(built.problem, lay, a.out / f"layout_{i}.png")
        print(f"layout {i}: {'feasible' if lay.feasible else 'INFEASIBLE'}, cost {lay.cost:.3f}")
    for note in result.notes:
        print(f"solver: {note}")
    print(f"wrote {a.out}")
    return 0 if result.layouts and result.layouts[0].feasible else 1


if __name__ == "__main__":
    sys.exit(main())
