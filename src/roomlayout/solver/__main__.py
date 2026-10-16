"""CLI: ``python -m roomlayout.solver PROBLEM.json --out DIR [--k 10 --seed 0 --budget 10]``.

Writes ``DIR/result.json`` (a SolveResult), ``DIR/diagnostics.json`` (search statistics
not yet in the contract) and ``DIR/layout_NN.png`` per layout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from roomlayout.schema import LayoutProblem

from .model import Model
from .solve import solve_detailed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roomlayout.solver", description=__doc__)
    ap.add_argument("problem", type=Path, help="LayoutProblem JSON")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--k", type=int, default=10, help="number of layouts (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=10.0, help="time budget in seconds")
    ap.add_argument("--workers", type=int, default=None, help="processes (default min(4, cpus))")
    ap.add_argument("--no-png", action="store_true", help="skip rendering")
    args = ap.parse_args(argv)

    problem = LayoutProblem.model_validate_json(args.problem.read_text())
    outcome = solve_detailed(
        problem, k=args.k, seed=args.seed, time_budget_s=args.budget, workers=args.workers
    )
    result = outcome.result
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    (args.out / "diagnostics.json").write_text(
        json.dumps(dataclasses.asdict(outcome.diagnostics), indent=2) + "\n"
    )
    if not args.no_png:
        from .render import render_layout

        model = Model(problem)
        for i, layout in enumerate(result.layouts):
            render_layout(problem, layout, args.out / f"layout_{i:02d}.png", model)

    d = outcome.diagnostics
    n_feas = sum(lay.feasible for lay in result.layouts)
    ttff = "-" if d.time_to_first_feasible_s is None else f"{d.time_to_first_feasible_s:.2f}s"
    print(
        f"{problem.id}: {len(result.layouts)} layouts ({n_feas} feasible), "
        f"best cost {result.layouts[0].cost if result.layouts else float('nan'):.3f}, "
        f"first feasible {ttff}, {d.restarts} restarts, {d.seconds:.1f}s -> {args.out}"
    )
    for note in result.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
