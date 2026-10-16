"""Layout solver (WS-A, milestone M1): simulated annealing over 2D footprints (ADR-0003).

Public API::

    from roomlayout.solver import solve
    result = solve(problem, k=10, seed=0, time_budget_s=10.0)

CLI: ``python -m roomlayout.solver PROBLEM.json --out DIR``. Design: docs/solver.md.
"""

from .solve import SolveDiagnostics, SolveOutcome, evaluate, solve, solve_detailed

__all__ = ["SolveDiagnostics", "SolveOutcome", "evaluate", "solve", "solve_detailed"]
