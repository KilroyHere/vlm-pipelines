"""Exit criterion M1: every feasible example problem gets a feasible layout within 10 s
(4 workers). Slow (~1-2 min): deselect with ``-m 'not slow'``."""

from pathlib import Path

import pytest

from roomlayout.schema import LayoutProblem
from roomlayout.solver import solve_detailed

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "problems").glob("*.json"))
BUDGET_S = 10.0


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_suite_problem(path):
    problem = LayoutProblem.model_validate_json(path.read_text())
    out = solve_detailed(problem, k=10, seed=0, time_budget_s=BUDGET_S)
    layouts = out.result.layouts
    assert layouts
    if "infeasible" in problem.id:
        assert not any(lay.feasible for lay in layouts)
        assert any("no feasible layout" in n for n in out.result.notes)
    else:
        assert layouts[0].feasible, out.result.notes
        assert out.diagnostics.time_to_first_feasible_s is not None
        assert out.diagnostics.time_to_first_feasible_s <= BUDGET_S
    assert out.diagnostics.seconds <= BUDGET_S + 1.0  # small allowance for pool start/stop
