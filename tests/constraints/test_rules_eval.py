"""M3a evaluation: "minimal" variants of the feasible suite problems (every constraint
stripped except ``fixed``) + the rule library solve to layouts that pass the automated
sensibility checks. Tests may combine stages; the library itself never imports the solver.

Slow (~2 min, full 10 s budget per problem): selected with ``-m slow``.
"""

from pathlib import Path

import pytest

from roomlayout.constraints.checks import check_layout, failed
from roomlayout.constraints.rules import implied_constraints
from roomlayout.schema import Fixed, LayoutProblem
from roomlayout.solver import solve

EXAMPLES = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "examples" / "problems").glob("*.json")
    if "infeasible" not in p.stem
)


def minimal_with_rules(path: Path) -> LayoutProblem:
    p = LayoutProblem.model_validate_json(path.read_text())
    data = p.model_dump()
    data["id"] = f"{p.id}_minimal"
    data["constraints"] = [c.model_dump() for c in p.constraints if isinstance(c, Fixed)]
    minimal = LayoutProblem.model_validate(data)
    data["constraints"] += [c.model_dump() for c in implied_constraints(minimal)]
    return LayoutProblem.model_validate(data)


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_minimal_problem_solves_to_a_sensible_layout(path):
    problem = minimal_with_rules(path)
    result = solve(problem, k=10, seed=0, time_budget_s=10.0)
    top = result.layouts[0]
    assert top.feasible, result.notes
    bad = failed(check_layout(problem, top))
    assert not bad, [f"{r.name}: {r.detail}" for r in bad]
