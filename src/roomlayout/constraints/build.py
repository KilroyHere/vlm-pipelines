"""Stage 4 orchestration: user words + problem -> complete, solver-ready problem.

Order matters (ADR-0006/0007): the user's constraints are added first, then the rule
library fills in what the user didn't speak about, so user intent always wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomlayout.schema import Constraint, LayoutProblem

from .rules import implied_constraints

if TYPE_CHECKING:
    from .vlm import ParseResult


@dataclass
class BuildResult:
    problem: LayoutProblem  # original constraints + user (VLM) + rule library
    parse: ParseResult | None  # None when no text was given
    from_user: list[Constraint] = field(default_factory=list)
    from_rules: list[Constraint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_constraints(
    problem: LayoutProblem,
    text: str | None = None,
    *,
    client: Any = None,
    rules: bool = True,
    **parse_kwargs: Any,
) -> BuildResult:
    """Parse ``text`` with the VLM (if given), merge, then add rule-library constraints.

    A failed parse (refusal, truncation, invalid output) is not fatal: the result keeps
    the ``ParseResult`` so the app can tell the user, and the rule library still runs.
    """
    notes: list[str] = []
    parse = None
    from_user: list[Constraint] = []
    if text and text.strip():
        from .vlm import parse_request  # optional dependency (the `vlm` extra)

        parse = parse_request(problem, text, client=client, **parse_kwargs)
        if parse.status != "ok":
            notes.append(f"request not understood ({parse.status}): {parse.message or ''}")
        taken = {c.id for c in problem.constraints}
        for c in parse.constraints:
            if c.id in taken:
                notes.append(f"dropped user constraint {c.id!r}: id already in the problem")
                continue
            taken.add(c.id)
            from_user.append(c)

    merged = _with_constraints(problem, [*problem.constraints, *from_user])
    from_rules = implied_constraints(merged) if rules else []
    final = _with_constraints(merged, [*merged.constraints, *from_rules])
    return BuildResult(final, parse, from_user, from_rules, notes)


def _with_constraints(problem: LayoutProblem, constraints: list[Constraint]) -> LayoutProblem:
    data = problem.model_dump(mode="json")
    data["constraints"] = [c.model_dump(mode="json") for c in constraints]
    return LayoutProblem.model_validate(data)  # re-run cross-reference validation
