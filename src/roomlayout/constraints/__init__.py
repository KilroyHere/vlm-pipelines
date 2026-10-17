"""Stage 4: turn user intent + room/assets into a complete constraint list.

- ``rules``: deterministic rule library. Implied relations people take for granted
  (seating faces the TV, coffee table in front of seating, ...) plus safety rules.
- ``vlm``: one Claude call that turns the user's own words into constraints from the
  closed vocabulary in ``roomlayout.schema.constraints`` (ADR-0006).

Imports ``roomlayout.schema`` only (ADR-0002). User/VLM constraints take precedence over
rule-library ones (docs/constraints.md).

``build_constraints(problem, text)`` runs both in the right order (``build``).
"""

from .build import BuildResult, build_constraints

__all__ = ["BuildResult", "build_constraints"]
