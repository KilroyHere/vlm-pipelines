# ADR-0007: Rule library: only safety is hard; strong soft relations

- Status: accepted
- Date: 2026-10-17

## Context
The rule library (M3a) adds the relations people take for granted, because layout
quality depends on constraint completeness (M1 finding). Implied relations can be
impossible in a particular room; hard ones would turn an awkward room into "no layout".

## Decision
- **Only safety rules are hard:** walkway, door/passage `keep_clear` 0.9 m. Every
  relational rule (seating↔TV, coffee table, desk+chair, bed+nightstands, storage on
  walls, ...) is soft.
- **Soft rule weights are ×5** relative to ~1. Measured on the minimal suite: ×1 → 9/10
  top layouts pass all checks; ×3 → 10/10; ×10 → 10/10 but crowded first-feasible
  0.2 s → 2.1 s. ×5 balances quality and speed.
- **Walkway: 0.8 m hard + 0.9 m soft (weight 2).** 0.8 m ≈ minimum clear doorway width;
  0.9 m is usual circulation guidance. A hard 0.9 m made the crowded studio infeasible.
- **User/VLM constraints take precedence** (ADR-0006); the library is idempotent.

## Consequences
- Rooms degrade gracefully instead of failing. Quality is enforced by the checks
  (`roomlayout.constraints.checks`) and the solver's top-k quality gate rather than by
  hardness.
- Weights are tuned on our suite; revisit with real scans.
