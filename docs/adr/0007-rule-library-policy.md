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

## Amendment (2026-10-17): first check against real rooms (ARKitScenes end-to-end)

Running the full chain on three real living rooms (`spikes/e2e_arkitscenes/`) showed two
places where the user did *not* actually take precedence:

- **Soft user requests weighed 1.0 against rules at ×5.** An armchair's default "faces the
  coffee table" (weight 5) pulled the table away from the sofa the user named (weight 1).
  Soft VLM/user constraints now get `USER_SOFT_WEIGHT = 3 × WEIGHT_SCALE` (15, the
  strongest default relation, seating↔TV). Hard requests are unaffected.
- **Faces precedence was per subject.** "The sofa faces the TV" suppressed every implied
  `faces(sofa, *)`, including "the coffee table is in front of the sofa", so the table
  drifted beside the sofa (2 of 3 rooms). An implied `faces(seat, coffee_table)` now
  survives a user `faces(seat, tv)`: the library emits that pair itself and the two agree
  when the table sits between seat and TV. Any other user `faces(seat, x)` still
  suppresses it.

Effect on the three rooms (box assets, k=3, same inputs before/after): layouts passing
every sensibility check, not counting the flag on a cabinet the user pinned with "don't
move it", went from 2 of 9 to 7 of 7 (one room now returns a single layout; the top-k
quality gate drops the costlier alternatives). User constraints satisfied in the top
layout went from 2/3, 5/5, 4/5 to 3/3, 5/5, 5/5.
