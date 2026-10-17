# Rule library and sensibility checks (`roomlayout.constraints`, WS-C / M3a)

M1 found that **layout quality is constraint completeness**. The solver reliably satisfies
what it is told, and a "valid but odd" layout almost always means a relation that people
take for granted was never stated: the TV turned side-on to the sofa, the desk chair at
the end of the desk, the coffee table next to the sofa. The rule library states those
relations, and the checks verify the result the way a person looking at the plan would.

```python
from roomlayout.constraints.rules import implied_constraints
from roomlayout.constraints.checks import check_layout, failed

extra = implied_constraints(problem)  # only the constraints to add
problem = problem.model_copy(update={"constraints": [*problem.constraints, *extra]})
# ... solve (orchestrator) ...
problems = failed(check_layout(problem, result.layouts[0]))
```

Both modules import `roomlayout.schema` (+ shapely) only, never the solver (ADR-0002).

## Roles

An asset's role is `(category, subtype)` from the shared `roomlayout.schema.taxonomy`.
Sources are tried in order:
1. `asset.subtype` (resolved through the taxonomy, so aliases work);
2. `asset.category` (an alias such as `couch` resolves to `sofa`);
3. if no subtype yet, the words of the **label**, then of the **id**, longest phrase
   first (`"Coffee Table"` → `coffee_table`, id `desk_chair` → `office_chair`,
   `tv_console` → `tv_stand`).

A word-derived subtype is accepted only if it belongs to the asset's category. So
"low console under the TV" (storage) never becomes a television, and "TV on stand"
(television) never becomes a TV stand.

| role | how it's picked |
|---|---|
| primary seating | the largest-footprint asset of category `sofa` (ties: first listed) |
| secondary seating | the other sofas, and `armchair` / `loveseat` / `sectional` |
| TV | the widest `television`; if there is none, the first `tv_stand` stands in for it |
| coffee table | the largest `coffee_table` |
| desk + chair | each `desk` gets an `office_chair`, paired by shared id tokens (`desk_a` ↔ `chair_a`), then order. Generic chairs (`chair`, no subtype) fill missing desk chairs. |
| table set | `dining_table` / `round_table` / `meeting_table` + `dining_chair`s, plus the generic chairs left over; split round-robin over several tables |
| zone invitations | assets left without a role, in this order: `piano_bench`, `bench`, generic chairs, `stool`. Each clearance zone whose `allowed_categories` admit one gets one member (desks and table sets excluded: their chairs are assigned above). Benches and stools never become desk or dining chairs. |
| piano bench, no zone | a `piano_bench` that no zone invited goes to a piano-labelled asset (`piano` is matched on words: it is not a taxonomy category) |
| ottoman | each `ottoman` is a footrest for the primary seating, else the first secondary seat |
| bed + nightstands | each `bed` + `nightstand`s, round-robin over beds |
| wall storage | `bookshelf`, `wardrobe`, `dresser`, `sideboard`, `cabinet`, `tv_stand` |

Unknown categories (`piano`, `plant`, `thing`) get no role and no relational rules. They
are still subject to the safety rules.

## Hard vs soft, and weights

- **Only safety rules are hard.** A relational rule that happens to be impossible in a
  particular room must degrade the layout, not make the problem infeasible. M1 showed how
  quickly hard relations stack up into infeasibility in crowded rooms.
- **Weight scale.** All relational weights are multiplied by `WEIGHT_SCALE = 5`. The
  solver weighs hard violations at 100 per unit (m² / m / rad). With weights around 1,
  relations were settled only after the hard rules and got frozen out in crowded rooms:
  the minimal `crowded_studio` returned a sofa 73° off its TV. Weights of 5–15 let them
  shape the search from the start, while staying far below any hard violation.
  - Measured on the minimal suite (seed 0, effective weight multiplier ×1 / ×3 / ×10):
    top layout passes all checks on 9/10, 10/10 and 10/10 problems; good layouts in the
    top-10s total 59, 73 and 77.
  - ×10 slowed the first feasible layout in `crowded_studio` (0.2 s → 2.1 s), so we use 5.
- The *effective* weight (after scaling) is what the table below lists.

## Rule table

| rule | when it fires | constraint(s) | hard / soft (weight) | rationale |
|---|---|---|---|---|
| seating ↔ TV | primary seating and a TV | `faces(seat, tv, 20°)`, `faces(tv, seat, 30°)` | soft 15 each | you watch from the sofa; the screen is turned to the sofa (screens are watchable a bit off-axis, hence 30°) |
| viewing distance | primary seating and a TV | `near(seat, tv, clamp(2.5·diag − 0.3, 2.4, 4.0))`, `distance_at_least(seat, tv, max(1.2, diag − 0.3))` | soft 5 each | footprint-to-footprint band around common 1–2.5× diagonal guidance; diagonal estimated from the TV width (16:9); −0.3 m ≈ eye position behind the seat front |
| TV on its stand | a television and a separate `tv_stand` | `near(tv_stand, tv, 0.05)` | soft 5 | the TV stands on, or hangs above, its stand (needs `elevation` on the TV for "above") |
| coffee table | primary seating and a coffee table | `faces(seat, table)`, `near(table, seat, max(0.5, legroom + 0.15))`, `distance_at_least(table, seat, max(0.35, legroom))`, `parallel(table, seat)` | soft 10 / 7.5 / 5 / 2.5 | in front, within reach, outside the seat's own legroom zone (its depth is read from the seat's clearance zones that a `table` may not enter, so the band is always satisfiable), squared to the seat |
| secondary seating | armchair / loveseat / other sofa | coffee table present: `faces(s, table, 30°)` + `near(s, table, 1.0)`; else TV: `faces(s, tv, 45°)`; else `faces(s, primary, 45°)` | soft 5 + 2.5; 2.5; 2.5 | joins the conversation / TV group |
| side table | each side table, round-robin over seats | `near(side_table, seat, 0.1)` | soft 5 | a side table belongs beside a seat |
| free-standing TV | television that is not wall-mounted | `against_wall(tv)` | soft 10 | back to a wall (wall-mounted ones are already handled by the solver's `implicit:wall_mount`) |
| desk + chair | each desk/chair pair | `near(chair, desk, 0.15)`, `faces(chair, desk)`, `faces(desk, chair)` | soft 10 each | the chair is pulled up at the desk's **front**. Mutual `faces` is what excludes "beside the desk's end" (M1). |
| desk on wall | each desk | `against_wall(desk)` | soft 2.5 | weak preference; desks can face into a room |
| zone invitation | an asset (not a desk / table set) has a clearance zone with `allowed_categories`, and an otherwise unassigned asset of that category exists (piano bench, bench, leftover generic chair, stool) | `near(member, owner, 0.3)`, `faces(member, owner)`, and `faces(owner, member)` if the zone is in front of the owner | soft 7.5 each | the data already says the member belongs there: a piano's `bench_area` admits chairs, so the piano bench goes in it. One member per zone, no taxonomy entry needed. |
| ottoman | each ottoman | `near(ottoman, seat, 0.6)` | soft 5 | footrest / extra seat by the main seating |
| piano bench at a piano | a `piano_bench` without an inviting zone, and a piano-labelled asset | `near(bench, piano, 0.3)`, `faces(bench, piano)`, `faces(piano, bench)` | soft 7.5 each | the bench is at the keyboard (the piano's front) |
| table set | each chair of a dining / meeting table set | `near(chair, table, 0.15)`, `faces(chair, table)` | soft 5 each | chairs at the table, facing it. There is no even-distribution primitive, so none is faked (two chairs at one end is valid). |
| bed | each bed | `against_wall(bed)` | soft 15 | the headboard (the bed's back, local −y) is against a wall |
| nightstands | each nightstand of a bed | `near(ns, bed, 0.1)`, `parallel(ns, bed)`, `against_wall(ns)` | soft 7.5 / 2.5 / 5 | right next to the bed, squared to it, against the headboard wall. Together with the bed's own `against_wall` this puts it beside the head end. |
| nightstands on both sides | a bed with exactly 2 nightstands | `distance_at_least(ns1, ns2, 0.8 × bed width)` | soft 5 | proxy for "one on each side": two nightstands next to the bed and that far apart can't be on the same side |
| wall storage | bookshelf, wardrobe, dresser, sideboard, cabinet, tv_stand | `against_wall(x)` | soft 10 | storage stands with its back to a wall |
| walkway | always (once per problem) | `walkway_width(0.8)` hard + `walkway_width(0.9)` soft | **hard** + soft 2 | every door and every piece of furniture stays reachable. 0.9 m (≈36 in) is the usual circulation guidance; 0.8 m (≈32 in, a typical minimum clear doorway width) is the floor we never go below. |
| doors / passages | each door and doorless passage | `keep_clear(opening, 0.9)` | **hard** | a person stands in front of a door; same depth as the walkway |
| windows | each window | `keep_clear(window, 0.4)` | soft 0.5 | keep windows reachable; only assets taller than the sill count (solver semantics) |

**Ids** are deterministic: `rule:<type>:<a>[:<b>]`, e.g. `rule:faces:sofa:tv`,
`rule:against_wall:bookshelf`, `rule:keep_clear:door_main`, `rule:walkway_width`,
`rule:walkway_width:preferred`.

### Why the hard walkway is 0.8 m and not 0.9 m
With 0.9 m hard, the minimal `crowded_studio` (10 assets in 22 m²) was infeasible. The
original hand-written problem used 0.8 m. So 0.9 m is kept as a strong preference and
0.8 m is the hard floor, and every problem in the suite gets 0.9 m wherever it fits. If
you want a strict 0.9 m, it is one constant (`WALKWAY_MIN`). The cost is that crowded
rooms then report "infeasible" rather than offering a slightly tighter layout.

## Precedence (ADR-0006) and idempotence

User/VLM constraints (`origin` ≠ `rule_library`) override rules on the same subject.

| implied rule | skipped when |
|---|---|
| `faces(a, *)` | any non-rule `faces(a, *)` exists (the user has said where `a` faces) |
| `near` / `distance_at_least` on {a, b} | any non-rule `near` or `distance_at_least` on the same pair, either order |
| `parallel` on {a, b} | any non-rule `parallel` on the pair |
| `against_wall(a)` | `a` already has any `against_wall`; `a` has any `fixed` (hard or soft); `a` is immovable; `a` is wall-mounted |
| any relation on {a, b} | both are pinned (hard `fixed` or immovable): the relation would be a constant |
| `walkway_width` | any `walkway_width` exists (both the hard and the soft one are skipped) |
| `keep_clear(opening)` | that opening already has any `keep_clear` |
| any rule | a constraint with the same id exists (**idempotence**: running the library on its own output adds nothing) |

Rule-origin constraints with a *different* id don't count as user intent. Only the exact
id dedupes them, so stale rule output never silences a rule.

## Sensibility checks

`check_layout(problem, layout)` returns a `CheckResult(name, passed, detail)` for every
check that applies to the furniture present. Names carry the subjects, e.g.
`desk_chair_at_desk_front[desk:desk_chair]`. Thresholds are a bit more lenient than the
rules that aim for them, so a check failing means visibly wrong, not merely
non-optimal.

| check | passes when |
|---|---|
| `seating_faces_tv[seat:tv]` | the TV's centre is within 30° of the seat's front |
| `tv_faces_seating[tv:seat]` | the seat is within 40° of the screen's front |
| `tv_viewing_distance[seat:tv]` | footprint distance 1.0–4.5 m |
| `tv_line_of_sight[seat:tv]` | no asset taller than 1.0 m (bottom below 1.8 m; not the TV stand) crosses the seat→TV centre line |
| `coffee_table_between_seating_and_tv[t]` | the table's centre projects strictly between seat and TV, ≤ max(0.5 m, half the seat width) off the line |
| `coffee_table_in_front_of_seating[t:seat]` | in front of the seat's front edge, within its width, ≤ 0.8 m away |
| `secondary_seating_oriented[s]` | faces the coffee table, TV or primary seat within 45° |
| `side_table_by_seating[t]` | ≤ 0.3 m from a seat |
| `desk_chair_at_desk_front[desk:chair]` | chair centre in front of the desk (0–0.7 m), within its width, facing it within 30° |
| `dining_chair_at_table[c]` | ≤ 0.3 m from the table, facing it within 30° |
| `bed_head_against_wall[bed]` | back edge within 0.1 m of a wall, facing in within 15° |
| `nightstand_beside_bed_head[ns]` | beside the bed (≤ 0.8 m out from its side) and within 0.7 m of the headboard line |
| `nightstands_on_both_sides[bed]` | two nightstands on opposite sides |
| `storage_against_wall[x]` / `tv_against_wall[tv]` | as for the bed (immovable storage is skipped) |
| `invited_at_zone[owner:member]` | the member's centre lies in the inviting zone (+0.1 m) |
| `seat_at_front_of[piano:bench]` | the bench is in front of the piano, within its width, ≤ 0.6 m away |
| `ottoman_by_seating[o]` | ≤ 0.8 m from its seat |
| `door_unblocked[door]` | no footprint below the opening's top in the first 0.6 m in front of a door / passage |

The checks are the precursor to the VLM critic, and they turn "layouts look sensible"
into a test.

## Evaluation

For each feasible suite problem, the *minimal* variant strips every constraint except
`fixed` (the user's essentials), then adds `implied_constraints`, then solves with the
default 10 s budget and 4 workers. `tests/constraints/test_rules_eval.py` (slow) asserts
that the top layout is feasible and passes every check. The table covers seeds 0 / 1 / 2.
"good" = layouts in the returned top-10 that are feasible and pass all checks.

| problem (minimal) | rules emitted | top feasible (s0/s1/s2) | top-layout checks passed | good layouts in top-k | max first feasible [s] |
|---|---|---|---|---|---|
| `angled_reading_room` | 10 (2 hard) | y / y / y | 4/4 / 4/4 / 4/4 | 10/10 / 10/10 / 10/10 | 0.12 |
| `bedroom_basic` | 17 (2 hard) | y / y / y | 7/7 / 7/7 / 7/7 | 4/5 / 2/5 / 4/5 | 1.17 |
| `crowded_studio` | 20 (2 hard) | y / y / y | 12/12 / 12/12 / 12/12 | 2/6 / 2/3 / 3/3 | 2.87 |
| `dining_room` | 18 (3 hard) | y / y / y | 9/9 / 9/9 / 9/9 | 4/10 / 3/8 / 2/8 | 2.95 |
| `fixed_piano_room` | 14 (2 hard) | y / y / y | 5/5 / 5/5 / 5/5 | 9/10 / 9/10 / 9/10 | 0.08 |
| `l_shaped_living_dining` | 24 (3 hard) | y / y / y | 14/14 / 14/14 / 14/14 | 3/7 / 6/8 / 3/10 | 1.78 |
| `living_room_basic` | 16 (2 hard) | y / y / y | 10/10 / 10/10 / 10/10 | 10/10 / 10/10 / 10/10 | 0.12 |
| `narrow_study` | 11 (2 hard) | y / y / y | 5/5 / 5/5 / 5/5 | 10/10 / 7/7 / 5/7 | 0.33 |
| `office_with_column` | 15 (2 hard) | y / y / y | 6/6 / 6/6 / 6/6 | 10/10 / 10/10 / 10/10 | 0.23 |
| `wall_tv_living` | 17 (2 hard) | y / y / y | 10/10 / 10/10 / 10/10 | 7/10 / 8/10 / 5/10 | 0.22 |

All 30 top layouts (10 problems × 3 seeds) are feasible and pass every applicable check. The weaker number is the rest of the top-k (see Known gaps). Some runs return fewer than 10 layouts: the minimal problems carry two walkway checks, so restarts are slower, and the diversity filter finds fewer distinct feasible layouts within 10 s.

## Known gaps

- **Lower-ranked layouts** (fixed in the solver, v0.6 round):
  - The top-k used to be padded with feasible but poor layouts. The solver's quality gate
    now returns only layouts within `cost_margin` (1.0) of the best, and counts only those
    toward early stopping.
  - Desk+chair, bed+nightstand and table+chair ties (soft `near` ≤ 0.3 m, weight ≥ 5) now
    move as rigid groups in the search.
  - See the before/after table under Evaluation.
- **Head end of the bed.** There is no "beside the head end" primitive. It's approximated
  by `near` + `against_wall` + `parallel` + the both-sides distance; checked by
  `nightstand_beside_bed_head`.
- **Even distribution** of dining chairs and **line of sight** are not expressible (backlog,
  per the tech lead). Line of sight is checked, not enforced.
- **Taxonomy.** v0.6.0 added `bench`, `piano_bench`, `ottoman` and `meeting_table`, with
  roles above. `piano` and `plant` are still not categories (deferred: they need a policy
  for non-RoomPlan categories), so pianos are matched on words.
  - The example `angled_reading_room` declares its ottoman as category `table`. The word
    "ottoman" resolves to a `chair` subtype, so the category check rejects it and that
    ottoman gets no role. That's an authoring issue in the example; the rule is correct.
  - A plain `bench` with no inviting zone gets no relation (a hall bench needs none).
- **Generic chairs** (category `chair`, no subtype, no telling label/id) are guessed:
  first to desks lacking a chair, then to dining tables, otherwise no role.
