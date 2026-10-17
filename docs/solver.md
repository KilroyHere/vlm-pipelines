# Layout solver (`roomlayout.solver`, WS-A / M1)

Simulated annealing over 2D footprints (ADR-0003). Input: a `LayoutProblem`; output: a
`SolveResult` with up to `k` mutually diverse layouts, best first.

```python
from roomlayout.solver import solve

result = solve(problem, k=10, seed=0, time_budget_s=10.0)
```

```bash
python -m roomlayout.solver examples/problems/bedroom_basic.json --out out/bedroom \
    [--k 10 --seed 0 --budget 10 --workers 4 --no-png]
```

The CLI writes `result.json` (a `SolveResult`; `stats` carries `restarts`,
`time_to_first_feasible_s` and `time_limited` since contract v0.3.0), `diagnostics.json`
(extra search statistics: feasible restarts, feasible layouts returned) and `layout_NN.png`
per layout.

## Modules

| module | role |
|---|---|
| `geometry.py` | `RoomGeom` (prepared floor, walls with interior normals, openings, door-swing discs, keep-clear rectangles, obstacles, allowed yaws), `AssetTemplate` (local footprint + zones as arrays) and `Posed` world geometry, cached per `(asset, x, y, yaw)` |
| `rules.py` | one evaluator per constraint type and implicit rule, each returning a continuous violation; the walkway check |
| `model.py` | compiles a problem into unary / pair / walkway cost terms; `State` = incrementally maintained cost; `Model.report()` = full per-constraint evaluation |
| `anneal.py` | one restart: randomized greedy construction + simulated annealing |
| `diversity.py` | layout distance and greedy diverse top-k |
| `solve.py` | `solve` / `solve_detailed` / `evaluate`: restarts (multiprocessing), stopping rule, selection, infeasibility notes |
| `render.py`, `__main__.py` | top-down PNG (matplotlib Agg) and CLI |

`solve` imports `roomlayout.schema` only. matplotlib is imported only when rendering.

## Semantics decisions

Frames follow ADR-0001: front = local +Y, `front = (-sin yaw, cos yaw)`. Walls run CCW,
and the interior normal is the left of start → end. **Assumption:** wall segments lie on the
floor polygon's boundary (the floor polygon is the interior face of the walls).

All violations are continuous so annealing has a gradient. Areas below `AREA_EPS = 1e-6 m²`
count as zero, so touching footprints are fine.

### Implicit hard rules (always reported as `implicit:<name>`)

| rule | definition | violation |
|---|---|---|
| `inside_room` | footprint ⊆ floor polygon | m² outside, plus the centre's distance to the floor if that is > 0 (gives a gradient when fully outside) |
| `no_overlap` | pairwise footprints | m² of intersection |
| `clear_of_obstacles` | footprints vs `Room.obstacles` | m² of intersection |
| `clearance_zones` | see below | m² |
| `door_swing` | footprints vs the interior swing quarter-disc of each door | m² |
| `wall_mount` | `mount: wall` assets satisfy `against_wall` (any wall, `max_gap` 0.05 m) | as `against_wall` |

`implicit:wall_mount` is our name for the contract's "wall-mounted assets are placed with
their back flush to a wall". It is always listed, and is trivially satisfied when nothing
is wall-mounted.

**Clearance zones (contract v0.2.0).**
1. *Pair part:* another asset's footprint inside a zone is a violation, unless its
   `category` is in the zone's `allowed_categories` (e.g. chairs in a desk's chair zone or
   a dining table's seating zones). Zones may overlap other zones, door swings and
   keep-clear areas. The contract clips zones to the floor before this check. Footprints
   must lie inside the floor (hard rule), so zone ∩ footprint equals clipped-zone ∩
   footprint for every feasible layout, and the explicit clip is skipped for speed.
2. *Unary part:* a zone overlapping a fixed obstacle (column, radiator) is a violation,
   since you can't stand there. A zone that leaves the floor (cut off by a wall) is a
   violation too, with one exception that implements WS-B's request:
   - If the asset's back is **flush against a wall** (the `against_wall` test with a 5 cm
     gap passes), the part of a zone *behind its back-edge line* is clipped silently. That
     part pokes through the wall the asset stands against.
   - Zone parts at or in front of the back line must stay inside the floor.
   - Why not clip everything (the literal contract)? Then a wardrobe could face a wall
     10 cm away, a bed could stand with one access side in a corner, and a dining table
     could be pushed into a wall over its seating zone. All of these are "valid" but
     unusable. The last one showed up in `l_shaped_living_dining` with an earlier, looser
     version of this rule.
   - The policy is one constant, `geometry.ZONE_WALL_POLICY`: `"front"` (default),
     `"clip"` (pure contract: zones never conflict with walls), or `"strict"` (v0.1: any
     part of a zone outside the floor counts).

**Door swing.** A quarter disc of radius = opening width, centred on the hinge jamb
(`hinge: start` = the jamb nearer the wall start). It sweeps from the closed leaf (along
the wall) to 90° open on the interior side. `into_room: false` → nothing to check inside
the room. A door with `swing: null` → no swing check (its `keep_clear`, if any, still
applies). Asset clearance zones may overlap a swing.

### Constraint vocabulary

| type | definition | violation |
|---|---|---|
| `against_wall` | back edge = the footprint vertices at minimal local y (their x-range; a point for round shapes). Both endpoints within `max_gap` of the wall *segment*, the front within 10° of that wall's interior normal, and the back edge not across a door/passage span (there is no wall there) nor across a window if the asset is taller than the sill (or is wall-mounted). With `wall_id` only that wall counts, otherwise the best wall. | m gap beyond `max_gap` + rad beyond 10° + m of back edge across openings |
| `faces(a,b)` | angle between a's front and the vector from a's origin to b's origin (bounding-box centres) ≤ `max_angle_deg` | rad beyond the limit |
| `near(a,b)` | shapely distance between footprints ≤ `max_distance` (0 when they touch or overlap) | m beyond |
| `distance_at_least(a,b)` | footprint distance ≥ `min_distance` | m short |
| `parallel(a,b)` | \|Δyaw\| modulo 180° ≤ `max_angle_deg` | rad beyond |
| `keep_clear(opening)` | rectangle = opening width × `distance` along the interior normal. For doors/passages every footprint counts except assets mounted entirely above the opening; for windows only assets whose top (`elevation + height`) is above `sill_height` | m² of footprint inside |
| `walkway_width` | see below | see below |
| `fixed(a)` | hard: the asset is **pinned** at `initial` and never moved (always satisfied). Soft: the asset may move; violation = m displacement + rad rotation from `initial` | as stated |

`movable: false` assets are pinned at their `initial` pose too (the contract requires one
since v0.3.0). A pair constraint relating an asset to itself is ignored, with a note.

### Elevation (contract v0.3.0)

Each asset occupies the vertical extent `[elevation, elevation + height]`.
- `no_overlap` applies only to pairs whose extents overlap; extents that merely touch,
  e.g. a TV at 1.0 m over a 0.5 m console, don't conflict. This is decided once per pair
  when the problem is compiled.
- Clearance zones and walkways are floor-level space for a walker `WALKER_HEIGHT = 1.8 m`
  tall. Assets with `elevation ≥ 1.8 m` never enter a zone and never block the walkway;
  anything lower does.
- Obstacles stand on the floor, so an asset conflicts with one only if its bottom is below
  the obstacle's top (e.g. a wall cabinet above a radiator is fine).
- The door swing applies only to assets whose bottom is below the opening's top.
- Window `keep_clear` counts assets whose top is above the sill. Door/passage
  `keep_clear` ignores assets mounted entirely above the opening.
- `against_wall` / `wall_mount` may not cross a window whose glass `[sill, sill + height]`
  overlaps the asset's extent, nor a door/passage unless the asset is above it.
- The diversity signature includes elevation. The "footprints exceed the floor" proof
  only sums floor-standing assets (`elevation = 0`), which all overlap vertically.

**Walkway width.** Free floor = floor − footprints − obstacles. It is eroded by
`r = min_width / 2` and computed as `erode(floor, r) − ⋃ dilate(footprint, r) − ⋃
dilate(obstacle, r)`. Dilations are cached per pose, and buffers use 4 segments per
quarter circle. The result is the locus of centres of a walker of width `min_width`.

Targets:
- every door and passage: the centre of the opening on the wall line;
- every floor asset with an access-purpose zone: the union of those zones;
- every other floor asset: its front edge, 2 cm in front.
- Wall-mounted assets have no target.

A component C of the eroded space *reaches* a target if `distance(target, C) ≤ r + 0.05 m`,
i.e. the walker's body touches it. The walkway is the component reaching the most targets
(ties go to the largest area).

Chaining rule (one level): a front-edge target also counts as reached if it lies within
5 cm of an access zone that is itself reached. This is what makes a nightstand next to the
bed reachable through the bed's 0.6 m side zone instead of needing a 0.8–0.9 m corridor of
its own. Zones are guaranteed free standing space.

Violation = Σ over missed targets of `0.1 + min(distance − r − 0.05, 3)` m.

*Laziness:* the walkway check is the expensive one (~0.5–1 ms), so `State.propose` only
evaluates it when every cheap hard term of the candidate is zero. Otherwise it carries the
last exact value; such candidates already pay `W_HARD` and are infeasible anyway. A layout
is feasible only if its walkway was evaluated exactly. Final layouts are re-evaluated from
scratch by `Model.report`.

*Several walkways* (the rule library emits 0.8 m hard + 0.9 m soft): `Model.eval_walkways`
evaluates them widest first, and once one is fully satisfied every narrower one is too, so
it is skipped. Why that holds: the centre locus for a narrower width `w'` contains the
`(w − w')/2` dilation of the locus for `w`, while the reach tolerance `r + 5 cm` shrinks
by exactly that amount, so any target reached at `w` is reached at `w'`. The same goes for
chained zone targets. A property test checks this against brute force on random layouts.
In the common late-search case this halves the walkway cost. Search and report use the
same helper, so they cannot disagree.

## Cost function

```
cost = W_HARD · Σ hard violations + Σ weight · soft violations      (W_HARD = 100)
```

`Layout.feasible` = every hard violation is 0. `Layout.results` lists every constraint of
the problem plus the six `implicit:*` rules, with the violation and a per-asset detail
string. Units are mixed (m², m, rad), and each is chosen so that "small but visible" is
about 0.01–0.1: 10° ≈ 0.17 rad, 0.1 m, 0.1 m². `W_HARD = 100` puts any visible hard
violation above typical soft totals.

## Search

**Per restart** (`anneal.py`):
1. *Construction:* pinned assets at `initial`, then hard wall-bound assets
   (`against_wall` hard or wall-mounted), then the rest, by decreasing footprint+zone area
   (slightly shuffled). Each is placed at the best of 24 random candidates, judged by the
   cost so far:
   - wall-bound: back on a random wall (weighted by length), facing in;
   - others: next to a `near`/`faces` partner (facing it for `faces`), on a wall, or
     anywhere, with an allowed yaw.

   Restart 0 starts from the problem's `initial` poses, if any.
2. *Annealing:* `300 + 250 · movable` steps. The temperature falls geometrically from 3 to
   1e-4, and the translation sigma shrinks linearly from 0.6 m to 0.03 m. Each step picks
   one movable asset and a move:
   - **translate**: Gaussian x/y step;
   - **slide**: along its own left-right axis, which keeps wall contact;
   - **rotate**: to another allowed yaw;
   - **jump**: anywhere, or next to a related asset;
   - **snap**: back to a random wall, facing in;
   - **swap**: exchange poses with another asset; wall-bound ones re-snap to the nearest
     wall;
   - **back**: return to the initial pose (soft `fixed` only).

   Wall-bound assets mostly slide and snap.
   - *Group moves* (p = 0.5): smaller assets tied to the moved one by a `near` of
     ≤ 0.3 m that is hard **or soft with weight ≥ 5** (`GROUP_SOFT_WEIGHT`; the rule
     library's desk → chair, bed → nightstands, table → chairs ties) follow it rigidly.
   - *Focus* (p = 0.5): after 30% of the schedule, while the state is infeasible, the moved
     asset is drawn from those involved in a violated hard term.
3. *Result:* the best state seen, by (infeasible, cost), plus up to 3 late "just became
   feasible" states as extra diversity candidates.

**Allowed yaws:** multiples of 90°, plus every wall's facing yaw and its 90° rotations, so
assets sit square to angled walls (see `angled_reading_room`).

**Restarts and stopping** (`solve.py`): restart `i` uses `random.Random(seed·1000003 + i)`.
Restarts run in a process pool: `min(4, cpu)` workers, with `fork` where available
(otherwise `spawn`). Results are consumed in restart-index order.

Search stops at the first index `i ≥ min_restarts − 1` (default 16) at which restarts
`0..i` contain `k` mutually diverse layouts **that pass the quality gate** (below), or at
`max_restarts` (512), or when the time budget is used up (5% is reserved for selection and
reporting).

**Quality gate** (`cost_margin`, default `1.0`; `None` disables it): only feasible layouts
with `cost ≤ best feasible cost + cost_margin` are returned, and only those count toward
the stopping rule.
- Rationale: with soft-only relations (the rule library), "feasible" says little about
  quality, and diversity used to pad the top-k with feasible but visibly wrong layouts
  (nightstands on one side, the chair at the desk's end). Fewer, better layouts are the
  right answer.
- Scale of 1.0: with rule-library weights of 5–15, that is about 4–6° beyond a `faces`
  limit or 0.1–0.2 m beyond a `near` limit.
- When fewer than `k` layouts pass, `SolveResult.notes` says how many more distinct
  feasible layouts were found and left out.

**Determinism:** the result depends only on (problem, k, seed, limits), not on the number
of workers or scheduling (tested, workers 1 vs 2). The caveat: if the *time budget* rather
than the stopping rule ends the search, the result depends on machine speed. This always
happens for infeasible problems, which search until the budget runs out.
`SolveDiagnostics.time_limited` says which case applies. For bit-exact reproducibility
regardless of speed, pass `max_restarts` (and optionally `iterations`) with a generous
budget.

## Diversity

The distance between two layouts is the mean over movable assets of
`|Δposition| + 0.5 m · |Δyaw| / 90°`.

Identical assets (same category, footprint, zones incl. `allowed_categories`, mount,
height) form interchangeable groups, matched by the cheapest assignment (exhaustive up to
4 assets, greedy above). Swapping two identical dining chairs is therefore not a new
layout.

Selection is greedy over candidates sorted by (infeasible, cost): a candidate is kept if
its distance to every kept layout is ≥ **0.3 m** (`MIN_LAYOUT_DISTANCE`). As a feel for
0.3 m: with 5 movable assets, moving one of them 1.5 m clears it, and so does turning
three of them 90°.

If there are fewer than `k` distinct feasible layouts within the cost margin, fewer are
returned, with a note;
infeasible layouts are never used to pad a feasible result. If nothing is feasible, the
best diverse infeasible layouts are returned with `feasible: false`.

## Infeasibility notes

When nothing feasible is found, `SolveResult.notes` contains:

- **Static proofs**, prefixed `infeasible:`:
  - footprint area > free floor area;
  - an asset larger than the room's bounding box in both orientations;
  - a hard `near` max smaller than a hard `distance_at_least` min on the same pair;
  - a hard `against_wall(wall_id)` wider than the longest usable run of that wall;
  - pinned assets already violating hard rules.
- **Heuristic:** footprints plus their own zones need > 85% of the free floor ("very
  crowded").
- **Statistics:**
  - the hard constraints most often violated by each restart's best layout, as a % of
    restarts;
  - the most frequent specific conflicts, e.g. `implicit:clearance_zones (bed×wardrobe)`.
- If no static proof was found, a line saying how many restarts were searched and whether
  the budget ran out.

## Rendering

One PNG per layout. It shows:
- the floor, with walls drawn outward by their thickness and gaps at doors and passages;
- windows (blue), door swings (yellow quarter disc with the open leaf), keep-clear
  rectangles (red dotted), obstacles (cross-hatched);
- the walkway free space when the problem has a walkway constraint (green: the locus of
  the walker's centre);
- footprints with id label, front arrow and thick back edge, and clearance zones
  (hatched);
- a side panel listing violated constraints with details.

## Test suite

`examples/problems/`, all validating against the schema (tests/test_schema.py):

| problem | what it exercises |
|---|---|
| `living_room_basic` | the reference problem (unchanged) |
| `bedroom_basic` | bed with side access zones, 2 nightstands (`near` 5 cm), wardrobe access, desk + chair (chair zone with `allowed_categories`), window keep-clear |
| `l_shaped_living_dining` | L-shaped floor, passage + door, living group + dining table with seating zones and 4 chairs, sideboard |
| `office_with_column` | column obstacle, 2 desk sets, shelves, an octagonal (non-rectangular) meeting table |
| `narrow_study` | 2.2 × 5.5 m room, `against_wall` with `wall_id`, chained walkway |
| `wall_tv_living` | `mount: wall` TV at `elevation` 1.0 m hanging above a console (`near` 0 m, no overlap conflict), `parallel`, viewing distance (`distance_at_least` + soft `near`), the window blocks hanging the TV on the glass |
| `fixed_piano_room` | hard `fixed` piano, `movable: false` built-in, soft `fixed` sofa, `initial` poses (some violating rules), piano bench zone with `allowed_categories` |
| `crowded_studio` | 10 assets in 22 m², bed + sofa/TV + dining + desk |
| `dining_room` | table with seating zones (`allowed_categories: [chair]`), 6 identical chairs, sideboard, passage |
| `angled_reading_room` | a 45° wall, shelf `against_wall(wall_id=w_diag)` |
| `infeasible_tiny_bedroom` | king bed + wardrobe + desk in 2.6 × 2.4 m (crowding) |
| `infeasible_conflicting_constraints` | hard `near ≤ 1 m` vs `distance_at_least ≥ 2 m` (static proof) |

The problems were authored the way a rule library should emit them. For example,
`faces(chair, desk)` **and** `faces(desk, chair)`: with only the former, a chair beside the
desk's end "faces" it. The same goes for `faces(sofa, tv)` **and** `faces(tv, sofa)`, without which the screen can end up side-on to the sofa. Another example is a soft `near(sofa, tv, ~3 m)`: without it the
sofa can face the TV from across the whole room.

Tests (`tests/solver/`):
- `test_rules.py`: hand-placed poses for every rule and constraint type, incl. door-swing
  disc vs square, hinge side, window sill, `wall_id`, zone chaining, `allowed_categories`,
  zone clipping, L-shaped footprints;
- `test_solve.py`: determinism (same seed → identical result, workers 1 vs 2; different
  seed → different), result vs independent evaluation, pinning, infeasibility notes,
  diversity, CLI;
- `test_suite.py`: `@pytest.mark.slow`, about 2 min. Every example with the full 10 s
  budget: a feasible layout for every feasible problem, none plus notes for the
  `infeasible*` ones. Skip it with `pytest -m "not slow"`.

## Performance

4 cores, `time_budget_s=10`, `k=10`, seeds 0/1/2. "first feasible" = wall time from the
`solve` call until any restart first reached a feasible state (min / max over seeds).
"feasible" = feasible layouts among the returned top-10. "restarts" = feasible restart
results / restarts used. "time" = max wall time over seeds, incl. pool start-up.

| problem | assets | first feasible [s] (min / max) | feasible in top-10 (s0 / s1 / s2) | best cost | feasible restarts / restarts used | time [s] (max) |
|---|---|---|---|---|---|---|
| `angled_reading_room` | 6 | 0.07 / 0.19 | 10 / 10 / 10 | 0 / 0 / 0 | 14/16 / 13/16 / 14/16 | 4.7 |
| `bedroom_basic` | 6 | 0.13 / 0.21 | 10 / 10 / 10 | 0 / 0 / 0 | 18/22 / 13/17 / 16/30 | 5.3 |
| `crowded_studio` | 10 | 0.66 / 1.45 | 7 / 8 / 10 | 0.234 / 0.249 / 0.249 | 5/49 / 6/51 / 7/28 | 9.6 |
| `dining_room` | 8 | 0.17 / 0.41 | 10 / 10 / 10 | 0 / 0 / 0 | 8/16 / 10/24 / 10/16 | 6.8 |
| `fixed_piano_room` | 7 | 0.04 / 0.06 | 10 / 10 / 10 | 0 / 0 / 0 | 13/16 / 13/16 / 12/16 | 3.1 |
| `infeasible_conflicting_constraints` | 3 | - | 0 / 0 / 0 | 100 / 100 / 397 | 0/231 / 0/226 / 0/205 | 9.6 |
| `infeasible_tiny_bedroom` | 3 | - | 0 / 0 / 0 | 307 / 307 / 1.56e+03 | 0/108 / 0/103 / 0/106 | 9.6 |
| `l_shaped_living_dining` | 9 | 0.31 / 0.58 | 10 / 10 / 10 | 0 / 0 / 0 | 7/16 / 8/22 / 8/18 | 6.5 |
| `living_room_basic` | 5 | 0.04 / 0.05 | 10 / 10 / 10 | 0 / 0 / 0 | 14/16 / 16/16 / 13/16 | 3.9 |
| `narrow_study` | 6 | 0.09 / 0.18 | 10 / 10 / 10 | 0 / 0 / 0 | 16/16 / 13/16 / 16/16 | 3.8 |
| `office_with_column` | 8 | 0.09 / 0.29 | 10 / 10 / 10 | 0 / 0 / 0 | 15/16 / 15/16 / 16/16 | 5.5 |
| `wall_tv_living` | 7 | 0.05 / 0.17 | 10 / 10 / 10 | 0 / 0 / 0 | 9/16 / 9/16 / 8/16 | 4.4 |

Measured 2026-10-16 on the shared 4-core dev box. Another workstream's test runs
sometimes competed for CPU, so treat the times as upper bounds. Feasible problems stop
early via the stopping rule (k diverse feasible layouts after ≥ 16 restarts); infeasible
ones use the whole budget.

The weakest case is `crowded_studio`: only 5–7 of ~30–50 restarts end feasible, and it
returns 7–10 distinct feasible layouts. Its best cost is ~0.25 because the soft
`sofa_wall` rarely fits alongside everything else.

Throughput is about 2,000–2,500 candidate evaluations per second per core (shapely-bound).

## Known limitations

- **Elevation is a single slab per asset.** Tuck-under geometry (a chair under a table)
  still needs height-banded footprints (ADR-0003).
- **No line of sight** (backlog). `faces` is purely angular: a sofa can "face" a TV across a dining
  table (seen in `crowded_studio`).
- **Under-constrained problems give arbitrary but valid layouts.** Many zero-cost layouts
  tie, and nothing prefers e.g. a coffee table *in front of* the sofa over beside it
  unless the problem says so (`living_room_basic` lacked `faces(sofa, coffee_table)` and
  `faces(tv_stand, sofa)` until v0.3.0).
  There is no built-in aesthetic prior; the rule library must emit these preferences.
- **No even distribution** (backlog). Chairs around a table end up where they fit (two at each end
  of a 1.8 m table is valid); nothing in the vocabulary expresses "3 per long side".
- Yaw is snapped (multiples of 90° + wall-aligned). There are no free angles (e.g. a
  45° armchair in a corner).
- **Walkway is a topological check:** it doesn't measure path *length* or detours, and
  targets are reachable "by touch" (`r + 5 cm`).
- **Time-dependent results** when the budget is binding (see "Determinism").
- Walls are assumed to coincide with the floor polygon's edges; there are no curved walls.

## Next steps

- Backlog: line-of-sight for `faces` (tall blockers only) and even distribution of
  chairs; height-banded footprints for tuck-under.
- A rule library that turns categories into the standard relations used in the examples
  (desk ↔ chair mutual faces, sofa–TV distance, coffee table in front of sofa, seating
  zones).
- Speed: vectorize pair checks (shapely array ops) or switch to an exact
  oriented-rectangle fast path for the common rectangle-vs-rectangle case. Expect 3–5×.
- Optional free-angle refinement (gradient or local continuous yaw) after annealing.
- A VLM critic on the rendered top-k (ARCHITECTURE stage 7) to rerank "valid but odd".

## Schema change requests

All resolved in contract v0.3.0:
- `Asset.elevation`: implemented as above.
- `SolveStats.restarts` / `time_to_first_feasible_s` / `time_limited`: populated.
- Required `initial` for immovable assets: done; the fallback path was removed.
- Zone/wall policy: accepted as ADR-0005, with the contracts.md wording updated by the
  lead.
- Line of sight and even distribution: deferred to the backlog.
