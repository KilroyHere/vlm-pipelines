# ADR-0008: One composition module; rules among pinned assets alone are facts

- Status: accepted
- Date: 2026-10-18

## Context
RoomPlan ingest (M4a) produces box assets without clearance zones, because `ingest` may
import `roomlayout.schema` only (ADR-0002), and the zones live in `roomlayout.assets`.
Going from a scan to layouts needs ingest, then zones, then rules, then the solver.
Until now that composition lived in tests and spike scripts.

Real RoomPlan exports also showed that immovable things break the solver's implicit
rules as scanned: a fireplace recessed into its wall crosses the floor outline,
abutting kitchen cabinets overlap by millimetres, wall cabinets hang above base
cabinets, and an L-shaped kitchen's access zones run into the other arm. The solver then
proved every such room infeasible, although nothing the user could move was wrong.

## Decision
- **`roomlayout.pipeline` is the one module that composes stages** (ingest, then asset
  defaults, then constraints, then solver), through their public APIs. Stage packages
  still import `roomlayout.schema` only. It is also the CLI for the owner's own-room
  test: `python -m roomlayout.pipeline Room.json --out DIR [--request TEXT]`.
- `roomlayout.assets.with_default_zones(asset)` is the public way to give an asset
  that never went through `normalize()` its category's clearance zones.
- **Implicit rules apply only to terms with at least one movable asset.** A pinned
  asset (immovable, or a hard `fixed`) has no unary implicit terms (inside room, clear
  of obstacles, own zones vs walls, door swing, wall mount), and there is no
  no-overlap/zone term between two pinned assets. Those are facts of the room, which the
  search can't change. Movable assets must still respect pinned ones fully. This mirrors
  the rule library, which already skips relations between two pinned assets.
- Ingest does not edit scanned positions to satisfy the solver. It only drops things
  that aren't in the room (a TV on the other side of a wall) and pulls slightly-outside
  objects in.

## Consequences
- A real kitchen no longer makes the room infeasible by itself.
- A pinned asset that really is misplaced (e.g. a user-`fixed` sofa blocking a door)
  is no longer reported as an implicit-rule violation; user hard constraints and the
  walkway check still apply to it.
- New stages (GOR-IS inpainting, the iOS capture app) plug into `roomlayout.pipeline`,
  not into each other.
