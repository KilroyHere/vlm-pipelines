# Data contracts

The contract lives in `src/roomlayout/schema/` (Pydantic v2). JSON Schemas for
non-Python consumers (iOS app, VLM structured output) are exported to `schemas/`
by `python -m roomlayout.schema.export`, and a test fails if they're stale.

**Ownership:** the tech lead owns the schema. Workstreams consume it and must not
edit it directly; request changes in your PR description or status report.
Breaking changes bump `SCHEMA_VERSION`.

## Frames and units (ADR-0001)

| | Convention |
|---|---|
| Units | meters, radians |
| World | right-handed, +Z up, floor at z = 0 |
| Yaw | rotation about +Z, CCW seen from above |
| Asset local frame | origin = centre of the bounding-box footprint at the asset's bottom; +Z up; **+Y = front** |
| Front in world | `Pose2D.front = (-sin yaw, cos yaw)` |
| Polygons | `Polygon2D`, stored CCW, not closed |
| Walls | run CCW around the floor; interior is on the left of `start → end` |

RoomPlan/ARKit use a Y-up frame. Conversion happens **only** at the ingest boundary
(future `roomlayout.ingest.roomplan`); nothing downstream sees Y-up data.

## Objects

- **`Room`**: floor polygon, walls, openings (door/window/passage, positioned by
  `wall_id` + `offset` of the centre along the wall), and fixed obstacles (world-frame
  footprints).
- **`Asset`**: `category` (coarse, RoomPlan-aligned) + optional `subtype`, dimensions,
  optional local-frame `footprint` (defaults to the bounding rectangle),
  `clearance_zones` (local-frame regions other assets' footprints must not enter,
  except categories listed in the zone's `allowed_categories`, e.g. chairs in a dining
  table's chair zone), `mount`, optional `mesh` (already in canonical frame), and
  `source` provenance + quality signals (`front_confidence`, `aspect_mismatch`).
- **`LayoutProblem`**: room + assets + constraints + optional `initial` poses. This
  is the solver input.
- **`SolveResult`**: ranked, mutually diverse `Layout`s, each with `placements`,
  `cost`, `feasible`, and per-constraint `results`, plus `notes` explaining
  infeasibility.

## Constraint vocabulary (closed set)

Every constraint has `id`, `hard`, `weight` (soft only), `origin`
(`rule_library | user | vlm`), and `rationale`.

| type | meaning |
|---|---|
| `faces(a, b, max_angle_deg=20)` | a's front points at b's centre |
| `against_wall(a, wall_id?, max_gap=0.05)` | a's back edge is within `max_gap` of a wall |
| `near(a, b, max_distance)` | footprint-to-footprint distance ≤ max |
| `distance_at_least(a, b, min_distance)` | footprint-to-footprint distance ≥ min |
| `parallel(a, b, max_angle_deg=5)` | orientation axes parallel (mod 180°) |
| `keep_clear(opening_id, distance=0.6)` | no footprint within `distance` in front of the opening; for windows only assets taller than the sill count |
| `walkway_width(min_width=0.9)` | doors/passages and every access zone mutually reachable through free floor at least `min_width` wide |
| `fixed(a)` | a keeps its `initial` pose |

**Implicit hard rules** (always enforced, reported as `implicit:<name>`):
- `inside_room`, `no_overlap`, `clear_of_obstacles`.
- `clearance_zones`: no other asset's footprint inside an asset's clearance zone,
  unless its category is in the zone's `allowed_categories`. Zones must lie inside the
  room, except the part behind a back edge that is flush against a wall (ADR-0005).
- `door_swing`: the swing quarter-disc stays clear. A door with `swing: null` has no
  swing check.
- `wall_mount`: `mount: wall` assets have their back flush to a wall.

Vertical extent of an asset is `[elevation, elevation + height]`. Footprint conflicts
only count when vertical extents overlap (a wall TV can hang above a console).

**Walkway targets** (`walkway_width`): every door/passage entry point, and for each
asset its access-purpose zone, or a point just in front of its front edge if it has
none. A target also counts as reachable through one level of another asset's access
zone (a nightstand is reachable via the bed's side zone).

## Handoffs between stages

```
assets.normalize ──► Asset (footprint, dimensions, mesh in canonical frame)
                                     │
ingest.roomplan (later) ──► Room ────┼──► LayoutProblem ──► solver ──► SolveResult
                                     │
constraints.vlm (later) ──► Constraint[] (validated against this vocabulary)
```

## Changelog
- 0.3.0: `Asset.elevation`; `SolveStats.restarts`, `time_to_first_feasible_s`,
  `time_limited`; immovable assets require an `initial` pose; documented implicit
  `wall_mount`, walkway targets and the zone/wall policy (requested by WS-A).
- 0.2.0: additive. `ClearanceZone.allowed_categories`, `Asset.subtype`,
  `AssetSource.front_confidence`, `AssetSource.aspect_mismatch` (requested by WS-B).
- 0.1.0: initial contract.
