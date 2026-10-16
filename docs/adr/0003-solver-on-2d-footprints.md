# ADR-0003: The layout solver works on 2D footprints, never meshes

- Status: accepted
- Date: 2026-10-16

## Context
Layout search must evaluate thousands of candidates per second. Meshes are heavy, and
most floor-furniture validity questions (overlap, clearance, walkways) are 2D.

## Decision
- The solver sees each asset as a local-frame footprint polygon + clearance zones +
  height (height only matters for window sills).
- Search is simulated annealing over (x, y, yaw), with yaw snapped to multiples of
  90° by default, many restarts, and diversity filtering of the top-k.
- 3D validation (MuJoCo, convex decomposition) is deferred until stacking or a
  simulatable twin is in scope.

## Consequences
- The asset pipeline's job is to produce an accurate footprint. Mesh quality affects
  looks, not validity.
- Tucking a chair under a table (height-dependent footprints) is out of scope for v1.
