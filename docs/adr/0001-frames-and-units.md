# ADR-0001: One world frame, meters, Z-up; assets face +Y

- Status: accepted
- Date: 2026-10-16

## Context
Inputs come from systems with different conventions: RoomPlan/ARKit (Y-up), glTF
(Y-up), SAM 3D output (generator-specific), trimesh/most robotics tooling (Z-up).
Mixed frames are the most common source of silent integration bugs.

## Decision
- World: right-handed, meters, +Z up, floor at z = 0, yaw CCW about +Z.
- Asset local frame: origin at the bounding-box footprint centre on the asset's
  bottom; +Y is the asset's functional front.
- Conversion happens at ingest boundaries only (RoomPlan ingest, mesh
  normalization). Everything inside the pipeline is in this frame.

## Consequences
- The solver and asset code share one set of geometry helpers (`Pose2D`).
- GLB export must convert back to glTF's Y-up at write time. This is the asset
  workstream's responsibility, and it's covered by a round-trip test.
