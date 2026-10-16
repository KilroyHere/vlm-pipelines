# ADR-0004: Asset generators run out-of-process; size never comes from the generator

- Status: accepted
- Date: 2026-10-16

## Context
SAM 3D Objects needs a large NVIDIA GPU, has its own dependency stack and license,
and single-image reconstruction is not metric. We may swap generators later (Object
Capture, catalog retrieval).

## Decision
- Generators run as separate processes (a CLI on a GPU host for now, a service later)
  that take `image + mask` and write a raw mesh. We never import their code.
- Everything after the raw mesh is our CPU code in `roomlayout.assets`: canonicalize
  (Z-up, bottom-centre origin), fit to measured `Dimensions`, choose the front,
  extract the footprint, and export a GLB plus an `Asset` record.
- **Measured dimensions (RoomPlan box, listing, user entry) always win over the
  generator's scale.**

## Consequences
- The asset pipeline is fully testable on CPU with synthetic meshes.
- The GPU host is a single replaceable component, so the spike's go/no-go is about
  output quality only.
