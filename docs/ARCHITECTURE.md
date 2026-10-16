# Room Layout Pipeline — Architecture

Goal: from photos/video of a user's room and photos of their furniture, produce a
small set of furniture layouts that satisfy the user's functional and aesthetic
rules, are physically valid (no interpenetration, stable), and can be viewed in a
photoreal reconstruction of the room.

**Output decision (affects Stage 6):** the primary deliverable is a *validated
layout* rendered in the reconstructed room. A fully simulatable "interactive twin"
is a secondary goal. Physics is therefore used for validation, not as the search
engine.

---

## Pipeline overview

| # | Stage | Input | Core engine | Output |
|---|-------|-------|-------------|--------|
| 0 | Capture & metric frame | Room video (+ LiDAR depth if available) | SfM / feed-forward pose estimation, IMU gravity, metric scale | Posed images in a metric, Z-up world frame |
| 1a | Environment — visual | Posed images | 3DGS + **GOR-IS** object removal | Photoreal empty-room splat + estimated lighting |
| 1b | Environment — geometry | Posed images / depth | Floor & wall plane fitting (or 2DGS / GOF mesh extraction) | Floor polygon, walls, ceiling height, doors/windows as primitives; room collision geometry |
| 2 | Asset generation | Furniture photos | **SAM 3D Objects** (single image) · **Apple Object Capture** (multi-view) · LGM (fallback) | Textured visual mesh per item |
| 3 | Asset processing | Visual meshes + real dimensions | Scale/orient, cleanup, **CoACD / V-HACD** | Metric, canonically oriented asset with convex collision parts, footprint, front vector, clearance zones, mass |
| 4 | Constraint extraction | User rules (text) + asset list + room summary | **VLM**, one call, schema-constrained output + deterministic rule library | Validated, typed constraint graph (hard + weighted soft) |
| 5 | Layout search | Constraint graph + room polygon + asset footprints | 2.5D oriented-footprint search: simulated annealing / CMA-ES (+ CP-SAT for discrete choices), gradient refinement | Top-k (≈10) *diverse* feasible layouts |
| 6 | Physics validation | Top-k layouts + collision parts | Static collision query, then short **MuJoCo** settle | Validated layouts; failures fed back to Stage 5 |
| 7 | Render & rank | Validated layouts + splat + meshes | Composite render, optional VLM critic on ≤10 renders | Final ranked layouts for the user |

```
 room video ─► [0 metric frame] ─┬─► [1a 3DGS + GOR-IS] ───────────────────────────┐
                                 └─► [1b room geometry] ──┐                        │
 furniture photos ─► [2 assets] ─► [3 process] ───────────┤                        │
 user rules ─────────────────────► [4 VLM → graph] ───────┤                        │
                                                          ▼                        ▼
                                              [5 layout search] ◄─┐      [7 render & rank]
                                                          │       │ failures       ▲
                                                          ▼       │                │
                                              [6 physics validation] ──────────────┘
```

---

## Stage 0 — Capture & metric frame

Everything downstream assumes one shared, metric, gravity-aligned coordinate frame.
The original design skipped this step, but every later stage depends on it.

- Camera poses: COLMAP, or a feed-forward model (MASt3R / VGGT class) for speed.
- **Metric scale** (SfM is scale-ambiguous), in order of preference:
  1. Phone LiDAR / ARKit depth (iPhone Pro, iPad Pro): metric directly.
  2. ARKit/ARCore session poses (VIO is metric).
  3. A user-measured reference (e.g. "this wall is 4.2 m"), or a known-size object.
- Gravity: IMU or floor-plane normal → Z-up. Floor plane at z = 0.

## Stage 1a — Environment, visual (3DGS + GOR-IS)

- Train 3DGS on the posed images.
- Segment existing furniture to remove (SAM 2/3 masks lifted to Gaussians).
- Remove with **GOR-IS** ([arXiv 2605.00498](https://arxiv.org/abs/2605.00498),
  CVPR 2026). It decouples material from lighting and inpaints in intrinsic space,
  so shadows and non-Lambertian reflections of removed objects are handled better
  than appearance-space inpainting.
- **Scope of what GOR-IS gives us:** better *visual* removal (the paper reports
  gains in LPIPS/PSNR). It does **not** produce collision geometry, and residual
  artifacts are expected on real captures. Treat it as a rendering backdrop.
- Keep its estimated lighting; Stage 7 can use it to relight inserted assets.
- Fallback if GOR-IS code is unavailable or too slow: Gaussian Grouping / GScream-
  style removal + 2D inpainting; accept baked shadows.

## Stage 1b — Environment, geometry

The original design gave furniture collision meshes but left the room itself
without any. Physics needs a floor and walls to settle against.

- Primary: fit planes (RANSAC on depth/points) → floor polygon, wall segments,
  ceiling height. Represent the room as boxes/planes. This is robust, convex by
  construction, and is what the layout solver wants anyway.
- Detect doors, windows, radiators, and outlets (2D detector + lift) → keep-out /
  clearance zones and anchor targets.
- Optional: 2DGS / Gaussian Opacity Fields mesh extraction for non-planar rooms
  (bay windows, columns), followed by convex decomposition like any asset.

## Stage 2 — Asset generation

Pick the generator by the input the user can provide:

| Input available | Generator | Notes |
|---|---|---|
| 1 photo (e.g. product image) | **SAM 3D Objects** (facebookresearch/sam-3d-objects) | Single-image shape + texture, robust to occlusion. Also predicts object pose/scale relative to camera (useful prior, still not metric). |
| 20–100+ photos around the object | **Apple Object Capture** (RealityKit photogrammetry, macOS) | Highest fidelity and true geometry. Requires a Mac in the loop and a deliberate capture. |
| Few views, no Mac | LGM | Multi-view → Gaussians → mesh. Mesh conversion is lossy; fallback only. |
| Known product SKU | Retrieve a catalog / Objaverse-style "digital cousin" | Often better than generating; cuts cost. |

Meta 3D Gen is text-to-3D and is not used for reconstruction.

## Stage 3 — Asset processing

1. **Metric scale:** user-entered W×D×H, product-listing dimensions, or
   measurement from the room capture if the item is already in the room. Scale
   the mesh anisotropically only if the error is small; otherwise flag it.
2. **Canonical orientation:** upright (+Z) and a "front" vector (+Y). Use PCA +
   gravity heuristics, with a VLM tie-break on renders ("which side is the front
   of this sofa?").
3. **Cleanup:** trimesh / PyMeshLab to remove floating islands and degenerate
   faces, merge vertices, and decimate the *visual* mesh to a budget.
4. **Collision:** MuJoCo collides each mesh geom as its **convex hull**, so a
   non-convex asset (an L-sofa, a chair) must be split into convex parts. Use
   **CoACD** (better preserves concavities like seats/under-table space) with
   **V-HACD** as fallback. Both tolerate non-manifold input, so cleanup is for
   quality, not crash prevention. Cap parts (~8–32 per asset).
5. **Metadata for search:** oriented footprint polygon, height, front vector,
   functional clearance zones (chair pull-out, drawer/door swing, seating legroom),
   wall-placeable flag, stackable flag, and estimated mass/inertia (category density
   × volume) plus friction.

## Stage 4 — Constraint extraction (VLM, once)

- One VLM call per user request. Inputs: user rules, asset list with metadata,
  room summary (dimensions, doors, windows, a top-down render).
- Output is constrained to a **fixed JSON schema** (structured output). The VLM
  chooses from a closed vocabulary of predicates; it does not invent new ones:
  - Relations: `faces(a,b)`, `against_wall(a)`, `near(a,b,max_d)`,
    `distance_ge(a,b,d)`, `aligned(a,b)`, `centered_on(a, wall|window)`,
    `in_zone(a, zone)`, `parallel(a,b)`, `not_blocking(a, door|window)`.
  - Each constraint is `hard` or `soft` with a weight.
- **Deterministic rule library** is always added and not left to the VLM:
  walkway ≥ 0.9 m, door swing clear, no blocking of windows/radiators, and asset
  clearance zones kept free.
- Validation: schema check, all referenced objects exist, a quick infeasibility
  check (e.g. total footprint vs. floor area, conflicting hard constraints). Show
  the user the parsed constraints for confirmation when the VLM's output is unclear.

## Stage 5 — Layout search

- Model the layout as **2.5D**: each object has (x, y, yaw) on the floor, or is
  mounted on a wall/stacked on a parent. Collision in search uses oriented footprint
  polygons (+ clearance zones), not the convex parts.
- Objective: hard constraints as feasibility (or large penalties); soft constraints
  as a weighted cost.
- Solver (layered):
  1. **Discrete choices** (which wall a sofa goes against, room zones, yaw
     snapping to 0/90/180/270°): CP-SAT, or enumerate when small.
  2. **Continuous placement:** simulated annealing or CMA-ES over (x, y, yaw), with
     many parallel restarts. This handles non-convex, non-differentiable overlap
     constraints, which plain gradient descent does not.
  3. **Local refinement:** gradient descent on a smoothed penalty to polish
     alignment/spacing only.
- Return the **top-k diverse** layouts: deduplicate by layout distance so the user
  doesn't get ten near-identical sofa positions.
- Expected cost: ~10³–10⁴ candidate evaluations per second per core in vectorized
  NumPy/JAX for typical rooms (5–20 objects). Budget seconds, not milliseconds.

## Stage 6 — Physics validation (top-k only)

1. **Static check first:** load the room primitives + convex parts in MuJoCo (or
   FCL) and run a collision query at the proposed poses. Any penetration > δ
   fails. This is cheap and catches most problems.
2. **Short settle (MuJoCo):** drop objects from a few mm, simulate ~1–2 s. Pass if
   max displacement < ε, no tip-over (up-vector stays within θ), and stacked items
   stay put. Settling matters mainly for stacked/wall-leaning items and tall,
   tippy furniture. For pure floor placement, the static check is usually
   sufficient.
3. **Closed loop:** each failure becomes feedback to Stage 5 (an added exclusion
   or penalty for the offending pair/pose), then re-search. The loop is bounded
   (e.g. 3 rounds). If still short of k, return what passed and report why.

## Stage 7 — Render & rank

- Composite the validated meshes into the GOR-IS-cleaned splat. Inserted assets
  won't match scene lighting by default; relight them using GOR-IS's estimated
  lighting where feasible (stretch goal), at minimum add contact shadows.
- Optional **VLM critic** on the ≤10 final renders for aesthetic ranking and
  explanations. Inference cost stays bounded because only ≤10 renders are scored.

---

## Corrections vs. the previous draft

| Previous claim | Correction |
|---|---|
| SAM 3D only does segmentation | SAM 3D Objects (Nov 2025) is a single-image 3D reconstruction model; it's now a primary asset generator. |
| Generative meshes "fatally crash" MuJoCo | MuJoCo uses each mesh geom's convex hull. The real problem is *wrong collision shape* for concave objects. Convex decomposition fixes this, and it tolerates non-manifold input. |
| GOR-IS yields a "mathematically viable empty volume" | GOR-IS improves *visual* removal. Collision geometry for the room comes from Stage 1b. |
| Bounding-box gradient descent finds layouts | Overlap/clearance constraints are non-convex and partly discrete. The plan uses annealing/CMA-ES + CP-SAT, with gradient descent only for refinement. |
| "Thousands of floor plans in milliseconds" | Realistic budget is seconds for a good, diverse top-k. |
| MuJoCo gravity settling for every layout | Static collision check first; dynamics only where stability matters. |
| *(missing)* | Added Stage 0 metric frame, asset scale/orientation, room collision geometry, deterministic rule library, and the validation → search feedback loop. |

## Risks & open questions

- **Asset scale accuracy:** single-image reconstruction is not metric. Without
  user/listing dimensions, layouts can be confidently wrong. Require dimensions,
  or measure them in the room scan.
- **GOR-IS availability/runtime:** this is recent research. Confirm code release
  and per-scene training time before committing; keep the fallback.
- **Splat + mesh compositing realism:** lighting mismatch is the most visible
  artifact. It's a polish item, not a correctness one.
- **Non-planar rooms:** plane fitting fails on curved walls; the mesh-extraction
  path is needed there.
- **VLM constraint fidelity:** ambiguous phrasing ("cozy", "open feel") needs
  mapping to soft constraints. Log and evaluate this mapping.

## Suggested build order

De-risk the core value (layout search) before the expensive perception work.

1. **M1 — Solver core:** hand-authored room polygon + box assets + hand-written
   constraint JSON → Stage 5 + Stage 6 static check. Proves layouts are good.
2. **M2 — Constraints:** Stage 4 VLM → schema, plus the rule library and an eval set
   of user prompts.
3. **M3 — Assets:** Stages 2–3 with SAM 3D Objects + CoACD; MuJoCo settle.
4. **M4 — Room:** Stages 0 + 1b (ARKit capture → planes) replace the hand-authored room.
5. **M5 — Visual:** Stage 1a 3DGS + GOR-IS, Stage 7 compositing and VLM critic.
