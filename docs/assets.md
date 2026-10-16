# Asset normalization (`roomlayout.assets`, WS-B / M2a)

Turns a raw generator mesh (SAM 3D Objects, or any glb/gltf/obj/ply/stl) into a
solver-ready `roomlayout.schema.Asset`, a GLB in the asset's canonical frame, a sidecar
report and front-candidate renders. CPU only. It imports `roomlayout.schema` and nothing
else from the repo, and never imports generator code (ADR-0004).

```
raw mesh ─► load + flatten ─► up-axis → Z-up ─► remove debris ─► level (≤ 8°)
        ─► align footprint rectangle to X/Y ─► choose front (4 hypotheses) ─► fit to W×D×H
        ─► footprint polygon ─► clearance zones ─► Asset JSON + GLB (Y-up) + report + PNGs
```

## CLI

```bash
python -m roomlayout.assets RAW_MESH --category sofa --dims 2.2 0.95 0.85 --out out/
    [--up y|z]                 # source up axis; default y (glTF, SAM 3D)
    [--id ID] [--label TEXT]   # default id: file stem; label: from category
    [--dims-from roomplan|listing|manual]   # provenance of --dims (default manual)
    [--geometry-from sam3d|object_capture|catalog|manual|box]   # default sam3d
    [--mount floor|wall] [--front A|B|C|D] [--mismatch-tol 0.15]
    [--max-footprint-vertices 32] [--footprint-tol 0.015] [--no-clearance] [--no-render]
```

Writes to `out/`:

| file | content |
|---|---|
| `<id>.asset.json` | `Asset`, validates against `roomlayout.schema.Asset`; `mesh.uri` is relative (`<id>.glb`) |
| `<id>.glb` | mesh in the asset's local frame, written in glTF convention (Y-up, front = +Z) |
| `<id>.report.json` | scale factors, mismatch flag, front hypotheses/scores/choice/confidence, debris, tilt, footprint stats, warnings, timings |
| `<id>.front_candidates.png` | contact sheet for a VLM/user: "which letter is the front?" |
| `<id>.front_{A,B,C,D}.png` | one straight-on view per candidate side |

Exit code 0 with warnings on stderr. Exit code 2 for bad input (missing/unreadable file,
non-positive dims). Python API: `normalize(...)` returns a `NormalizeResult` (asset, mesh,
report) and `write_outputs(result, out_dir)` writes the files.

To confirm a front after looking at the renders, re-run with `--front B`. That uses
hypothesis B and sets `front_confirmed: true`. It's deterministic, so the rest of the output
is identical.

## Frames (ADR-0001)

- Our frame: meters, right-handed, +Z up; asset front = local +Y, so the asset's right is +X.
- **Import** `--up y` (glTF/GLB, SAM 3D): `(x, y, z) → (-x, z, y)`; `--up z`: unchanged.
- **Export** to GLB applies the same matrix (it is its own inverse). So our +Z up becomes
  glTF +Y, our front +Y becomes glTF +Z (glTF's "front faces +Z"), and our +X becomes
  glTF -X (glTF's "right is -X"). `load_mesh(export_glb(m), up="y") == m` is tested.
- Source yaw conventions don't matter: SAM 3D's own Z-up→Y-up rotation, `(x, y, z) →
  (x, z, -y)`, differs from ours by 180° about up, and yaw is re-derived from geometry
  anyway. Only the up axis must be right.
- Evidence for SAM 3D being Y-up: `to_glb` in `sam3d_objects/model/backbone/tdfy_dit/
  utils/postprocessing_utils.py` ends with "rotate mesh (from z-up to y-up)". See
  `spikes/sam3d/README.md`.

## Steps and policies

**Debris.** Connected components (vertices welded by position, so UV seams don't split
parts). A component is dropped when its area is < 0.5 % of the total **and** its bbox
diagonal is < 10 % of the mesh's. The largest component is always kept. Legs and pedestals
survive (big diagonal). Specks go. A knob modelled as a separate island may go too:
visual-only loss. A warning fires when > 5 % of the area was removed.

**Levelling.** Generators can be a few degrees off vertical, which inflates the depth of
tall items: 2° on a 1.8 m shelf adds 6 cm to 35 cm. The support plane is the area-weighted
mean normal of the convex-hull faces facing within 8° of straight down (legs, base). We
rotate it onto -Z. Larger tilts are left alone with a warning, because that's a wrong up
axis rather than a tilt.

**Horizontal alignment.** Minimum-area rectangle of the projected convex hull, longer edge
→ X. This fixes yaw modulo 90°.

**Front (4 hypotheses).** Hypothesis A/B/C/D turns side +Y/+X/-Y/-X of the aligned mesh to
the front (yaw 0/90/180/270°). Per side we measure, from 20k surface samples:

| feature | meaning |
|---|---|
| `low` | top-surface height of the far half minus the near half, over the height: the seat side of a sofa/chair, the foot of a bed |
| `recess` | mean depth of the first surface seen from that side: open shelves, knee space, seat in front of a backrest; ~0 for a flat back panel |
| `long` | side length / longest side |
| `dims` | `-|log(w/d) - log(W/D)|`: the hypothesis's width/depth ratio vs. the measured one |

Weights per category (`categories.py`):

| group | low | recess | long | dims | rule |
|---|---|---|---|---|---|
| sofa / sectional / loveseat | 1.0 | 0.5 | 0.2 | 1.0 | seat side (lower) is the front |
| chairs | 1.0 | 0.5 | 0 | 1.0 | same; chairs are often deeper than wide |
| bed | 1.0 | 0.3 | -0.2 | 1.0 | foot end (headboard is higher) |
| storage (bookshelf, dresser, ...) | 0 | 1.0 | 0.5 | 1.0 | long side that is open/recessed |
| desk | 0 | 0.5 | 0.5 | 1.0 | long side with knee space |
| table, TV, appliances, unknown | 0 | 0 | 0.3-0.5 | 1.0 | long side; 180° unresolved |

Confidence = `clip((best - second) / 0.25, 0, 1)`. Below 0.5 there's a warning, and the
renders must decide. `Asset.front_confirmed` is only true with `--front`.

The `dims` term dominates for long items, so a user who swaps W and D would silently turn
the front by 90°. We therefore also rank without it. When geometry alone is confident
(≥ 0.5) and points to the other axis, the report sets `front.dims_conflict` and warns
"measured width/depth may be swapped". No false positives in 48 synthetic runs with the
correct dims.

Known failure modes of the front heuristic:
- Symmetric objects (tables, TVs, cabinets with flat doors, ottomans): confidence ≈ 0 by
  construction, and that's correct.
- Seating with a high front (chaise end of a sofa photographed as a separate piece,
  daybeds), backless benches, beds without a headboard: `low` is ~0 or misleading.
- Storage with closed flat doors on the front and a flat back: `recess` can't tell them
  apart. Handles are usually below the 64³ resolution.
- Generator hallucinates a backrest on the unseen side, or fuses the back into a slab: the
  heuristic follows the geometry, so the front is wrong if the geometry is wrong.
- Corner sofas with backs on two sides: `low` favours one of the two open sides. Check the
  renders.

Tested on synthetic L-sectional, chair, bookshelf in random yaw/scale/offset, with debris,
noise and a 2.5° tilt (100 % correct). Real-photo accuracy is measured in the SAM 3D spike
(EVAL.md).

**Fit to measured dimensions (ADR-0004: measured dims always win).** The mesh is scaled per
axis so its bbox equals W×D×H exactly. The report gives `scale = (W/w, D/d, H/h)` and
`aspect_mismatch = max_i |s_i / s_geo - 1|` (s_geo = geometric mean). 0 means the generator's
proportions were right, and 0.15 means one axis was stretched 15 % relative to the others.
Above `--mismatch-tol` (default 0.15) the report is `flagged`, with a warning (not a
failure): show it to the user and offer a retake. `up_check` also flags when treating
another mesh axis as vertical would cut the mismatch by more than half.

No dims: the mesh keeps the generator's scale (not metric for SAM 3D), the Asset gets
`source.dimensions_from = "mesh"`, and the first warning starts with `DIMENSIONS NOT
MEASURED`. The CLI prints it on stderr. Downstream should refuse or visibly mark such assets.

**Footprint** (`footprint.py`).
1. Rasterize all triangles' XY projections (vertical faces too, drawn as lines) at
   longest side / 400, clamped to 2-10 mm. The raster contains the true projection and
   overshoots by under one pixel, which is conservative.
2. Trace the exact pixel boundary (union of row runs).
3. Morphological closing (1 cm).
4. Drop parts < 1 % of the area. **If several parts remain, take their convex hull**
   (conservative: nothing can be placed between the pieces). A chair's legs never get here
   because the seat covers them in projection.
5. Fill holes.
6. Clip to the W×D box.
7. Douglas-Peucker simplify at 1.5 cm, growing the tolerance until there are ≤ 32 vertices.
   Vertices are a subset of the clipped boundary, so it never exceeds the bbox.
8. Snap to the exact rectangle when it covers ≥ 98 % of the bbox.

The report gives the vertex count, area, bbox fill and the Hausdorff distance between the
raw traced region and the final polygon. The simplification can cut convex curves inward
by up to the tolerance (a 1.2 m round table loses ~1.3 % area at 16 vertices).
An exact shapely union of triangles was ~10 s for 80k faces; the raster is < 1 s.

**Clearance zones** (`categories.py`, `zones.py`). A zone of depth `d` on a side is the
footprint swept `d` outward minus the footprint (Minkowski sum with a segment). For an
L-sectional the legroom therefore covers the notch in front of the main seats, which a
bbox-strip would miss. Disjoint pieces become separate zones (`legroom`, `legroom_2`).

## Category defaults

`Asset.category` is the RoomPlan category (`CapturedRoom.Object.Category`: bathtub, bed,
chair, dishwasher, fireplace, oven, refrigerator, sink, sofa, stairs, storage, stove, table,
television, toilet, washerDryer; verified against Apple's docs 2026-10-16). The finer type
goes into `label` and `report.fine_type`. Aliases like couch, bookcase or L-shaped sofa
are accepted.

| fine type | RoomPlan | zones (depth, purpose) | rationale (rule of thumb) |
|---|---|---|---|
| sofa, sectional, loveseat | sofa | legroom front 0.40 (use) | 14-18 in (35-45 cm) sofa-to-coffee-table |
| chair, armchair | chair | legroom front 0.40 (use) | as sofa; generic RoomPlan `chair` = lounge chair |
| dining_chair, office_chair | chair | pull_out back 0.45 (use) | about half of the 36 in table-edge allowance |
| stool | chair | – | |
| bed | bed | access left/right 0.60, foot 0.60 (access) | ≥ 24 in (60 cm) around a bed |
| table, coffee_table, side_table | table | – | a zone would push the sofa away |
| dining_table, round_table | table | chairs 0.75 on all 4 sides (use) | 30-36 in from table edge (NKBA: 32-36 in) |
| desk | table | chair_pull_out front 0.90 (use) | ~36 in chair space |
| storage, bookshelf, cabinet, sideboard | storage | access front 0.60 | stand and reach |
| tv_stand | storage | access front 0.50 | |
| dresser | storage | drawers front 0.90 | drawer pull-out + person |
| wardrobe | storage | doors front 0.90 (swing) | door swing + person |
| nightstand | storage | – | |
| television | television | **none** | the viewing area isn't a keep-out zone; use `faces` |
| refrigerator, oven, stove, dishwasher, washerDryer | same | door_access front 0.90 (swing) | door + standing room |
| sink 0.75, toilet 0.60, bathtub 0.75, fireplace 0.50, stairs 0.90 | same | front (use/access) | NKBA / residential-code minimums |

These are conventional design guidance and NKBA planning guidelines. They weren't
re-verified against the primary texts in this session. Tune them in one place:
`categories.PROFILES`. Fixtures (oven, sink, ...) default to `movable: false`.

## Limitations

- Upright mesh assumed up to 8° of tilt. A sideways or upside-down generator output is
  *flagged* (`up_check`), not fixed.
- Front heuristic is geometry-only (see failure modes). VLM confirmation isn't wired up yet.
- Single footprint for the whole height. Chair-under-table needs height-banded footprints
  (deferred in ARCHITECTURE.md).
- Dining-table / desk zones conflict with separately placed chairs (no exemption mechanism
  in the schema). See the schema change request in the WS-B report.
- Zones can extend through walls when an asset stands against one. The solver should clip
  zones to the floor before any reachability test.
- No decimation of the output mesh. SAM 3D raw meshes are passed through as-is, apart from
  debris removal.
- Textures: SAM 3D's default output uses vertex colours, which are preserved. Textured
  inputs keep their UVs/material through the pipeline (trimesh), but this isn't tested.

## Next steps

1. **VLM front confirmation**: send `<id>.front_candidates.png` with the question printed on
   it, parse a letter, re-run with `--front`. Use it whenever confidence < 0.5, or always if
   the spike shows < 90 % heuristic accuracy.
2. **RoomPlan box ingestion**: take W×D×H and the box yaw from `CapturedRoom.Object`
   (`dimensions`, `transform`) as `--dims-from roomplan`. The box yaw can also disambiguate
   the front against the scan frame's viewing direction.
3. Height-banded footprints (seat vs. under-table space) once the solver supports them.
4. Output decimation and a convex-decomposition hook (CoACD) for the physics stage.
