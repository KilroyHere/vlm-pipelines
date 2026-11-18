# Web viewer (`roomlayout.viewer`)

A static three.js page that shows a solved room. It replaces the matplotlib renders
for review and is the first app surface (ADR-0010).

```bash
# after python -m roomlayout.fit ... --out out/fit_demo (or any problem + result pair)
python -m roomlayout.viewer export out/fit_demo/problem.json out/fit_demo/result.json \
    --fit out/fit_demo/fit.json --out out/viewer/demo [--meshes DIR] [--request "..."]
python -m roomlayout.viewer serve out/viewer/demo --port 8000   # http://127.0.0.1:8000/
```

The page can be opened as a plain static page, which shows layouts and explanations. Served
with `serve`, it can also edit (below). three.js 0.160.0 is loaded from jsDelivr, so
the page needs network access.

## What it shows
- **The room as scanned.** Floor with a plank texture. Walls built in sections around
  their openings, with window glass and frames, door frames and swing arcs. Obstacles
  (stairs, partitions) are drawn too. Walls nearest the camera drop to a short stub
  (dollhouse cutaway). "Top" switches to a plan view.
- **Furniture.** A GLB from `roomlayout.assets` when `--meshes` has `<asset_id>.glb`.
  Otherwise a tidy procedural model inside the listing's box: sofa (seat, back, arms,
  cushions), L-sectional (from the L footprint, chaise on its side), bed (frame,
  mattress, headboard, pillows), table and desk (top, legs), chair, open shelving,
  cabinet, TV. Colour comes from words in the label ("gray", "ash", "white"), else the
  category.
- **Lighting.** Image-based (RoomEnvironment) plus a soft-shadowed sun, ACES tone
  mapping.
- **Progressive loading.** The procedural model appears at once. A GLB fades in when it
  has loaded, and the procedural model fades out.
- **Layouts.** Tabs switch between the top-k layouts, and the pieces glide to their new
  places.
- **Why this layout** (`roomlayout.explain`). Measured reasons, user requests first:
  "The sectional faces the TV (4° off), 2.62 m away", "Every door and piece is reachable
  along paths at least 1.2 m wide". Soft wishes a layout misses are marked "!" with
  what was wanted. Broken hard rules are marked "✕". Hovering a line highlights its
  pieces.
- **Fit verdict.** "It fits" with furniture vs free floor area, or "It doesn't fit" with
  the advice from `roomlayout.fit`.

## Editing (served only)
- **Drag** a piece on the floor; **Q / E** or the buttons rotate it by 15°. The edit is
  checked at once (`POST /api/evaluate`) and appears as a "Your edit" tab, with its own
  explanation.
- **Lock** (L) the pieces you like, then **Re-arrange around locked pieces**
  (`POST /api/solve`). Each locked piece becomes a hard `fixed` constraint at your pose
  (origin "user"), and the solver returns new top-3 layouts around it.
- A lock that puts a piece through a wall or into an obstacle is refused with a message.
  Locked pieces are facts to the solver (ADR-0008), so it would not catch this itself.

## `scene.json`
Version 1. `room` (floor polygon, walls with compass names, openings with offset, width,
height, sill and swing, obstacles), `assets` (dims, local footprint, `l_shaped`,
elevation, mount, `mesh`), `pinned` (fixture poses), `layouts` (placements, feasibility,
cost, explanation lines), `fit`, `request`. Coordinates are ours (meters, Z-up, yaw
CCW). The page maps (x, y, z) to three.js (x, z, -y).

## Checked
- `tests/test_viewer.py`: scene export, page files, lock checks, and the server's
  ping/evaluate/solve on a real local port.
- Browser (Playwright + SwiftShader, 2026-11-18), no console errors:
  - 3D and top views of the Apple sample living room and bedroom;
  - the full edit loop: select, drag 0.4 m, "Your edit breaks a rule", lock, re-solve
    gives 3 layouts with the bed where it was dropped;
  - a GLB from `normalize` replacing the procedural sectional at the same pose and
    orientation.

## Not yet
- Dragging doesn't snap to walls and can't collide. The check only runs on drop.
- No undo. Edits are lost on reload.
- Headless-Blender stills and warm-start re-solves are later work.
