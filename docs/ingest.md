# RoomPlan ingest (WS-E, M4a)

`roomlayout.ingest.roomplan` turns an Apple RoomPlan export (the `CapturedRoom` JSON
that `JSONEncoder` writes, or a merged `CapturedStructure`) into a `LayoutProblem`
skeleton: room, box assets, and the scanned arrangement as `initial`. It imports
`roomlayout.schema` only. `roomlayout.pipeline` (ADR-0008) adds the category clearance
zones, rules, solving and rendering on top.

```bash
python -m roomlayout.ingest.roomplan Room.json --out problem.json --report ingest.json
python -m roomlayout.pipeline Room.json --out out/myroom --request "Sofa facing the TV"
```

```python
from roomlayout.ingest.roomplan import load

res = load("Room.json")  # .problem, .notes, .warnings, .source_ids, .floor_z
```

## Evidence

| what | source |
|---|---|
| `CapturedRoom` is `Codable`; fields `walls/doors/windows/openings/floors/objects/sections/story/version`; `Surface` and `Object` fields; attribute cases | Apple developer documentation (read 2026-10-17), `format.py` docstring |
| 16-float column-major transforms, enum encoding `{"door": {"isOpen": false}}`, `version` 2 | 12 public exports from GitHub repos (10 `CapturedRoom`, 2 `CapturedStructure`; surveyed, not committed) |
| floor `polygonCorners` x the floor transform lie on the wall segments | the same 12 exports, all aligned |
| object local axes: **+Z = front**, x = width, y = up (height) | the 12 exports: unambiguous wall-backed objects (4 chairs, 2 storage, 1 sofa: all +Z), chairs facing their table (19 of 33 +Z, the rest pushed aside), storage wider along x in 17 of 19 |
| attribute values (`ChairType` dining/swivel/stool, `TableType` coffee/dining, `StorageType` cabinet/shelf, `SofaType` lShaped/lShapedExtension/rectangular/singleSeat) | the 12 exports |

Not verified (handled defensively, warned): nested 4x4 transforms, string categories,
curved walls (`curve` was never non-null), iOS 16 exports (no floors), versions other
than 2. No export we have came from Voxelio or Room Scanner; the first one from the
owner's phone is the real test (M4a "real-capture validation").

## Mapping

- **Frame.** ARKit Y-up to ours Z-up at this boundary only (ADR-0001), a proper rotation,
  so winding is preserved. World xy is kept (no re-centring), so poses map straight back
  to the capture's camera frames. Heights are measured from the floor surface.
- **Floor.** RoomPlan floor `polygonCorners`, else the wall loop (open loops closed with
  a straight edge, noted or warned by gap size), else the convex hull of the wall ends
  (warned). Several floors: the largest (warned).
- **Walls.** One `Wall` per floor edge (`wall_1`... CCW from the south-west corner), with
  the height of the RoomPlan wall along it. Floor boundary with no wall scanned along it
  (≥ 0.8 m, the minimum walkway width) becomes a `passage` opening: open-plan sides stay reachable. Walls with floor
  on both sides become `partition_n` obstacles.
- **Doors, windows, openings** go on the floor edge nearest their centre (their parent
  wall preferred), with offset, width, height and sill; clipped to the wall if wider
  (warned). Door swing isn't in the data: assumed into the room, hinge at the wall start.
  Doors wider than 1.2 m are sliding/double/patio doors: no swing arc.
- **Objects** become box assets (`dimensions_from="roomplan"`, `geometry_from="box"`),
  ids `<subtype or category>_<n>`, subtype from attributes. Built-in fixtures (kitchen
  and bath appliances, fireplace) and built-in cabinetry (storage hung above 0.8 m, or
  touching a kitchen appliance or its cabinet run) are immovable; stairs are obstacles.
  `lShapedExtension` pieces merge into the sofa they touch (L or U footprint; cm gaps
  snapped). Storage carrying a TV becomes the `tv_stand`. A raised TV is wall-mounted
  only when its back is within 0.3 m of a wall.
- **Scan errors.** An object whose centre is outside the floor is pulled just inside
  (warned), unless it faces away from the room: then it's in the next room, seen
  through an opening, and is dropped (warned). Malformed elements are skipped (warned).
  Several stories, or a structure with several rooms and no `--room`, are errors.

Not mapped: `sections` (room labels), `confidence`, `completedEdges`, curved walls
(treated as straight), door `isOpen`.

## Validation on the 12 public exports

All 12 convert; floors, walls and openings line up with the raw RoomPlan walls (checked
visually). Findings that changed the mapping: built-in kitchen cabinetry and wall
cabinets are reported as plain `storage` (now fixed in place); wall-mounted TVs one room
over show up just outside the floor (now dropped); U-shaped sofas come as one piece
plus two extensions with cm gaps (now merged); fixtures cross the walls they're built
into (the solver now treats rules among immovable assets alone as facts, ADR-0008).
Solving results: [the M4a status entry in PROJECT_PLAN.md](PROJECT_PLAN.md).
