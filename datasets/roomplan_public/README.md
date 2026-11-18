# Public RoomPlan scans

25 single-room Apple RoomPlan `CapturedRoom` exports found on GitHub. They stand in for
the owner's own scan (ADR-0010).

```bash
python datasets/roomplan_public/fetch.py        # -> data/roomplan_public/<id>.json (gitignored)
python spikes/fit_baseline/run.py               # emptied + furnished with the 5 test products
```

**Not committed, research use only.** Apart from `edwardzke/heyday` (MIT) and
`AvnehSBhatia/ExeaLabsNvidiaHacks` (Apache-2.0), the source repos carry no license file.
The Apple "MyHome" rooms are captures from Apple's WWDC23 sample code, mirrored on GitHub.
`manifest.json` records repo, path, the file's git blob SHA-1 (`fetch.py` verifies it)
and the license we found.

## How they were chosen (2026-11-18)
1. GitHub code search for RoomPlan JSON keys (`completedEdges`, `coreModel`,
   `polygonCorners`, ...) gave 208 candidate files in about 40 repos.
2. All of them went through `roomlayout.ingest.roomplan`. 160+ convert. Merged
   multi-room `CapturedStructure` files are refused (v1 plans one room).
3. Hand-made test fixtures were dropped: rooms where 30% or more of the wall lengths
   are round to the centimetre (real captures have sub-millimetre noise), plus files
   under `fixtures/generated/`.
4. Duplicates were dropped. The Apple sample rooms appear in several forks; only one
   copy is kept.
5. Kept: residential rooms (bedrooms, living rooms, dining, open plan, a few empty
   scans), plus one fixture-heavy kitchen, a classroom and a large hall as stress cases.
   Corridors, bathrooms and slivers under 3 m² were left out.

`kind` in the manifest is our reading of the room (bedroom, living, dining, ...), not a
RoomPlan label.
