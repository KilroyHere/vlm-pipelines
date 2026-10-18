# Results: end-to-end on 3 real living rooms (ARKitScenes), 2026-10-17

**Verdict: the whole chain runs on real LiDAR rooms with generated furniture, and the
layouts make sense.** Every room got feasible layouts that satisfy the user's whole
request; all layouts pass our sensibility checks except one flag on a cabinet the user
pinned ("don't move it"). Two rule-library bugs surfaced and are fixed (ADR-0007
amendment). Asset quality from in-room video frames is clearly below the product-photo
pilot, and the capture step is the main reason.

Rooms: ARKitScenes validation captures 42445441 (3×4 m), 47204607 (5.5×4.5 m), 48018379
(3.5×6 m). How it was run: [README.md](README.md).

## End to end (SAM 3D assets, request -> constraints -> solve -> checks)

| room | assets (SAM 3D / box) | real room: user reqs met | real room: checks failed | layout 1: feasible | layout 1: user reqs met | layout 1: checks failed | layouts passing all checks | mean move vs real (m) |
|---|---|---|---|---|---|---|---|---|
| 42445441 | 8 / 0 | 2/3 | 4 | yes | 3/3 | 0 | 3/3 | 1.27 |
| 47204607 | 7 / 1 | 3/5 | 1 | yes | 5/5 | 1 * | 0/3 * | 2.85 |
| 48018379 | 6 / 4 | 3/5 | 2 | yes | 5/5 | 0 | 1/1 | 2.96 |

\* the only failure in all three layouts is `storage_against_wall[cabinet_1]`: the user
said "don't move the cabinet", and it really stands 0.27 m off the wall (a radiator
cover). 48018379 returns one layout: the top-k quality gate dropped the costlier ones.
With box assets instead (`out/boxes`) the layout columns are the same (feasible, user
requests 3/3, 5/5, 5/5, same check results); the real room fails 5 / 2 / 2 checks (box
footprints are larger than the generated shapes) and mean move is 2.48 / 2.64 / 2.52 m.
Solve time 20 s budget per room.

The real arrangements fail our checks too (4, 1, 2 failures; e.g. storage 0.2-0.5 m off
the wall, a TV stand turned away from the sofa) and fail our hard rules on centimetre
overlaps from the annotation boxes. Two readings: annotation slop, and our clearance
zones being stricter than how people actually live. Worth a look before we tune further.

"Mean move" is large because no constraint asks to keep things in place: these are
"rearrange my room" layouts, not "tidy my room" layouts. A "minimal change" mode would
be a `near(asset, initial pose)` family or a solver term; not in the vocabulary today.

Pictures: `out/sheet_sam3d_<room>.png` (real vs layout 1, top-down and 3D with the
generated meshes; not committed, regenerate with `summarize.py`).

## Asset generation from the capture video

21 of 26 objects got a SAM 3D asset; 5 stayed boxes (product fallback): 4 had no frame
showing at least half of them, 1 had only a frame of the wall in front of it.

| | pilot (product photos, 7 items) | this run (capture frames, 21 items) |
|---|---|---|
| proportion mismatch vs measured dims, mean | within 15% for all | 21%; 11/21 over the 15% flag |
| shape, my rating | 7/7 usable, mean 4.0/5 | 16 good, 4 partial, 1 wrong |
| SAM 3D time per item (A40) | 14-20 s | median 15 s (11-43 s), peak VRAM 20.6 GB |

Partial: two tall shelving units seen in one frame both came out as the merged pair; a TV
shelf with the TV in the mask; a TV stand seen steeply from above reads as a cabinet.
Wrong: the round coffee table, base hidden by the rug, became a slab on two legs (its
top-down footprint is still round, so the layout is right). Fronts were not measured
here (no ground-truth front in ARKitScenes; the pilot measured 6/6).

Masks: SAM 2 prompted with the scan box projected into the frame worked for 19/21
without any text detector; 1 grabbed the rug under a coffee table (fixed with the OWLv2
text detector), 1 was the wall (item dropped). A product check should compare the mask
with the projected box and fall back to the detector when they disagree.

## What this means for the product

1. **Capture needs a step-back shot per large object.** A scanning walk passes close to
   furniture: big sofas were at most 39-58% visible in any frame. RoomPlan's object list
   can drive a "now take one photo of the sofa from across the room" step.
2. **Frame choice by geometry works.** Camera pose + scan box picks a usable frame and
   gives a mask prompt for free; this is the default path, not the text detector.
3. **The rule library had to change for real rooms** (ADR-0007 amendment): soft user
   requests now weigh 15 (were 1 vs rules at 5), and "sofa faces TV" no longer cancels
   "coffee table in front of the sofa". Before the fix, 2 of 9 box layouts passed all
   checks; after, 7 of 7.
4. **Still untested:** doors and windows (not annotated in ARKitScenes, so door rules
   never ran), a live VLM call (answers hand-written in its output format; no API key),
   and the RoomPlan importer itself (the user's own scan will test it).

## Cost

One A40 Secure pod (`v39xw1o6z9hkyu`, $0.59/h) from 14:52 to ~15:57 UTC, terminated:
~$0.65 (the billing read at 16:15 showed $0.22 and lags). Most of the hour was the
environment build on a fresh volume (pytorch3d compile); generation itself took 7 min (416 s for 21 items).
The bootstrap's last bug (roomlayout installed into the conda env instead of its venv)
is fixed, so the next run can normalize on the pod.

## Furnishing an emptied room with new furniture (ADR-0009), 2026-10-18

`furnish_demo.py`: each room is emptied of its scanned furniture
(`pipeline.empty_room`), then refurnished with the pilot's SAM 3D assets made from
retailer product photos (manufacturer sizes). Request, the same for every room: *"Put the
sofa against a wall with the coffee table in front of it, and the armchair by the
coffee table so we can talk. I need a small work spot: the desk against a wall with the
chair at it. The dresser goes against a wall too."* (VLM answer hand-written in its
output format, as above; 9 constraints, 6 without the work spot). Solver: 30 s, k=3.

| room (free floor) | variant | furniture footprint (m²) | fits | request met (layout 1) | layouts passing all checks |
|---|---|---|---|---|---|
| 47204607 (21.05 m²) | sectional | 5.85 | yes | 9/9 | 3/3 |
| 47204607 (21.05 m²) | sofa | 4.92 | yes | 9/9 | 3/3 |
| 47204607 (21.05 m²) | sofa, no desk | 3.98 | yes | 6/6 | 3/3 |
| 48018379 (18.11 m²) | sectional | 5.85 | yes | 9/9 | 2/2 |
| 48018379 (18.11 m²) | sofa | 4.92 | yes | 9/9 | 3/3 |
| 48018379 (18.11 m²) | sofa, no desk | 3.98 | yes | 6/6 | 3/3 |
| 42445441 (10.99 m²) | sectional | 5.85 | no | 7/9 | 0/3 |
| 42445441 (10.99 m²) | sofa | 4.92 | no | 9/9 | 0/3 |
| 42445441 (10.99 m²) | sofa, no desk | 3.98 | yes | 6/6 | 1/1 |

Variants are sets the user is choosing between: sectional = KIVIK sectional, armchair,
coffee table, desk, chair, dresser; sofa = the same with a 3-seat sofa instead of the
sectional.

The small room is the useful case: the solver says the sectional set doesn't fit
("footprints plus clearance zones need about 9.85 m² of 10.99 m² free floor (90%)"), the
sofa set doesn't either (no walkway reaches everything), and the set without the work spot
does. That is the product's answer to "will it fit?". Contact sheet:
`out/furnish/demo_sheet.png` (not committed).
