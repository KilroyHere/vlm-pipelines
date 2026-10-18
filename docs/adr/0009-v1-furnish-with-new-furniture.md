# ADR-0009: v1 furnishes an emptied room with new furniture from product photos

- Status: accepted (owner decision, 2026-10-18)
- Date: 2026-10-18

## Context
The v1 plan rearranged the user's *own* furniture, modelled from the scan video. The
ARKitScenes end-to-end run (spikes/e2e_arkitscenes/RESULTS.md) showed that this is the
weakest link: assets from capture frames came out 21% off in proportion on average
(16/21 good shapes), and big pieces are rarely fully in one frame. Assets from product
photos were good (pilot: 7/7 usable, within 15%, sizes from the listing). Crowded real
rooms were the other failure (5 of 12 public RoomPlan exports had no layout).

## Decision
v1's product is **"will it fit, and how should I arrange it?"** for furniture the user is
considering:
1. Scan the room (RoomPlan). Movable furniture is removed (`pipeline.empty_room`);
   walls, doors, windows, fixtures and built-in cabinetry stay.
2. Add new pieces from product photos plus listing dimensions (SAM 3D + `normalize`;
   paste a photo and type the size when a listing can't be read).
3. Request in words, then rules, solver and checks, as before. "Doesn't fit" is an answer, with
   the reason (e.g. footprints + clearances need 90% of the floor), and the user can
   try a variant (another sofa, drop the desk).

Rearranging the user's own furniture remains a later mode. Nothing built is dropped:
ingest still reads existing furniture, the pipeline keeps the full-scan path.

## Consequences
- Removes per-object capture (no step-back photos, no frame picking) and the weakest
  asset path from v1. Sizes come from manufacturers, not scan boxes.
- The room is shown as a clean 3D model of the scan (walls, floor, openings, fixtures).
  A photoreal view of the real room with its furniture erased needs object removal
  (GOR-IS), still deferred.
- Getting product photos and sizes becomes a product surface: listing parsers where
  sites allow (IKEA works; 12 of 20 sites in the pilot blocked scripted access), manual
  photo + size otherwise.
- Competes with retailer tools that do the same for one catalogue (e.g. IKEA's room
  scanning app). Our difference: any furniture from any listing or photo, and automatic
  arrangement from the user's words.
- Crowded-room solver work (chairs under tables) drops in priority: users choose what
  goes in, and the solver reports what doesn't fit.
