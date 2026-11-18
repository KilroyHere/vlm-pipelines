# ADR-0010: Web viewer first, research-only data, box-first "will it fit?"

- Status: accepted (owner decisions, 2026-11-18)
- Date: 2026-11-18

## Context
After the v1 pivot (ADR-0009) the owner set the next phase's terms:
- There is no own-room LiDAR scan yet; public scans stand in for it.
- The app surface to build first is a web viewer, not the iOS capture app.
- The project is research-only for now.
- The e2e run (spikes/e2e_arkitscenes/RESULTS.md) found that box assets from listing
  dimensions give the same feasibility and request results as SAM 3D meshes on real
  rooms.

## Decision
1. **Web first.** A three.js viewer, run from the pipeline's output files (problem,
   result, fit report, optional GLBs), replaces the matplotlib renders as the way
   layouts are shown and reviewed. The iOS capture app waits.
2. **Research-only data.** Non-commercial datasets and models are allowed (ABO,
   ARKitScenes, SAM 3D, TRELLIS). Public RoomPlan exports found on GitHub mostly carry no
   license. They are listed with source and git blob hash in
   `datasets/roomplan_public/manifest.json`, fetched into the gitignored `data/` folder,
   and never committed or redistributed. The same holds for retailer product photos:
   URLs are recorded, the images are not committed.
3. **Box first.** "Will it fit?" is answered from listing dimensions alone
   (`roomlayout.assets.listing` + `roomlayout.fit`), in seconds, before any 3D model
   exists. When nothing fits, the answer is actionable: which product to drop, and the
   widest version of the biggest piece that fits. SAM 3D meshes, when they arrive,
   replace the boxes in the viewer only. They don't change the answer.

## Consequences
- Answers no longer wait on a GPU. Image-to-3D becomes a quality-of-picture concern,
  evaluated separately (Phase 2, ABO ground truth).
- A commercial launch would need this revisited: dataset and model licenses, and
  retailer image use.
- The fit advice uses extra solver runs (one per product, plus about 5 for the width
  search), so a "doesn't fit" answer takes several times longer than a "fits" answer.
