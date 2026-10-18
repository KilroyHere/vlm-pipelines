# Project plan: v1

Scope: see "v1 scope" in [ARCHITECTURE.md](ARCHITECTURE.md). Contracts:
[contracts.md](contracts.md). Decisions: [adr/](adr/).

## Workstreams

| ID | Workstream | Owner | Package / dir | Depends on |
|---|---|---|---|---|
| WS-0 | Contract, integration, docs | Tech lead | `roomlayout.schema`, `docs/` | — |
| WS-A | Layout solver (Step 1) | Engineer A | `roomlayout.solver` | WS-0 |
| WS-B | Asset normalization + SAM 3D spike (Step 2) | Engineer B | `roomlayout.assets`, `spikes/sam3d/` | WS-0 |
| WS-C | Rule library (M3a) | Engineer A | `roomlayout.constraints.rules` | WS-0, M1 |
| WS-D | VLM constraint extraction (M3b) | Engineer D | `roomlayout.constraints.vlm` | WS-0 |
| WS-E | RoomPlan ingest (M4a) | Engineer B | `roomlayout.ingest` | WS-0 |
| EXT | GPU access, LiDAR device + Mac, API credentials | Product owner | — | — |

WS-A and WS-B share **only** `roomlayout.schema`. Neither imports the other.

## Milestones and exit criteria

### M1: Solver core (WS-A)
- [x] `solve(problem, k=10, seed, time_budget_s)` → `SolveResult`, deterministic for
      a given seed.
- [x] All implicit rules + every constraint type in the vocabulary implemented, with
      unit tests per type.
- [x] Test suite of ≥ 10 problems (`examples/problems/`), including an infeasible
      one and a crowded one.
- [x] Feasible layouts found for every feasible suite problem within 10 s on 4 cores.
- [x] Top-k layouts are mutually diverse (documented distance threshold).
- [x] Infeasibility explained in `SolveResult.notes`.
- [x] CLI `python -m roomlayout.solver PROBLEM.json --out DIR` writes the result JSON
      + top-down PNG per layout.

Result: 12 problems; every feasible one reaches a first feasible layout in ≤ 1.5 s
(seeds 0–2, 4 cores); `crowded_studio` is the tightest (7–10 distinct feasible
layouts). Details: [solver.md](solver.md). Contract v0.3.0 `elevation` supported (wall TV above a console).

### M2a: Asset normalization (WS-B, CPU)
- [x] Raw mesh (any common format) → canonical frame (ADR-0001), fitted to measured
      `Dimensions`, exported as GLB with correct glTF Y-up conversion (round-trip test).
- [x] Footprint extraction → simplified local-frame `Polygon2D`, validated by tests on
      synthetic L-sofa / round table / chair meshes.
- [x] Front-orientation candidates: 4 yaw hypotheses + side renders for a later
      VLM/user pick; heuristic default.
- [x] Mismatch report when the generator's aspect ratio disagrees with measured
      dimensions beyond a tolerance.
- [x] CLI `python -m roomlayout.assets RAW_MESH --dims W D H --category C --out DIR`
      → GLB + `Asset` JSON that validates and drops straight into a `LayoutProblem`.

### M2b: SAM 3D spike (WS-B harness; GPU run by EXT)
- [x] Verified requirements: GPU/VRAM, license terms (commercial?), input format
      (image + mask).
- [x] Runbook + scripts to run SAM 3D Objects on a folder of `image + mask` pairs on a
      rented GPU, then feed outputs through M2a.
- [x] Evaluation protocol: 20-photo test set spec, scoring rubric, go/no-go thresholds.
- [ ] **Go/no-go decision**: blocked on EXT (GPU + Hugging Face checkpoint approval).

Verified SAM 3D facts (details and sources: `spikes/sam3d/README.md`): linux x86-64 + NVIDIA GPU
≥ 32 GB VRAM; checkpoints gated on Hugging Face with manual approval; output mesh is Y-up and
not metric; SAM License has no non-commercial clause (engineering reading, **not** legal advice).
Estimated spike cost: 3–4 GPU-hours on a 48 GB card, < $10.

### M3a: Rule library (WS-C)
- [x] `implied_constraints(problem) -> list[Constraint]` (origin `rule_library`, each with a
      rationale) covering seating↔TV, coffee table, desk + chair, dining set, bed +
      nightstands, storage against walls, door/window/walkway safety rules.
- [x] Precedence: never emits an implied relation for a subject the user/VLM constrained.
- [x] "Minimal" variants of the suite problems (user essentials only) solve to layouts
      that pass automated sensibility checks (e.g. coffee table between sofa and TV).
- [x] docs/constraints.md (rule table with rationale).

Result: from a problem with no relational constraints, the rule library alone yields a top layout
that is feasible and passes every sensibility check on all 10 feasible suite rooms × 3 seeds
(30/30); lead re-verified on an unseen seed. Policy: ADR-0007. Details: [constraints.md](constraints.md).
Solver quality (WS-A): top-k quality gate (`cost_margin`, default 1.0), group moves for strong soft
ties, exact single-pass walkway check. Minimal variants: 261/263 returned layouts pass every check
(was 200/252); original suite unchanged within noise. Tight rooms may return fewer than k layouts
(crowded_studio seed 1: one) by design; product decision pending on a UI minimum.

### M3b: VLM constraint extraction (WS-D), ADR-0006
- [x] `parse_request(problem, text, client=...) -> ParseResult` (constraints with
      `origin="vlm"`, restatements, unexpressed items, dropped items with reasons).
- [x] Structured-output schema accepted by the API (anyOf, no oneOf); static across
      requests; prompt caching on the static prefix.
- [x] Fully tested without credentials (recorded responses); live eval script over
      ≥ 50 realistic requests with expected constraints, runnable when credentials exist.

Result: 73-case eval set; schema verified offline against the documented limits (anyOf, no
oneOf). **Not yet verified live** (no credentials): schema acceptance by the API, cache hits,
real cost (estimated ~$0.03/call warm, ~$0.07 cold), extraction accuracy. Details: [vlm.md](vlm.md).

### M3c: orchestration (lead)
- [x] `build_constraints(problem, text)` = VLM + rule library + precedence, plus an
      integration test from user text to solved layouts.

Result: `roomlayout.constraints.build_constraints` (user/VLM first, then rule library).
Integration checkpoint 2 in `tests/test_integration.py`: user words → (offline, synthetic) VLM
answer → rule library → solver → feasible layout honouring the user's hard request and passing
every sensibility check; user intent overrides defaults; a refused parse still yields a problem.

### M4a: RoomPlan ingest, Python side (WS-E) (done except real-capture validation)
- [x] `roomlayout.ingest.roomplan`: Apple RoomPlan `CapturedRoom` JSON → `LayoutProblem`
      skeleton (Room + box Assets with `dimensions_from="roomplan"` + `initial` poses, no
      constraints). ARKit Y-up → our Z-up at this boundary only (ADR-0001).
- [x] Walls → CCW floor polygon (prefer RoomPlan floor polygon when present), doors/windows/
      openings mapped onto walls with offset, width, height, sill; fixtures (kitchen/bath
      appliances, fireplace) and built-in cabinetry immovable; stairs obstacles.
- [x] Every field we read verified against Apple's documentation or the 12 public exports,
      or marked unverified (docs/ingest.md); synthetic fixtures with known geometry
      (tests/ingest); graceful errors on multi-room/multi-story/malformed scans.
- [x] CLI `python -m roomlayout.ingest.roomplan CAPTURE.json --out problem.json`, plus
      `python -m roomlayout.pipeline CAPTURE.json --out DIR [--request TEXT]` (ADR-0008) for
      scan → zones → rules → layouts.
- [ ] Real-capture validation: needs a LiDAR iPhone/iPad export from the owner (EXT).

### M4b+: iOS capture app, per-object frames for SAM 3D, end-to-end. Not started.

**Requirement learned in M1:** layout quality is driven by *constraint completeness*,
not the solver. With only `faces(sofa, tv)`, the solver correctly put the coffee
table behind the sofa and the TV side-on. The M3 deterministic rule library must
emit the implied relations (mutual `faces` for seating↔TV and desk↔chair, coffee
table in front of seating, a soft viewing distance), and the VLM prompt must ask for
them.

## Backlog
- **Vocabulary growth (ADR-0006 amendment), decide from live-eval `out_of_vocabulary` counts:**
  `faces`/`near` to openings (top candidate), obstacles as targets, corner/centred placement.
- First live VLM eval run (needs credentials): schema acceptance, cache hits, effort low vs
  medium, per-tag accuracy. See docs/vlm.md.
- Line of sight for `faces` (only tall assets block view): WS-A request, deferred.
- Even distribution of chairs around a table: deferred.
- Height-banded footprints (chair tucked under desk): ADR-0003, deferred. **Now evidenced by
  real scans:** dining rooms with 10-13 chairs tucked under the table (public RoomPlan
  exports s2, s4) find no feasible layout even with 90 s, because chairs may not overlap
  the table footprint. Top solver item after the own-room test.
- Multi-room scans (several rooms in one capture, partition walls): v1 warns and plans one
  outline; per-room planning later.

## Integration checkpoints (tech lead)
1. ✅ An asset from M2a loads into a `LayoutProblem` and solves with M1
   (`tests/test_integration.py`: raw SAM 3D-convention mesh → normalize → disk →
   `Asset` JSON → solve).
3. ✅ User words → constraints → layouts (M3c, see above).
2. Schema change requests from either workstream are reviewed here and applied by
   the lead.

## Risk register

| Risk | Impact | Owner | Mitigation | Status |
|---|---|---|---|---|
| SAM 3D quality too low on real room photos | Lose the novelty feature | WS-B / EXT | Spike with go/no-go rubric; fallback: catalog retrieval, Object Capture | open |
| SAM 3D license blocks commercial use | Must swap generator | EXT | Counsel confirmed our intended use is fine (2026-10-17) | **closed** |
| Hugging Face checkpoint approval delay | M2b blocked | EXT | Access requested 2026-10-16; prepare photos + masks while waiting | open: awaiting approval |
| Front heuristic wrong on real (blobby/hallucinated) meshes | Assets face the wall | WS-B | Contact sheets for VLM/user confirmation; `front_confidence` surfaced in the schema | open |
| No GPU available for the spike | M2b blocked | EXT | Runpod connected. Plan from live reads: A40 Secure $0.59/h in CA-MTL-1, template `runpod-torch-v220`, pod volume disk (network volumes are Secure-only and absent where A40s are); bootstrap script. **No Blackwell** | pilot done; both pods terminated 2026-10-17 | closed for the pilot |
| Layouts satisfy rules but look bad | Core value | WS-A / M3 | Confirmed cause: under-specified constraints. Rule library emits implied relations (M3); visual review; VLM critic later | open: understood |
| Wrong furniture size | Silently wrong layouts | WS-B | Measured dimensions always win (ADR-0004); mismatch report | mitigated by design |
| Schema churn between workstreams | Integration rework | Lead | Single owner, version bumps, integration test | mitigated |
| Capture walk too close to furniture: big objects never fully in one frame (ARKitScenes: sofas 39-58% visible) | asset quality from the user's own video | WS-E / M4b | capture flow adds one step-back photo per large RoomPlan object; frame picker already ranks by visible share | open (found by the e2e run, 2026-10-17) |

## Status log
- 2026-10-16: Contract v0.1.0, ADRs 0001–0004, and the plan landed. WS-A and WS-B kicked off.
- 2026-10-16: WS-B delivered M2a (asset normalization, tested on CPU) and the M2b harness. Contract v0.2.0
  (additive: `allowed_categories`, `subtype`, quality signals) accepted from WS-B review. Open follow-up:
  front-candidate side-view render bug. M2b go/no-go waits on GPU + checkpoint access.
- 2026-10-16: WS-A delivered M1 (all exit criteria met). Integration checkpoint 1 passes. Contract
  v0.3.0 (`elevation`, solve stats, immovable assets need `initial`, ADR-0005 zone/wall policy).
  Deferred: line of sight, chair distribution. WS-A implementing `elevation`.
- 2026-10-16: WS-A delivered v0.3.0 solver support (`elevation`, solve stats, immovable assets).
  Lead review: fast + slow suites green, renders checked. **M1 and M2a closed; M2b waits on EXT**
  (Hugging Face checkpoint access, GPU, license confirmation). Next: M3 constraints + rule library.
- 2026-10-16: HF checkpoint access requested. GPU plan: 48 GB Ampere/Ada card on Vast.ai or RunPod Community;
  Blackwell excluded (SAM 3D's torch 2.5.1+cu121 has no sm_120 kernels).
- 2026-10-17: Counsel confirmed SAM 3D license is fine for our use. GPU provider: RunPod. Contract v0.4.0
  (shared `taxonomy`). ADR-0006 (VLM extraction design). M3a (rule library, WS-C) and M3b (VLM, WS-D)
  kicked off. No API credentials in the dev environment yet: VLM is built and tested offline.
- 2026-10-17: WS-D delivered M3b (VLM extraction, offline-verified, 73-case eval). Contract v0.5.0 (`Wall.label`,
  unique constraint ids). ADR-0006 amended (`create` + raw schema; evidence-driven vocabulary growth).
- 2026-10-17: M3a accepted (ADR-0007). Contract v0.6.0 (taxonomy: bench, piano_bench, ottoman, meeting_table).
  M3c landed: `build_constraints` + integration checkpoint 2. WS-A on solver quality (top-k gate, group moves).
- 2026-10-17: Runpod agent plugin install blocked by session policy (self-modification); owner offered connector or
  `RUNPOD_API_KEY` env secret instead. M4a (RoomPlan ingest) kicked off with WS-B as WS-E.
- 2026-10-17: WS-A solver quality accepted (lead re-ran fast + slow suites: green). Runpod skill added by owner;
  picks up in a new session.
- 2026-10-17: Runpod connector connected (no pods/volumes on the account). Runbook corrected from live reads
  (network volumes are Secure-Cloud-only; A6000 out of stock; A40 Secure $0.59/h) + `runpod_bootstrap.sh`.
- 2026-10-17: HF access approved; Runpod secret `hf_token` created. Spike pod `z8wwqdkf02w7at` (A40, CA-MTL-1,
  $0.59/h, runpod-torch-v220, 100 GB /workspace) created and Jupyter verified. Next: owner runs the bootstrap + spike.
- 2026-10-17 05:00 UTC: Overnight plan (owner asleep): spike made fully automatic (OWLv2 box -> SAM 2 mask,
  photos fetched from photo_sources.md; WS-D). Lead runs SAM 3D on pod `z8wwqdkf02w7at`, downloads + scores, then
  terminates it. Safety check-ins scheduled 08:59 and 12:59 UTC (stop the pod if idle/stuck). The pod's own key cannot
  stop it (REST 403), so stopping relies on the lead/connector. M4a (RoomPlan ingest) paused by the owner; partial
  work is in the WIP commits.
- 2026-10-17 13:05 UTC: SAM 3D env built and Meta's demo passed on the pod at 05:47 (torch 2.5.1+cu121, 13 GB
  checkpoints). Spike NOT run: the session worker restarted, which ended the WS-D data agent (no photo package
  produced) and interrupted the 08:59 check-in before its stop. The 12:59 check-in stopped the pod. Runpod spend
  today $4.56 (GPU $4.41, disk $0.15); ~$4 of it idle 05:47-13:05. Lesson: in-session agents and watchers don't
  survive worker restarts; long unattended GPU work needs a pod-side job with its own deadline.
- 2026-10-17 14:40 UTC: **SAM 3D pilot done (7 web items, fully automatic).** All 7 usable (mean shape 4.0/5),
  proportions within 15% of manufacturer dims, L-sectional preserved, front heuristic 6/6, 14-20 s/item on A40.
  Promising, not yet go/no-go (needs 20 items). Details: spikes/sam3d/PILOT_RESULTS.md. Both pods terminated;
  Runpod spend today $5.17 (pilot ~$0.60, idle night ~$4).
- 2026-10-17 16:20 UTC: **First end-to-end on real rooms** (3 ARKitScenes living rooms, LiDAR scan + capture
  video standing in for the user's own scan; spikes/e2e_arkitscenes/RESULTS.md). Scan -> room + boxes -> best frame
  per object by camera pose -> SAM 2 mask from the projected box -> SAM 3D (21 objects, median 15 s) -> normalize ->
  request -> constraints -> solve -> checks. All 3 rooms: feasible layouts meeting the whole request, passing checks
  (one flag on a user-pinned cabinet). Found and fixed 2 rule-library precedence bugs (ADR-0007 amendment). Asset
  quality from capture frames is below the photo pilot (21% mean proportion mismatch, 16/21 good shapes): capture
  must add a step-back photo per large object. Not yet exercised: doors/windows, live VLM, RoomPlan importer.
  Pod `v39xw1o6z9hkyu` terminated (~$0.65).
- 2026-10-18: **M4a RoomPlan importer done** (except validation on the owner's own scan).
  `roomlayout.ingest.roomplan` + CLI, `roomlayout.pipeline` (ADR-0008), 18 ingest tests on
  synthetic captures. All 12 public exports convert, and floors, walls and openings match the
  raw RoomPlan walls. Real scans forced 6 mapping rules (docs/ingest.md):
  - kitchen cabinetry and wall cabinets are fixed;
  - a TV one room over is dropped;
  - L/U sofa pieces are merged;
  - wide doors have no swing arc;
  - doors in partitions are gaps;
  - passages need at least 0.8 m.
  One solver policy also changed: rules among pinned assets alone are facts (ADR-0008), and
  a `hard_terms` crash on infeasible problems with pinned assets is fixed.
  Rule library + solver on the 12 exports (30 s):
  - 7 rooms get feasible layouts: 5 empty, a bedroom (4 items) and a meeting room (8 items);
  - 5 crowded rooms don't:
    - 2 dining rooms with 10-13 tucked-in chairs, still infeasible at 90 s (see backlog);
    - a room with a 3.5 x 2.5 m U-sofa;
    - an open-plan kitchen/dining/living room with 17 items, needing 94% of the free floor
      for footprints and zones;
    - a multi-room scan (now warned).
- 2026-10-18: **Pivot (owner, ADR-0009): v1 furnishes an emptied room with new furniture from
  product photos.** Rearranging the user's own furniture becomes a later mode.
  - `roomlayout.pipeline` gains `empty_room`, `load_asset` and `furnish`, plus CLI
    `--empty --add ASSET.json` (tests).
  - Demo (spikes/e2e_arkitscenes/furnish_demo.py, RESULTS.md): 3 real ARKitScenes rooms
    emptied, refurnished with the pilot's 7 product-photo assets.
  - Two bigger rooms: both the KIVIK sectional set and the 3-seat sofa set fit, meeting the
    whole 9-rule request with all checks passing.
  - Small 3x4 m room: neither fits (the sectional set needs 90% of the free floor). The set
    without the desk does fit (6/6).
  - Next: product-photo intake (listing parser + manual photo/size), hosted SAM 3D, web viewer
    MVP; the owner's own-room scan is still the real-capture test.

