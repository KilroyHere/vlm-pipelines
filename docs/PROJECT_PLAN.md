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
In progress (WS-A): solver top-k quality gate + group moves for strong soft ties, so lower-ranked
layouts are as good as the top one.

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

### M4a: RoomPlan ingest, Python side (WS-E)
- [ ] `roomlayout.ingest.roomplan`: Apple RoomPlan `CapturedRoom` JSON → `LayoutProblem`
      skeleton (Room + box Assets with `dimensions_from="roomplan"` + `initial` poses, no
      constraints). ARKit Y-up → our Z-up at this boundary only (ADR-0001).
- [ ] Walls → CCW floor polygon (prefer RoomPlan floor polygon when present), doors/windows/
      openings mapped onto walls with offset, width, height, sill; fixtures (kitchen/bath
      appliances, fireplace, stairs) immovable.
- [ ] Every field we read verified against Apple's documentation or marked unverified;
      synthetic fixtures with known geometry; graceful errors on multi-room/odd scans.
- [ ] CLI `python -m roomlayout.ingest.roomplan CAPTURE.json --out problem.json` whose output
      solves with the rule library.
- [ ] Real-capture validation: needs a LiDAR iPhone/iPad export (EXT).

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
- Height-banded footprints (chair tucked under desk): ADR-0003, deferred.

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
| No GPU available for the spike | M2b blocked | EXT | **RunPod** Community A40/A6000 (~$0.45–0.5/h) with a network volume; **no Blackwell** (torch 2.5.1+cu121) | open: rent after HF approval |
| Layouts satisfy rules but look bad | Core value | WS-A / M3 | Confirmed cause: under-specified constraints. Rule library emits implied relations (M3); visual review; VLM critic later | open: understood |
| Wrong furniture size | Silently wrong layouts | WS-B | Measured dimensions always win (ADR-0004); mismatch report | mitigated by design |
| Schema churn between workstreams | Integration rework | Lead | Single owner, version bumps, integration test | mitigated |

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
