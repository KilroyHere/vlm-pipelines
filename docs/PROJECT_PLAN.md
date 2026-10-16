# Project plan: v1

Scope: see "v1 scope" in [ARCHITECTURE.md](ARCHITECTURE.md). Contracts:
[contracts.md](contracts.md). Decisions: [adr/](adr/).

## Workstreams

| ID | Workstream | Owner | Package / dir | Depends on |
|---|---|---|---|---|
| WS-0 | Contract, integration, docs | Tech lead | `roomlayout.schema`, `docs/` | — |
| WS-A | Layout solver (Step 1) | Engineer A | `roomlayout.solver` | WS-0 |
| WS-B | Asset normalization + SAM 3D spike (Step 2) | Engineer B | `roomlayout.assets`, `spikes/sam3d/` | WS-0 |
| EXT | GPU access, LiDAR device + Mac, license decision | Product owner | — | — |

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
layouts). Details: [solver.md](solver.md). In progress: `elevation` support (v0.3.0).

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

### M3+: constraints (VLM), RoomPlan ingest + iOS capture, end-to-end. Not started.

**Requirement learned in M1:** layout quality is driven by *constraint completeness*,
not the solver. With only `faces(sofa, tv)`, the solver correctly put the coffee
table behind the sofa and the TV side-on. The M3 deterministic rule library must
emit the implied relations (mutual `faces` for seating↔TV and desk↔chair, coffee
table in front of seating, a soft viewing distance), and the VLM prompt must ask for
them.

## Backlog
- Line of sight for `faces` (only tall assets block view): WS-A request, deferred.
- Even distribution of chairs around a table: deferred.
- Height-banded footprints (chair tucked under desk): ADR-0003, deferred.

## Integration checkpoints (tech lead)
1. ✅ An asset from M2a loads into a `LayoutProblem` and solves with M1
   (`tests/test_integration.py`: raw SAM 3D-convention mesh → normalize → disk →
   `Asset` JSON → solve).
2. Schema change requests from either workstream are reviewed here and applied by
   the lead.

## Risk register

| Risk | Impact | Owner | Mitigation | Status |
|---|---|---|---|---|
| SAM 3D quality too low on real room photos | Lose the novelty feature | WS-B / EXT | Spike with go/no-go rubric; fallback: catalog retrieval, Object Capture | open |
| SAM 3D license blocks commercial use | Must swap generator | EXT | License text has no non-commercial clause; counsel to confirm. Generator boundary (ADR-0004) keeps a swap cheap | open: needs legal confirmation |
| Hugging Face checkpoint approval delay | M2b blocked | EXT | Request access now (manual approval; asks for legal name, DOB, organization) | open: **needs owner action** |
| Front heuristic wrong on real (blobby/hallucinated) meshes | Assets face the wall | WS-B | Contact sheets for VLM/user confirmation; `front_confidence` surfaced in the schema | open |
| No GPU available for the spike | M2b blocked | EXT | Rent a ≥ 32 GB NVIDIA card hourly (48 GB ≈ $0.6–1.1/h); everything else runs on CPU | open: **needs owner action** |
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
