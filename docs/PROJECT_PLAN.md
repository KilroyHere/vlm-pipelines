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
- [ ] `solve(problem, k=10, seed, time_budget_s)` → `SolveResult`, deterministic for
      a given seed.
- [ ] All implicit rules + every constraint type in the vocabulary implemented, with
      unit tests per type.
- [ ] Test suite of ≥ 10 problems (`examples/problems/`), including an infeasible
      one and a crowded one.
- [ ] Feasible layouts found for every feasible suite problem within 10 s on 4 cores.
- [ ] Top-k layouts are mutually diverse (documented distance threshold).
- [ ] Infeasibility explained in `SolveResult.notes`.
- [ ] CLI `python -m roomlayout.solver PROBLEM.json --out DIR` writes the result JSON
      + top-down PNG per layout.

### M2a: Asset normalization (WS-B, CPU)
- [ ] Raw mesh (any common format) → canonical frame (ADR-0001), fitted to measured
      `Dimensions`, exported as GLB with correct glTF Y-up conversion (round-trip test).
- [ ] Footprint extraction → simplified local-frame `Polygon2D`, validated by tests on
      synthetic L-sofa / round table / chair meshes.
- [ ] Front-orientation candidates: 4 yaw hypotheses + side renders for a later
      VLM/user pick; heuristic default.
- [ ] Mismatch report when the generator's aspect ratio disagrees with measured
      dimensions beyond a tolerance.
- [ ] CLI `python -m roomlayout.assets RAW_MESH --dims W D H --category C --out DIR`
      → GLB + `Asset` JSON that validates and drops straight into a `LayoutProblem`.

### M2b: SAM 3D spike (WS-B harness; GPU run by EXT)
- [ ] Verified requirements: GPU/VRAM, license terms (commercial?), input format
      (image + mask).
- [ ] Runbook + scripts to run SAM 3D Objects on a folder of `image + mask` pairs on a
      rented GPU, then feed outputs through M2a.
- [ ] Evaluation protocol: 20-photo test set spec, scoring rubric, go/no-go thresholds.
- [ ] **Go/no-go decision** (needs the GPU run).

### M3+: constraints (VLM), RoomPlan ingest + iOS capture, end-to-end. Not started.

## Integration checkpoints (tech lead)
1. An asset from M2a's CLI loads into a `LayoutProblem` and solves with M1. This is
   the "fits together" test, kept in `tests/test_integration.py`.
2. Schema change requests from either workstream are reviewed here and applied by
   the lead.

## Risk register

| Risk | Impact | Owner | Mitigation | Status |
|---|---|---|---|---|
| SAM 3D quality too low on real room photos | Lose the novelty feature | WS-B / EXT | Spike with go/no-go rubric; fallback: catalog retrieval, Object Capture | open |
| SAM 3D license blocks commercial use | Must swap generator | EXT | Check before building on it; generator boundary (ADR-0004) keeps the swap cheap | open |
| No GPU available for the spike | M2b blocked | EXT | Hourly cloud GPU; everything else runs on CPU | open: **needs owner action** |
| Layouts satisfy rules but look bad | Core value | WS-A | Visual review of the suite; scoring iteration; VLM critic later | open |
| Wrong furniture size | Silently wrong layouts | WS-B | Measured dimensions always win (ADR-0004); mismatch report | mitigated by design |
| Schema churn between workstreams | Integration rework | Lead | Single owner, version bumps, integration test | mitigated |

## Status log
- 2026-10-16: Contract v0.1.0, ADRs 0001–0004, and the plan landed. WS-A and WS-B kicked off.
