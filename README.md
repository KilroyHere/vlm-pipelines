# roomlayout

Scan a room, pick furniture you're considering (a product photo plus the listing's
size), say how you want it, and find out whether it fits: the top 3 layouts that do, or
what to change when nothing does (ADR-0009, ADR-0010).

## Docs
- [Architecture](docs/ARCHITECTURE.md): full pipeline and v1 scope
- [Project plan](docs/PROJECT_PLAN.md): workstreams, milestones, risks, status
- [Data contracts](docs/contracts.md): frames, units, schema, constraint vocabulary
- ["Will it fit?"](docs/fit.md): listing-size boxes, fit check and advice
- [RoomPlan ingest](docs/ingest.md), [request parsing](docs/vlm.md),
  [rule library](docs/constraints.md), [solver](docs/solver.md), [assets](docs/assets.md)
- [Decision records](docs/adr/)

## Layout
```
src/roomlayout/schema/       shared data contract (lead-owned)
src/roomlayout/ingest/       RoomPlan JSON -> LayoutProblem
src/roomlayout/constraints/  request parsing (LLM) + rule library
src/roomlayout/solver/       layout search
src/roomlayout/assets/       mesh normalization + listing-size boxes
src/roomlayout/pipeline.py   scan -> empty room -> furnish -> plan
src/roomlayout/fit.py        "will it fit?" + advice
spikes/                      SAM 3D GPU harness, ARKitScenes e2e, fit baseline
examples/problems/           LayoutProblem fixtures
examples/products/           test products (listing sizes)
datasets/                    manifests + fetch scripts for public data (data/ is gitignored)
schemas/                     exported JSON Schemas (generated, committed)
```

## Setup
```bash
uv venv && uv pip install -e ".[solver,assets,vlm,dev]"
.venv/bin/pytest                      # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m roomlayout.schema.export   # after any schema change
```
