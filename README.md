# roomlayout

Scan a room, generate 3D models of your own furniture, describe what you want, and
get back ~10 layouts that actually fit.

## Docs
- [Architecture](docs/ARCHITECTURE.md): full pipeline and v1 scope
- [Project plan](docs/PROJECT_PLAN.md): workstreams, milestones, risks, status
- [Data contracts](docs/contracts.md): frames, units, schema, constraint vocabulary
- [Decision records](docs/adr/)

## Layout
```
src/roomlayout/schema/   shared data contract (lead-owned)
src/roomlayout/solver/   layout search            (WS-A)
src/roomlayout/assets/   mesh normalization       (WS-B)
spikes/sam3d/            SAM 3D GPU harness + eval (WS-B)
examples/problems/       LayoutProblem fixtures
schemas/                 exported JSON Schemas (generated, committed)
```

## Setup
```bash
uv venv && uv pip install -e ".[solver,assets,dev]"
.venv/bin/pytest                      # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m roomlayout.schema.export   # after any schema change
```
