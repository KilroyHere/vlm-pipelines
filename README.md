# roomlayout

Will it fit, and how should I arrange it? Scan a room, empty it of its movable furniture
(walls, openings, fixtures and built-ins stay), add the pieces you're considering (a
product photo plus the listing's size), and say how you want them. You get up to 3
distinct layouts that fit (`--k 3` by default; the solver's quality gate keeps only
layouts close to the best one, so it can return fewer), or, when nothing fits, the reason
and what to change (ADR-0009, ADR-0010).

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
src/roomlayout/pipeline.py   scan -> empty room -> furnish -> plan (CLI below)
src/roomlayout/fit.py        "will it fit?" + advice
src/roomlayout/explain.py    plain-words "why" per layout
src/roomlayout/viewer/       three.js web viewer
spikes/e2e_arkitscenes/      end to end on real ARKitScenes rooms, incl. the furnish demo
spikes/                      also: SAM 3D GPU harness, image-to-3D eval, fit baseline
evals/                       eval sets (evals/vlm/cases.jsonl: request parsing)
examples/problems/           LayoutProblem fixtures
examples/products/           test products (listing sizes)
datasets/                    manifests + fetch scripts for public data (data/ is gitignored)
schemas/                     exported JSON Schemas (generated, committed)
```

## Setup
The extras are `solver`, `assets`, `vlm` (the Anthropic client for request parsing) and
`dev`.
```bash
uv venv && uv pip install -e ".[solver,assets,vlm,dev]"
.venv/bin/pytest                      # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m roomlayout.schema.export   # after any schema change
```

## Run
```bash
# empty the scanned room, add new pieces (Asset JSONs from roomlayout.assets), plan
python -m roomlayout.pipeline Room.json --out out/myroom \
    --empty --add sofa.asset.json --add table.asset.json \
    [--request "Sofa facing the TV"] [--room 0] [--k 3] [--seconds 20]
```
`--request` is parsed by the live VLM (needs `ANTHROPIC_API_KEY`); without it only the
rule library applies. Without `--empty`/`--add` it plans the scanned furniture as is.
Writes `problem.json`, `ingest_report.json`, `result.json` and `layout_<k>.png`
(`current.png` too when nothing new was added). For the box-only "will it fit?" from a
product file, see [docs/fit.md](docs/fit.md); to view results, [docs/viewer.md](docs/viewer.md).
