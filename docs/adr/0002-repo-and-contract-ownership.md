# ADR-0002: Single Python package; schema is a lead-owned contract

- Status: accepted
- Date: 2026-10-16

## Context
Several workstreams (solver, assets, later ingest, constraints, app) must fit
together. They will be built in parallel by different engineers.

## Decision
- One package `roomlayout` with stage subpackages: `schema`, `solver`, `assets`
  (later `ingest`, `constraints`). Optional dependency extras per stage keep heavy
  dependencies out of stages that don't need them.
- `roomlayout.schema` is the only shared surface. Stages depend on `schema`, never on
  each other.
- JSON Schemas are exported and committed so that non-Python consumers (Swift app,
  VLM structured output) validate against the same contract.
- Research code we don't own (SAM 3D, later GOR-IS) is never imported. It runs
  out-of-process behind a CLI/HTTP boundary (see ADR-0004).

## Consequences
- Integration is mechanical: if both sides validate against the schema, they fit.
- Schema changes are a coordination point, which is intentional.
