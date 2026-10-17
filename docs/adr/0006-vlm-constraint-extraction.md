# ADR-0006: Constraint extraction is one structured-output Claude call

- Status: accepted
- Date: 2026-10-17

## Context
Stage 4 turns a user's words ("sofa facing the window, keep it cozy") into constraints
from our closed vocabulary. The output must always validate, must only reference assets
and openings that exist, and the call must be cheap and testable without credentials.

## Decision
- **One call per request**, model `claude-opus-5-5`, adaptive thinking (always on for
  this model) with `output_config.effort` set explicitly (start `medium`, tune with the
  eval), server-side refusal `fallbacks: "default"`.
- **Structured outputs** (`client.messages.parse` / `output_config.format`) so the
  response is schema-valid JSON. Structured outputs support `anyOf` but not `oneOf`, and
  Pydantic discriminated unions emit `oneOf`. The VLM module therefore defines its own
  LLM-facing output model (a plain union with `Literal` type tags → `anyOf`) and converts
  it to `roomlayout.schema` constraints with `origin="vlm"`.
- **Static schema.** Asset/opening ids are plain strings, not per-request enums: a new
  schema per request would pay schema compilation every time and defeat caching.
  Referential integrity is checked in code; invalid items are dropped with a reason (one
  repair round-trip is allowed).
- The output also carries **what couldn't be expressed** (e.g. "cozy") with a reason, and
  a one-line restatement per constraint, so the app can show the user what was
  understood before solving.
- **Prompt caching**: the static system prompt (vocabulary, semantics, examples) comes
  first with `cache_control`; the per-request room summary and user text come after.
- **Testability**: the client is injected; tests use recorded responses. A live eval
  over a fixed prompt set runs only when credentials are present.
- **Precedence**: user/VLM constraints override rule-library ones on the same subject
  (the rule library skips an implied relation the user already spoke about).

## Consequences
- The constraint vocabulary is the product's expressiveness ceiling; requests outside
  it surface as "couldn't express", which is the signal for growing the vocabulary.
- A top-down image is an optional input (the orchestrator supplies a render); text
  summary alone must be sufficient.

## Amendment (2026-10-17, from implementation)
- The call is `client.beta.messages.create` with the raw JSON Schema in
  `output_config.format`, not `messages.parse()`. In SDK 1.15, `parse()` validates inside
  the call, so a `refusal` or `max_tokens` stop raises before `stop_reason` can be read.
  The SDK's schema transform also rewrites `const` tags that the API supports. We
  validate ourselves after checking `stop_reason`.
- Vocabulary growth is evidence-driven: the live eval and production logs count
  `unexpressed` items of kind `out_of_vocabulary`. Top candidates (from WS-D): `faces`/`near`
  targeting openings ("sofa facing the window"), obstacles as targets, room-relative
  placement (corner, centred on a wall).
