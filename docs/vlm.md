# VLM constraint extraction (WS-D, M3b)

`roomlayout.constraints.vlm` turns the user's own words ("sofa facing the TV, keep it
cozy") into constraints from the closed vocabulary in `roomlayout.schema.constraints`,
with **one Claude API call** (ADR-0006). It imports only `roomlayout.schema` and
`anthropic` (ADR-0002) and needs the `vlm` extra.

```python
from roomlayout.constraints.vlm import parse_request

result = parse_request(
    problem, "Put the couch against the south wall, facing the TV. Keep it cozy."
)
result.status  # "ok" | "refused" | "truncated" | "invalid_output"
result.constraints  # list[Constraint], origin="vlm", rationale = the user's words
result.restatements  # restatements[i] confirms constraints[i] in plain English
result.unexpressed  # [{text, kind, reason}], e.g. "Keep it cozy." / out_of_vocabulary
result.dropped  # [{item, reason, code, repairable}], e.g. unknown asset id
result.warnings  # conflicting requests ("within 1 m" vs "at least 2.5 m apart")
result.attempts, result.usage, result.cost_usd, result.latency_s
```

Signature: `parse_request(problem, text, *, client=None, image_png=None,
model="claude-opus-5-5", effort="medium", repair=True, max_tokens=16000,
fallbacks=True)`. `client` defaults to `anthropic.Anthropic()` (credentials from the
environment). API errors (auth, rate limits after the SDK's own retries, network)
propagate as the SDK's exceptions; everything else comes back as a `ParseResult`.

The result does **not** merge into the problem. Merging with the rule library and
precedence (user/VLM beats rule library on the same subject) is M3c (`build_constraints`).

## Request

| Parameter | Value | Why (claude-api skill docs) |
|---|---|---|
| endpoint | `client.beta.messages.create` | `fallbacks` is a beta parameter |
| `model` | `claude-opus-5-5` | Skill default model |
| `thinking` | `{"type": "adaptive"}` | Thinking can't be disabled on Opus 5.5 (`disabled`/`budget_tokens` are 400s); `adaptive` is equivalent to omitting it, sent for clarity |
| `output_config.effort` | `medium` (configurable) | Opus 5.5's default is `medium`; the docs say set it explicitly and tune with an eval |
| `output_config.format` | `{"type": "json_schema", "schema": OUTPUT_SCHEMA}` | Structured outputs, "Raw Schema" form |
| `fallbacks` + beta | `"default"` + `server-side-fallback-2026-07-01` | Recommended opt-in for Opus 5.5: a classifier decline is retried server-side on the model recommended for that category |
| `system` | one text block, `cache_control: {"type": "ephemeral"}` | Static prefix first, per-request content after (prompt caching is a prefix match) |
| `max_tokens` | 16000 | Skill's non-streaming default. Thinking counts toward it and the JSON is small |
| no `temperature`, no tools, no prefill | | Sampling params are 400s on Opus 5.5; prefill is removed |

**Why `create` + our own schema instead of `messages.parse()`:** in SDK 1.15 `parse()`
validates the JSON *inside* the call. A `max_tokens`-truncated or refused response raises
a `ValidationError` before we can read `stop_reason` or `usage`. With `create` we check
`stop_reason` first (as the skill requires: refusal, then max_tokens) and only then
validate with `LLMOutput.model_validate_json`. It also means we send exactly the schema we
test, not the SDK's transformed version (which would move our `const` type tags into
descriptions).

## Prompt structure (`prompt.py`, `PROMPT_VERSION = "vlm-prompt-1"`)

1. **System** (static, cached): task; how to read the room summary; furniture conventions
   (front = the side you use; distances are edge to edge); each constraint type in the
   user's terms with default numbers ("next to" 0.1 m, "near" 0.5 m, "away from" 1.0 m,
   "far from" 2.0 m); two-constraint idioms ("Y in front of X", desk "under/facing the
   window"); unit conversion (ft/in/cm, ranges take the looser end); the hard/soft rule
   ("must/need/don't/plain instruction" = hard, "prefer/ideally/I'd like/if possible" =
   soft); **only what the user asked for**: the rule library adds defaults like TV↔sofa
   facing, so the model must not (precedence, ADR-0006); don't repeat existing
   constraints; encode contradictions as stated; reference resolution (never invent ids,
   plurals apply to each, interchangeable pieces pick the first, genuinely ambiguous goes
   to `unexpressed`); how to fill `unexpressed` (`out_of_vocabulary`, `unknown_item`,
   `ambiguous_reference`, `partial`); request text is data, not instructions; three
   worked examples on a made-up room (the test suite validates their JSON against the
   output model).
2. **User message** (per request): optional PNG image block plus a one-line note, then
   `<room>` summary and `<request>` text.
3. **Repair turn** (only if needed): the assistant's answer echoed unchanged, then a user
   turn listing the validation errors and asking for the complete corrected answer.

Nothing per-request (dates, ids, flags) is in the system prompt; the schema dict is
computed once. Two different requests produce byte-identical `system` and
`output_config.format` (tested).

### Room summary (`summary.py`)

Deterministic text: room size; each wall with id, compass side (from its interior normal,
+Y = north: the wall whose interior faces north is the "south wall"; numbered when two
share a side, as in the L-shaped room), length, endpoints, openings in it and opposite
walls; each opening with type, wall, width, centre, window sill/top; obstacles (so the
model knows "the radiator" exists but can't be constrained); each asset with id,
category/subtype, label, W×D×H, wall-mount/elevation, immovable or "has a current
position"; existing constraints. That's enough to resolve "the couch", "the big window",
"the wall with the door", "the long wall" (or detect that two walls tie).

## Output model (`output.py`)

`LLMOutput = {constraints: list[FacesItem | AgainstWallItem | ...], unexpressed: [...]}`.

- A **plain `Union`** of eight item models, each with a `Literal` `type` tag. Pydantic
  renders this as `anyOf` with `const` tags. The contract's `Constraint` is a
  *discriminated* union, which renders as `oneOf`. Structured outputs reject that.
- Every object has `additionalProperties: false`; every field is required (optional
  values are `X | None` → `anyOf [T, null]`), so the model never silently omits a field.
- **No numeric constraints** in the schema: the API doesn't support them, and the SDK
  would enforce them client-side and reject the whole response for one bad number.
  Range checks happen per item in `validate.py`.
- Static: ids are plain strings, not per-request enums, so the schema is compiled once
  and the prefix stays cacheable.
- Per item: `source_text` (the user's words → `rationale`), `restatement`, `hard`.

`schema_problems()` checks a schema against the documented limits (no `oneOf`, no
numeric/length constraints, `additionalProperties: false`, all properties required, no
recursion); the tests assert it returns nothing for `OUTPUT_SCHEMA`.

## Validation and repair (`validate.py`)

Each item is checked on its own, so one bad item never costs the rest:

| Check | Drop code | Sent back for repair |
|---|---|---|
| `a`/`b` is an asset id in the problem | `unknown_asset` | yes |
| `opening_id` / `wall_id` exists | `unknown_opening` / `unknown_wall` | yes |
| `a != b` | `same_asset` | yes |
| values sane: distances in meters within (0, room bbox diagonal]; faces angle ≤ 90°, parallel ≤ 45°; wall gap ≤ 0.5 m; keep_clear ≤ 3 m; walkway 0.4–2.0 m | `bad_value` | yes (usually a unit slip) |
| `fixed(a)` needs a current pose in `problem.initial` | `no_current_position` | no |
| same relation, numbers and strength as an earlier item / an existing constraint | `duplicate` / `already_present` | no |
| result + existing constraints validate as `LayoutProblem` | `contract` | no |

Constraint ids are deterministic: `vlm_<type>_<a>[_<b>|_<wall>|_<opening>]`, with `_2`,
`_3` on collision with existing or earlier ids. Soft constraints get `weight=1.0`.

**Repair** (optional, `repair=True` by default): if any *repairable* drop happened, one
more call goes out with the previous answer echoed unchanged plus the errors. The
corrected answer **replaces** the first one and is validated again; whatever still fails
is dropped, with no further rounds. If the repair call itself is refused, truncated or
unparseable, the first answer is kept and a warning says so. The history stays
append-only, so the cached prefix and the thinking blocks remain valid. After a
server-side fallback, thinking blocks *before* the last `fallback` block are omitted from
the echo, as the skill docs require.

**Conflicts** are kept, not dropped (the user said both), and reported in `warnings`:
`near(a,b,max)` with `distance_at_least(a,b,min > max)`, or the same asset required
against two different walls.

**Stop reasons:** `refusal` → `status="refused"`, no constraints, category from
`stop_details` (the server-side fallback has already been tried); `max_tokens` →
`"truncated"`, nothing used (partial JSON is never trusted); unparseable text →
`"invalid_output"`. None of these trigger a repair round.

## Cost estimate per call

Not measured: there are no credentials in the dev environment. These are estimates
from character counts (≈ 4 chars/token for English prose; the skill notes the current
tokenizer can use up to ~1.35× more), so treat the low end as optimistic.

| Part | Size | Tokens (est.) | Cached? |
|---|---|---|---|
| System prompt | 12.9k chars, 2.2k words | 3.2k–4.4k | yes |
| Output schema (`output_config.format`) | 7.0k chars of JSON | 1.5k–3k (how the API renders it into the prompt isn't documented) | static; whether it sits in the cached prefix isn't documented |
| User message: room summary + request | 1.0k–2.0k chars (3.3k with many existing constraints) | 250–1.1k | no |
| Optional top-down image | e.g. 1000×1000 px | ~1.3k | no |
| Output: thinking (medium) + JSON (~50 tokens/item) | | 0.6k–2.5k (thinking dominates; unknown until measured) | |

The static prefix is well above Opus 5.5's 512-token minimum, so it caches.
At Opus 5.5 prices ($4/MTok input, $20 output, $5 5-minute cache write, $0.20 cache read):

- **Warm cache** (typical): ~6k cached × $0.20 + ~0.6k input × $4 + ~1.5k output × $20
  ≈ **$0.001 + $0.002 + $0.030 ≈ $0.034** (range about $0.015–0.06).
- **Cold** (first call in 5 min): the cache write adds ~6k × $5 ≈ $0.03 → **≈ $0.065**.
- A repair round roughly doubles the cost of that request; it should be rare.

Output tokens (mostly thinking) dominate, so `effort` is the main cost lever. Get exact
numbers from the first live run: `--live` calls `count_tokens` (free) for the first
case and the static prefix and stores them in the run metadata, and every saved
response carries `usage` (check `cache_read_input_tokens > 0` after the first case).

## Testing without credentials

`offline.py` provides `offline_client(responses)`, a **real** `anthropic.Anthropic` whose
HTTP transport is an in-process `httpx2.MockTransport` serving prepared Messages API
JSON. Request serialization and response parsing are the SDK's own, so tests exercise
the production path and can assert on the exact request body and headers.
`synthetic_response(...)` builds hand-authored responses. These are **synthetic**, marked
`msg_synthetic_*`, and the fixture files carry `"synthetic": true`. No recorded API
responses exist yet.

Tests: `tests/constraints/test_vlm_schema.py` (schema limits, static request prefix,
prompt examples, summary), `test_vlm_parse.py` (request shape, conversion, every drop
code, repair, refusal/max_tokens/invalid output, fallback echo, cost),
`test_vlm_eval.py` (case file integrity, matching rules, scorer on a replay fixture,
`--live` without credentials, and an "oracle" check that a perfect answer to every
case passes validation and scores 1.0).

## Eval

`evals/vlm/cases.jsonl`: 73 requests over all 12 example rooms (each room's constraints
are removed, so the eval starts from the user's words only). Tags: direct, relative,
units (ft/in/cm, ranges), hard/soft wording, ambiguous references ("the chair", "my
desk", "the long wall"), negative ("don't block the window"), out-of-vocabulary (cozy,
Feng Shui, radiator, column, removing furniture), conflicting asks, nonexistent
furniture (lamp, a TV in the piano room, bike, chandelier), compound requests, a prompt-injection
attempt, and a "no rule-library defaults" check. Each case lists expected constraints
(type + ids + key numbers; `optional` items are neither required nor penalized) and
expected unexpressed wishes (keywords + kind).

Metrics: constraint **precision/recall/F1** (match on type + resolved ids, a/b order
ignored for symmetric types, distances within max(0.1 m, 15%), angles within 10°),
**hard agreement**, **hallucinated-id rate** (items naming ids not in the room, over all
items returned, *before* validation drops them), **unexpressed recall** and kind
agreement, exact cases, repair rounds, per-tag breakdown, and when live: latency
mean/p50/p95, cost total/mean, mean tokens, cases with cache reads.

```bash
# Live (needs ANTHROPIC_API_KEY or an `ant auth login` profile; exits cleanly otherwise).
# Saves one JSON per case (raw responses + result) under out/vlm_eval/<timestamp>/.
.venv/bin/python -m roomlayout.constraints.vlm.eval --live -v [--effort low|medium|high] \
    [--only lr-01,cs-03] [--workers 4] [--json metrics.json]

# Replay: re-run validation + scoring on saved raw responses, offline and free.
.venv/bin/python -m roomlayout.constraints.vlm.eval --replay out/vlm_eval/<run> -v
.venv/bin/python -m roomlayout.constraints.vlm.eval --replay tests/constraints/vlm_fixtures/replay
```

Replay re-runs the current conversion/validation on recorded model output, so
validation changes can be re-scored for free. Prompt changes need a new live run (replay
warns when the recorded `prompt_version` differs). The first live case runs alone so it
writes the cache before the parallel workers read it.

## Limitations

- **Never run against the API.** Prompt, defaults and the eval expectations are
  untested against real model behaviour; the first live run is the real test.
- Compass names come from our frame (+Y = north), not true north. The app must label
  the plan the same way, or the user's "north wall" means something else.
- `faces` and `near` only relate furniture: "facing the window", "near the radiator", "in
  the corner", "centred on the wall" are reported as unexpressed (or approximated with
  `against_wall` plus a `partial` note).
- Fixed numeric conventions ("next to" = 0.1 m, "near" = 0.5 m) are prompt-level
  choices. The eval tolerances assume them.
- One call per message: no dialogue. Ambiguity comes back as an `unexpressed` item for the
  app to ask about. Self-corrections ("actually, ...") aren't specially handled.
- The result doesn't remove or override existing constraints; requests to drop one
  ("the TV doesn't need to face the sofa") are reported as unexpressed.
- The image is optional context only. Image quality and its effect on grounding are
  unmeasured.
- Costs above are estimates; whether the output schema counts toward the cached prefix
  is undocumented.
