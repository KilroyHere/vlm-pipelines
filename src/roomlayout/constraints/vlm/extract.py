"""``parse_request``: one structured-output Claude call -> validated constraints (ADR-0006).

Request shape (see docs/vlm.md for the reasoning and the skill docs each choice follows):

- ``client.beta.messages.create`` with ``output_config.format`` = the static JSON Schema
  from ``output.py``. We parse the JSON ourselves *after* checking ``stop_reason``:
  ``messages.parse()`` validates inside the call, so a ``max_tokens``-truncated or
  refused response would raise before we could see why.
- ``model="claude-opus-5-5"``, ``thinking={"type": "adaptive"}`` (thinking can't be
  disabled on this model), ``output_config.effort`` set explicitly (the model's default
  is ``medium``; tune with the eval).
- ``fallbacks="default"`` + beta ``server-side-fallback-2026-07-01``: a safety-classifier
  decline is retried server-side on Anthropic's recommended model for that category.
- System prompt first, one block with ``cache_control``; per-request content after it.
- Optional single repair round: the previous answer is echoed unchanged and the
  validation errors are appended as a new user turn (append-only history).

The client is injected: anything with ``client.beta.messages.create(**kwargs)`` returning
an SDK ``BetaMessage``-shaped object. Tests use the real SDK over an offline transport
(``offline.py``), so responses go through the same parsing path as in production.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from roomlayout.schema import Constraint, LayoutProblem

from .output import OUTPUT_SCHEMA, LLMOutput
from .prompt import IMAGE_NOTE, PROMPT_VERSION, SYSTEM_PROMPT, repair_text, user_text
from .summary import summarize_problem
from .validate import Dropped, conflicts, convert

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
# Non-streaming default from the skill docs; thinking counts toward it, the JSON is small.
DEFAULT_MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# USD per million tokens: (input, output, 5-minute cache write, cache read).
# claude-opus-5-5 from the claude-api skill (cached 2026-10-06); others are fallback
# targets, used only when a fallback served the request.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
}

# "error" is only produced by callers that catch API exceptions (the eval); parse_request
# itself lets SDK exceptions propagate.
Status = Literal["ok", "refused", "truncated", "invalid_output", "error"]


class Unexpressed(BaseModel):
    text: str
    kind: str
    reason: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in Usage.model_fields})

    def cost_usd(self, model: str) -> float | None:
        price = PRICING.get(model)
        if price is None:
            return None
        inp, out, write, read = price
        return (
            self.input_tokens * inp
            + self.output_tokens * out
            + self.cache_creation_input_tokens * write
            + self.cache_read_input_tokens * read
        ) / 1e6


class Attempt(BaseModel):
    """One API call (the initial one or the repair round)."""

    kind: Literal["initial", "repair"]
    status: Status
    stop_reason: str | None
    refusal_category: str | None = None
    served_by: str | None = None  # response.model (differs from requested on fallback)
    fallback_used: bool = False
    request_id: str | None = None
    latency_s: float
    usage: Usage
    cost_usd: float | None = None
    n_items: int = 0  # constraint items the model returned
    dropped: list[Dropped] = []


class ParseResult(BaseModel):
    """What the user's words turned into. ``constraints[i]`` pairs with ``restatements[i]``."""

    status: Status
    message: str | None = None  # why nothing came back, for non-ok statuses
    constraints: list[Constraint] = []
    restatements: list[str] = []
    unexpressed: list[Unexpressed] = []
    dropped: list[Dropped] = []  # from the answer that was used (after repair, if any)
    warnings: list[str] = []  # conflicting requests, failed repair, ...
    prompt_version: str = PROMPT_VERSION
    model: str
    effort: str
    attempts: list[Attempt] = []
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    cost_usd: float | None = None


def build_request(
    problem: LayoutProblem,
    text: str,
    *,
    image_png: bytes | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fallbacks: bool = True,
) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create`` (initial call)."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    content: list[dict[str, Any]] = []
    if image_png is not None:
        if not image_png.startswith(_PNG_MAGIC):
            raise ValueError("image_png is not a PNG")
        import base64

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(image_png).decode("ascii"),
                },
            }
        )
        content.append({"type": "text", "text": IMAGE_NOTE})
    content.append({"type": "text", "text": user_text(summarize_problem(problem), text)})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }
    if fallbacks:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return block.model_dump(exclude_none=True)


def echo_content(content: list[Any]) -> list[dict[str, Any]]:
    """Assistant content to send back in the repair turn, unchanged except: after a
    server-side fallback, thinking/tool blocks before the last ``fallback`` block are
    omitted (skill docs: "Echoing fallback turns back")."""
    blocks = [_block_dict(b) for b in content]
    last_fb = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(blocks) if not (i < last_fb and b.get("type") in drop)]


def _usage(resp: Any) -> Usage:
    u = _get(resp, "usage")
    if u is None:
        return Usage()
    return Usage(**{k: int(_get(u, k, 0) or 0) for k in Usage.model_fields})


def _fallback_used(resp: Any) -> bool:
    if any(_get(b, "type") == "fallback" for b in _get(resp, "content", []) or []):
        return True
    iterations = _get(_get(resp, "usage"), "iterations") or []
    return any(_get(it, "type") == "fallback_message" for it in iterations)


def _final_text(resp: Any) -> str:
    blocks = list(_get(resp, "content", []) or [])
    last_fb = max((i for i, b in enumerate(blocks) if _get(b, "type") == "fallback"), default=-1)
    return "".join(_get(b, "text", "") for b in blocks[last_fb + 1 :] if _get(b, "type") == "text")


def _call(
    client: Any, kwargs: dict[str, Any], kind: Literal["initial", "repair"]
) -> tuple[Any, Attempt, LLMOutput | None, str | None]:
    """Make one call. Returns (response, attempt, parsed output or None, message)."""
    t0 = time.perf_counter()
    resp = client.beta.messages.create(**kwargs)
    latency = time.perf_counter() - t0
    stop = _get(resp, "stop_reason")
    served = _get(resp, "model")
    usage = _usage(resp)
    attempt = Attempt(
        kind=kind,
        status="ok",
        stop_reason=stop,
        served_by=served,
        fallback_used=_fallback_used(resp),
        request_id=_get(resp, "_request_id") or _get(resp, "id"),
        latency_s=latency,
        usage=usage,
        cost_usd=usage.cost_usd(served or kwargs["model"]),
    )
    # Branch on stop_reason before reading content (refusals may carry partial or no text).
    if stop == "refusal":
        attempt.status = "refused"
        attempt.refusal_category = _get(_get(resp, "stop_details"), "category")
        cat = attempt.refusal_category or "unspecified"
        return resp, attempt, None, f"The model declined the request (category: {cat})."
    if stop == "max_tokens":
        attempt.status = "truncated"
        return resp, attempt, None, "The answer hit max_tokens and is incomplete; nothing used."
    text = _final_text(resp)
    try:
        out = LLMOutput.model_validate_json(text)
    except ValidationError as e:
        attempt.status = "invalid_output"
        return resp, attempt, None, f"Output did not match the schema ({e.error_count()} errors)."
    attempt.n_items = len(out.constraints)
    return resp, attempt, out, None


def parse_request(
    problem: LayoutProblem,
    text: str,
    *,
    client: Any = None,
    image_png: bytes | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    repair: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fallbacks: bool = True,
) -> ParseResult:
    """Turn the user's words into validated ``origin="vlm"`` constraints for ``problem``.

    Args:
        problem: the room, assets and any existing constraints. Not modified.
        text: the user's request in their own words.
        client: an ``anthropic.Anthropic`` (or anything with the same
            ``beta.messages.create``); default ``anthropic.Anthropic()`` from the environment.
        image_png: optional top-down render, sent as a base64 image block.
        model, effort: request settings (effort: low|medium|high|xhigh|max).
        repair: allow one repair round-trip when items fail validation.
        max_tokens: output cap including thinking.
        fallbacks: opt into server-side refusal fallbacks (``fallbacks="default"``).

    Returns a ``ParseResult``; ``status`` is ``ok`` unless the model refused, was cut off
    by ``max_tokens``, or returned unparseable output (then no constraints are returned).
    API errors (auth, rate limit after the SDK's retries, network) propagate as the SDK's
    exceptions.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    kwargs = build_request(
        problem,
        text,
        image_png=image_png,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        fallbacks=fallbacks,
    )
    result = ParseResult(status="ok", model=model, effort=effort)

    resp, attempt, out, message = _call(client, kwargs, "initial")
    result.attempts.append(attempt)
    if out is None:
        result.status, result.message = attempt.status, message
        return _finish(result)

    accepted, dropped, repair_errors = convert(out, problem)
    attempt.dropped = dropped
    if repair and repair_errors:
        repair_kwargs = dict(kwargs)
        repair_kwargs["messages"] = [
            *kwargs["messages"],
            {"role": "assistant", "content": echo_content(_get(resp, "content", []))},
            {"role": "user", "content": [{"type": "text", "text": repair_text(repair_errors)}]},
        ]
        _, attempt2, out2, message2 = _call(client, repair_kwargs, "repair")
        result.attempts.append(attempt2)
        if out2 is None:
            result.warnings.append(f"Repair round failed, kept the first answer: {message2}")
        else:
            out = out2
            accepted, dropped, _ = convert(out2, problem)
            attempt2.dropped = dropped

    result.constraints = [a.constraint for a in accepted]
    result.restatements = [a.restatement for a in accepted]
    result.unexpressed = [Unexpressed(**u.model_dump()) for u in out.unexpressed]
    result.dropped = dropped
    result.warnings += conflicts(problem, result.constraints)
    return _finish(result)


def _finish(result: ParseResult) -> ParseResult:
    total = Usage()
    cost: float | None = 0.0
    for a in result.attempts:
        total = total + a.usage
        cost = None if cost is None or a.cost_usd is None else cost + a.cost_usd
    result.usage = total
    result.cost_usd = cost
    result.latency_s = sum(a.latency_s for a in result.attempts)
    return result
