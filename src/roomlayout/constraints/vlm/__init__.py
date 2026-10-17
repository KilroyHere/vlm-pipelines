"""Stage 4b: the user's words -> constraints, with one Claude call (ADR-0006, docs/vlm.md).

    from roomlayout.constraints.vlm import parse_request
    result = parse_request(problem, "sofa facing the TV, keep it cozy")
    result.constraints   # list[roomlayout.schema.Constraint], origin="vlm"
    result.restatements  # one confirmation line per constraint
    result.unexpressed   # wishes outside the vocabulary ("cozy"), with reasons
    result.dropped       # invalid items (unknown ids, implausible values), with reasons

Imports ``roomlayout.schema`` and ``anthropic`` only (ADR-0002). Requires the ``vlm`` extra.
"""

from .extract import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    FALLBACK_BETA,
    Attempt,
    ParseResult,
    Unexpressed,
    Usage,
    build_request,
    parse_request,
)
from .output import OUTPUT_SCHEMA, LLMOutput, schema_problems
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT
from .summary import summarize_problem
from .validate import Dropped

__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "FALLBACK_BETA",
    "OUTPUT_SCHEMA",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "Attempt",
    "Dropped",
    "LLMOutput",
    "ParseResult",
    "Unexpressed",
    "Usage",
    "build_request",
    "parse_request",
    "schema_problems",
    "summarize_problem",
]
