"""LLM-facing output model (what Claude returns), separate from the contract.

Two layers:

- **Wire** (``WireItem``, ``WireOutput``; ``OUTPUT_SCHEMA``): what the API constrains the
  model to. One flat item object for every constraint type: ``type`` is an enum, the
  references (``a``, ``b``, ``wall_id``, ``opening_id``) are nullable, and the type's one
  number is ``value``. The first live run (2026-11-18) showed why: an ``anyOf`` of eight
  typed item objects is rejected by the API with "The compiled grammar is too large",
  on Opus 5.5 and Haiku 5.5 alike; the flat item is accepted.
- **Typed** (``FacesItem`` ...; ``LLMOutput``): one model per constraint type, used inside
  ``validate.py``. ``WireItem.to_typed()`` maps a wire item onto it, or says which field
  the type needs and didn't get (a repairable drop).

Why separate from the contract (ADR-0006):

- ``roomlayout.schema.Constraint`` is a *discriminated* union, which Pydantic renders as
  ``oneOf``. Structured outputs accept ``anyOf`` but not ``oneOf``. Here every item type
  is a plain ``Union`` member with a ``Literal`` ``type`` tag, which renders as ``anyOf``
  (each branch pinned by ``const``).
- Structured outputs ignore numeric constraints (``minimum`` ...). The SDK would strip them
  and validate client-side, which would reject the *whole* response for one bad number.
  So this model has no numeric constraints at all; sanity checks happen per item in
  ``validate.py``, where a bad item is dropped with a reason instead.
- Every field is required (optional values are ``X | None``), every object forbids extra
  properties, and nothing depends on the request: one static schema, compiled once
  by the API and cacheable.
- Each item carries the user's words (``source_text``), a plain-English ``restatement``
  and ``hard``, which the contract has no place for.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_ITEM_CONFIG = ConfigDict(extra="forbid")


class _Item(BaseModel):
    model_config = _ITEM_CONFIG

    source_text: str = Field(
        description="The user's own words this item comes from, quoted as written."
    )


class FacesItem(_Item):
    type: Literal["faces"]
    a: str = Field(description="Asset id whose front should point at b.")
    b: str = Field(description="Asset id being faced.")
    max_angle_deg: float | None = Field(description="Tolerance in degrees; null = default 20.")
    hard: bool
    restatement: str


class AgainstWallItem(_Item):
    type: Literal["against_wall"]
    a: str = Field(description="Asset id whose back goes against a wall.")
    wall_id: str | None = Field(description="Wall id, or null for any wall.")
    max_gap: float | None = Field(description="Max gap to the wall in meters; null = 0.05.")
    hard: bool
    restatement: str


class NearItem(_Item):
    type: Literal["near"]
    a: str
    b: str
    max_distance: float = Field(description="Max edge-to-edge distance in meters.")
    hard: bool
    restatement: str


class DistanceAtLeastItem(_Item):
    type: Literal["distance_at_least"]
    a: str
    b: str
    min_distance: float = Field(description="Min edge-to-edge distance in meters.")
    hard: bool
    restatement: str


class ParallelItem(_Item):
    type: Literal["parallel"]
    a: str
    b: str
    max_angle_deg: float | None = Field(description="Tolerance in degrees; null = default 5.")
    hard: bool
    restatement: str


class KeepClearItem(_Item):
    type: Literal["keep_clear"]
    opening_id: str = Field(description="Door, window or passage id.")
    distance: float | None = Field(description="Clear depth in meters; null = default 0.6.")
    hard: bool
    restatement: str


class WalkwayWidthItem(_Item):
    type: Literal["walkway_width"]
    min_width: float | None = Field(description="Min walkway width in meters; null = 0.9.")
    hard: bool
    restatement: str


class FixedItem(_Item):
    type: Literal["fixed"]
    a: str = Field(description="Asset id that stays where it currently is.")
    hard: bool
    restatement: str


LLMItem = (
    FacesItem
    | AgainstWallItem
    | NearItem
    | DistanceAtLeastItem
    | ParallelItem
    | KeepClearItem
    | WalkwayWidthItem
    | FixedItem
)

ITEM_TYPES: tuple[type[_Item], ...] = (
    FacesItem,
    AgainstWallItem,
    NearItem,
    DistanceAtLeastItem,
    ParallelItem,
    KeepClearItem,
    WalkwayWidthItem,
    FixedItem,
)

UnexpressedKind = Literal[
    "out_of_vocabulary",  # a wish no constraint type can express ("cozy", "lots of light")
    "ambiguous_reference",  # "the chair" when several chairs fit
    "unknown_item",  # furniture/feature that isn't in the room ("the piano")
    "partial",  # only part of the request could be expressed
]


class UnexpressedItem(BaseModel):
    model_config = _ITEM_CONFIG

    text: str = Field(description="The user's words that could not be expressed.")
    kind: UnexpressedKind
    reason: str = Field(description="One short sentence the app can show the user.")


class LLMOutput(BaseModel):
    """Root object of the structured output."""

    model_config = _ITEM_CONFIG

    constraints: list[LLMItem]
    unexpressed: list[UnexpressedItem]


ConstraintType = Literal[
    "faces",
    "against_wall",
    "near",
    "distance_at_least",
    "parallel",
    "keep_clear",
    "walkway_width",
    "fixed",
]

# Which typed field ``WireItem.value`` maps to, and whether the type requires it.
VALUE_FIELD: dict[str, tuple[str, bool] | None] = {
    "faces": ("max_angle_deg", False),
    "against_wall": ("max_gap", False),
    "near": ("max_distance", True),
    "distance_at_least": ("min_distance", True),
    "parallel": ("max_angle_deg", False),
    "keep_clear": ("distance", False),
    "walkway_width": ("min_width", False),
    "fixed": None,
}
_TYPED: dict[str, type[_Item]] = {
    t.model_fields["type"].annotation.__args__[0]: t  # type: ignore[union-attr]
    for t in ITEM_TYPES
}


class WireItem(BaseModel):
    """One constraint as the model writes it (flat: every field present, unused = null)."""

    model_config = _ITEM_CONFIG

    type: ConstraintType
    source_text: str = Field(
        description="The user's own words this item comes from, quoted as written."
    )
    a: str | None = Field(description="First asset id; null for keep_clear/walkway_width.")
    b: str | None = Field(description="Second asset id for faces/near/distance/parallel.")
    wall_id: str | None = Field(description="against_wall only: wall id, or null = any wall.")
    opening_id: str | None = Field(description="keep_clear only: door/window/passage id.")
    value: float | None = Field(
        description="The type's number (meters or degrees), or null for the default."
    )
    hard: bool
    restatement: str

    def to_typed(self) -> _Item:
        """The typed item. Raises ``ValueError`` naming a missing required field."""
        cls = _TYPED[self.type]
        data: dict[str, Any] = {
            "type": self.type,
            "source_text": self.source_text,
            "hard": self.hard,
            "restatement": self.restatement,
        }
        vf = VALUE_FIELD[self.type]
        if vf is not None:
            field, required = vf
            if required and self.value is None:
                raise ValueError(f"{self.type} needs a value ({field})")
            data[field] = self.value
        for ref in ("a", "b", "wall_id", "opening_id"):
            if ref not in cls.model_fields:
                continue
            v = getattr(self, ref)
            if v is None and ref != "wall_id":
                raise ValueError(f"{self.type} needs {ref}")
            data[ref] = v
        return cls.model_validate(data)


_WIRE_FIELDS = ("a", "b", "wall_id", "opening_id")
_VALUE_NAMES = {v[0] for v in VALUE_FIELD.values() if v is not None}


def wire_item(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten a typed item dict (``FacesItem`` field names) into the wire shape.

    Used for hand-authored test responses and the eval's oracle answers. Wire-shaped
    dicts pass through unchanged; unknown keys are kept so validation still sees them.
    """
    if "value" in item and not (_VALUE_NAMES & item.keys()):
        return {**{k: None for k in _WIRE_FIELDS}, **item}
    out = {k: v for k, v in item.items() if k not in _VALUE_NAMES}
    out.setdefault("value", None)
    for name in _VALUE_NAMES & item.keys():
        out["value"] = item[name]
    for k in _WIRE_FIELDS:
        out.setdefault(k, None)
    return out


def wire_output(output: dict[str, Any]) -> dict[str, Any]:
    """``wire_item`` applied to every constraint of an output dict."""
    if not isinstance(output.get("constraints"), list):
        return output
    items = [wire_item(c) if isinstance(c, dict) else c for c in output["constraints"]]
    return {**output, "constraints": items}


class WireOutput(BaseModel):
    """Root object of the structured output, as sent by the API."""

    model_config = _ITEM_CONFIG

    constraints: list[WireItem]
    unexpressed: list[UnexpressedItem]


def output_json_schema() -> dict[str, Any]:
    """The JSON Schema sent as ``output_config.format.schema``. Static across requests."""
    return WireOutput.model_json_schema()


# Computed once; identical bytes on every request (schema compilation is cached by the API).
OUTPUT_SCHEMA: dict[str, Any] = output_json_schema()

_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "oneOf",
        "not",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "if",
        "then",
        "else",
    }
)


def schema_problems(schema: dict[str, Any]) -> list[str]:
    """Structured-output compatibility problems in ``schema`` (empty list = OK).

    Checks the documented limits: no ``oneOf``, no numeric/string-length constraints,
    ``additionalProperties: false`` on every object, every property required (so the model
    never silently omits one), and no recursive ``$ref``.
    """
    problems: list[str] = []
    defs = schema.get("$defs", {})

    def walk(node: Any, path: str, stack: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", stack)
            return
        if not isinstance(node, dict):
            return
        for key in node.keys() & _UNSUPPORTED_KEYWORDS:
            problems.append(f"{path}: unsupported keyword {key!r}")
        if "minItems" in node and node["minItems"] not in (0, 1):
            problems.append(f"{path}: minItems other than 0/1")
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: object without additionalProperties: false")
            props = set(node.get("properties", {}))
            missing = props - set(node.get("required", []))
            if missing:
                problems.append(f"{path}: optional properties {sorted(missing)}")
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in stack:
                problems.append(f"{path}: recursive $ref {ref}")
            elif name in defs:
                walk(defs[name], f"$defs.{name}", (*stack, name))
            else:
                problems.append(f"{path}: dangling $ref {ref}")
        for key, value in node.items():
            if key in ("$defs", "$ref"):
                continue
            if key == "properties" and isinstance(value, dict):
                for prop, sub in value.items():
                    walk(sub, f"{path}.{prop}", stack)
            else:
                walk(value, f"{path}.{key}", stack)

    walk(schema, "$", ())
    return problems


def schema_json() -> str:
    """Deterministic serialization of the output schema (for hashing/docs)."""
    return json.dumps(OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":"))
