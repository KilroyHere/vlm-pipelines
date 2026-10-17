"""VLM output schema, prompt and problem summary: API compatibility and determinism."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from roomlayout.constraints.vlm import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    LLMOutput,
    build_request,
    schema_problems,
    summarize_problem,
)
from roomlayout.constraints.vlm.output import ITEM_TYPES, output_json_schema
from roomlayout.constraints.vlm.summary import wall_names
from roomlayout.schema import LayoutProblem

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "problems"
PROBLEM_FILES = sorted(EXAMPLES.glob("*.json"))


def load(name: str) -> LayoutProblem:
    return LayoutProblem.model_validate_json((EXAMPLES / f"{name}.json").read_text())


def _walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


# ------------------------------------------------------------------ output schema


def test_schema_has_no_oneof_anywhere():
    assert "oneOf" not in json.dumps(OUTPUT_SCHEMA)


def test_constraint_items_are_anyof_with_one_branch_per_type():
    items = OUTPUT_SCHEMA["properties"]["constraints"]["items"]
    assert set(items) == {"anyOf"}
    assert len(items["anyOf"]) == len(ITEM_TYPES) == 8
    tags = set()
    for branch in items["anyOf"]:
        name = branch["$ref"].removeprefix("#/$defs/")
        tags.add(OUTPUT_SCHEMA["$defs"][name]["properties"]["type"]["const"])
    assert tags == {
        "faces",
        "against_wall",
        "near",
        "distance_at_least",
        "parallel",
        "keep_clear",
        "walkway_width",
        "fixed",
    }


def test_every_object_forbids_additional_properties_and_requires_all_fields():
    objects = []
    _walk(OUTPUT_SCHEMA, lambda n: objects.append(n) if n.get("type") == "object" else None)
    assert len(objects) >= 10  # root + 8 items + unexpressed
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_no_numeric_or_string_length_constraints():
    banned = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    banned |= {"minLength", "maxLength", "pattern"}
    found = []
    _walk(OUTPUT_SCHEMA, lambda n: found.extend(n.keys() & banned))
    assert found == []


def test_schema_checker_passes_ours_and_catches_violations():
    assert schema_problems(OUTPUT_SCHEMA) == []
    bad = {
        "type": "object",
        "properties": {
            "x": {"oneOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
            "y": {"type": "object", "properties": {"z": {"type": "string"}}},
        },
        "required": ["x"],
    }
    problems = " | ".join(schema_problems(bad))
    assert "oneOf" in problems
    assert "minimum" in problems
    assert "additionalProperties" in problems
    assert "optional properties ['y']" in problems


def test_schema_is_static_across_requests():
    assert output_json_schema() == OUTPUT_SCHEMA
    r1 = build_request(load("living_room_basic"), "sofa facing the TV")
    r2 = build_request(load("dining_room"), "don't block the window", effort="low")
    assert r1["output_config"]["format"] == r2["output_config"]["format"]
    assert r1["output_config"]["format"]["schema"] is OUTPUT_SCHEMA
    assert json.dumps(r1["system"]) == json.dumps(r2["system"])


# ------------------------------------------------------------------ prompt


def _example_outputs() -> list[str]:
    """The JSON answers in the worked examples (a line starting with '{"constraints"')."""
    text = SYSTEM_PROMPT
    starts = [m.start() for m in re.finditer(r'\{"constraints"', text)]
    blobs = []
    for s in starts:
        depth = 0
        for i in range(s, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                blobs.append(text[s : i + 1])
                break
    return blobs


def test_worked_examples_are_valid_output():
    blobs = _example_outputs()
    assert len(blobs) == 3
    for blob in blobs:
        out = LLMOutput.model_validate_json(blob)
        assert out.constraints or out.unexpressed


def test_worked_examples_do_not_add_rule_library_defaults():
    ex1 = LLMOutput.model_validate_json(_example_outputs()[0])
    pairs = {(c.type, getattr(c, "a", None), getattr(c, "b", None)) for c in ex1.constraints}
    assert ("faces", "couch", "tv") in pairs
    assert ("faces", "tv", "couch") not in pairs


def test_system_prompt_is_static_and_versioned():
    assert PROMPT_VERSION
    # Nothing per-request may leak into the cached prefix.
    for marker in ("{summary}", "{text}", "<room>\n", "2026-"):
        assert marker not in SYSTEM_PROMPT
    for must in ("must", "prefer", "ft = 0.3048", "rule library", "Never invent an id"):
        assert must in SYSTEM_PROMPT


def test_system_prompt_is_long_enough_to_cache():
    # Minimum cacheable prefix on claude-opus-5-5 is 512 tokens; ~4 chars/token is a
    # conservative estimate for English prose (exact count: count_tokens, see docs/vlm.md).
    assert len(SYSTEM_PROMPT) / 4 > 2 * 512


# ------------------------------------------------------------------ summary


@pytest.mark.parametrize("path", PROBLEM_FILES, ids=lambda p: p.stem)
def test_summary_mentions_every_id_and_is_deterministic(path):
    problem = LayoutProblem.model_validate_json(path.read_text())
    s = summarize_problem(problem)
    assert s == summarize_problem(LayoutProblem.model_validate_json(path.read_text()))
    for a in problem.assets:
        assert f"- {a.id}: " in s
    for o in problem.room.openings:
        assert f"- {o.id}: {o.type.value} in {o.wall_id}" in s
    for w in problem.room.walls:
        assert f"- {w.id}: " in s


def test_wall_names_follow_interior_normal():
    names = wall_names(load("living_room_basic"))
    assert names == {
        "w_south": "south wall",
        "w_east": "east wall",
        "w_north": "north wall",
        "w_west": "west wall",
    }
    assert wall_names(load("angled_reading_room"))["w_diag"] == "northeast wall"
    l_shape = wall_names(load("l_shaped_living_dining"))
    assert l_shape["w_north"] == "north wall #1" and l_shape["w_north2"] == "north wall #2"


def test_summary_details_needed_for_grounding():
    s = summarize_problem(load("living_room_basic"))
    assert "- w_north: north wall, 5.00 m" in s and "has window_north" in s
    assert "opposite w_south" in s
    assert "sill 0.90 m" in s
    assert 'radiator: "radiator"' in s
    assert '- sofa: sofa, label "3-seat sofa", 2.20 W x 0.95 D x 0.85 H m' in s
    assert "- faces(sofa, tv_stand, max_angle_deg=20) hard, from user" in s
    piano = summarize_problem(load("fixed_piano_room"))
    assert "builtin_cabinet: storage" in piano and "immovable (built-in)" in piano
    assert "can be kept where it is" in piano
    tv = summarize_problem(load("wall_tv_living"))
    assert "wall-mounted at 1.00 m" in tv


def test_summary_without_constraints_says_none():
    p = load("bedroom_basic").model_copy(update={"constraints": []})
    assert summarize_problem(p).endswith("Constraints already in the problem:\n- none")
