"""parse_request end to end, offline.

Every test drives a real ``anthropic.Anthropic`` client whose transport serves canned
Messages API JSON (``offline_client``), so request serialization and response parsing are
the SDK's own. All responses here are SYNTHETIC (hand-authored with ``synthetic_response``),
not recordings from the API.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from roomlayout.constraints.vlm import (
    FALLBACK_BETA,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    LLMOutput,
    parse_request,
)
from roomlayout.constraints.vlm.extract import echo_content
from roomlayout.constraints.vlm.offline import OfflineExhausted, offline_client
from roomlayout.constraints.vlm.offline import synthetic_response as R
from roomlayout.schema import Faces, LayoutProblem

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "problems"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def load(name: str, keep: bool = False) -> LayoutProblem:
    p = LayoutProblem.model_validate_json((EXAMPLES / f"{name}.json").read_text())
    return p if keep else p.model_copy(update={"constraints": []})


def faces(a, b, hard=True, src="sofa facing the TV", angle=None):
    return {
        "source_text": src,
        "type": "faces",
        "a": a,
        "b": b,
        "max_angle_deg": angle,
        "hard": hard,
        "restatement": f"{a} faces {b}.",
    }


def near(a, b, d, hard=True, src="next to"):
    return {
        "source_text": src,
        "type": "near",
        "a": a,
        "b": b,
        "max_distance": d,
        "hard": hard,
        "restatement": f"{a} near {b}.",
    }


def out(*constraints, unexpressed=()):
    return {"constraints": list(constraints), "unexpressed": list(unexpressed)}


def run(problem, text, responses, **kw):
    requests: list[dict] = []
    client = offline_client(responses, requests=requests)
    return parse_request(problem, text, client=client, **kw), requests


# ------------------------------------------------------------------ request shape


def test_request_parameters():
    result, reqs = run(load("living_room_basic"), "sofa facing the TV", [R(out())])
    assert result.status == "ok"
    assert len(reqs) == 1
    req = reqs[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["anthropic-beta"] == FALLBACK_BETA
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16000
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert body["fallbacks"] == "default"
    assert "tools" not in body and "temperature" not in body
    # Static system prompt first, with the cache breakpoint; per-request text after it.
    assert body["system"] == [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    [msg] = body["messages"]
    assert msg["role"] == "user"
    [block] = msg["content"]
    assert block["type"] == "text"
    assert "<room>" in block["text"] and "sofa facing the TV" in block["text"]
    assert "- tv_stand: television" in block["text"]


def test_effort_and_fallbacks_configurable():
    _, reqs = run(load("living_room_basic"), "x", [R(out())], effort="high", fallbacks=False)
    body = reqs[0]["body"]
    assert body["output_config"]["effort"] == "high"
    assert "fallbacks" not in body
    assert "anthropic-beta" not in reqs[0]["headers"]
    with pytest.raises(ValueError, match="effort"):
        run(load("living_room_basic"), "x", [R(out())], effort="extreme")


def test_image_is_base64_block_before_text():
    _, reqs = run(load("living_room_basic"), "x", [R(out())], image_png=PNG)
    content = reqs[0]["body"]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text", "text"]
    src = content[0]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/png"
    assert base64.standard_b64decode(src["data"]) == PNG
    with pytest.raises(ValueError, match="PNG"):
        run(load("living_room_basic"), "x", [R(out())], image_png=b"GIF89a")


# ------------------------------------------------------------------ conversion


def test_conversion_to_contract():
    response = out(
        faces("sofa", "tv_stand", src="I want the sofa facing the TV"),
        {
            "source_text": "ideally the bookshelf on the west wall",
            "type": "against_wall",
            "a": "bookshelf",
            "wall_id": "w_west",
            "max_gap": None,
            "hard": False,
            "restatement": "If possible, the bookshelf goes against the west wall.",
        },
        {
            "source_text": "don't block the window",
            "type": "keep_clear",
            "opening_id": "window_north",
            "distance": None,
            "hard": True,
            "restatement": "Nothing may stand in front of the window.",
        },
        {
            "source_text": "a meter to walk",
            "type": "walkway_width",
            "min_width": 1.0,
            "hard": True,
            "restatement": "Walkways must be at least 1 m wide.",
        },
        unexpressed=[{"text": "cozy", "kind": "out_of_vocabulary", "reason": "a style"}],
    )
    problem = load("living_room_basic")
    result, _ = run(problem, "...", [R(response)])
    assert result.status == "ok"
    assert [c.id for c in result.constraints] == [
        "vlm_faces_sofa_tv_stand",
        "vlm_against_wall_bookshelf_w_west",
        "vlm_keep_clear_window_north",
        "vlm_walkway_width",
    ]
    c0 = result.constraints[0]
    assert isinstance(c0, Faces) and c0.origin == "vlm" and c0.hard
    assert c0.rationale == "I want the sofa facing the TV"
    assert c0.max_angle_deg == 20.0  # null -> contract default
    # a soft request weighs as much as the strongest default rule (ADR-0007 amendment)
    from roomlayout.constraints.rules import USER_SOFT_WEIGHT

    assert result.constraints[1].hard is False
    assert result.constraints[1].weight == USER_SOFT_WEIGHT
    assert result.constraints[2].distance == 0.6
    assert result.restatements[1] == "If possible, the bookshelf goes against the west wall."
    assert len(result.restatements) == len(result.constraints)
    assert result.unexpressed[0].text == "cozy"
    assert result.dropped == [] and result.warnings == []
    # The result is a valid LayoutProblem.constraints.
    LayoutProblem.model_validate(
        {**problem.model_dump(), "constraints": [c.model_dump() for c in result.constraints]}
    )
    assert len(result.attempts) == 1


def test_ids_unique_against_existing_and_each_other():
    problem = load("living_room_basic", keep=True)
    problem.constraints.append(
        Faces(id="vlm_faces_armchair_tv_stand", a="armchair", b="tv_stand", origin="vlm")
    )
    response = out(
        faces("armchair", "tv_stand", angle=45),
        faces("armchair", "tv_stand", angle=30),
    )
    result, _ = run(problem, "...", [R(response)])
    assert [c.id for c in result.constraints] == [
        "vlm_faces_armchair_tv_stand_2",
        "vlm_faces_armchair_tv_stand_3",
    ]


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "item, code",
    [
        (faces("sofa", "piano"), "unknown_asset"),
        (faces("sofa", "sofa"), "same_asset"),
        (near("sofa", "tv_stand", 10.0), "bad_value"),  # 10 m in a 5 x 4 m room
        (near("sofa", "tv_stand", -0.2), "bad_value"),
        (faces("sofa", "tv_stand", angle=170), "bad_value"),
        (
            {
                "source_text": "keep the patio door clear",
                "type": "keep_clear",
                "opening_id": "patio_door",
                "distance": None,
                "hard": True,
                "restatement": "x",
            },
            "unknown_opening",
        ),
        (
            {
                "source_text": "on the brick wall",
                "type": "against_wall",
                "a": "sofa",
                "wall_id": "w_brick",
                "max_gap": None,
                "hard": True,
                "restatement": "x",
            },
            "unknown_wall",
        ),
        (
            {
                "source_text": "paths of 5 cm",
                "type": "walkway_width",
                "min_width": 0.05,
                "hard": True,
                "restatement": "x",
            },
            "bad_value",
        ),
    ],
)
def test_invalid_items_dropped_with_reason(item, code):
    result, reqs = run(
        load("living_room_basic"), "...", [R(out(item, faces("sofa", "tv_stand")))], repair=False
    )
    assert len(reqs) == 1
    assert [c.id for c in result.constraints] == ["vlm_faces_sofa_tv_stand"]
    [d] = result.dropped
    assert d.code == code and d.repairable and d.reason
    assert d.item["type"] == item["type"]


def test_fixed_needs_current_position():
    fixed = {"source_text": "keep it", "type": "fixed", "a": "sofa", "hard": True}
    fixed["restatement"] = "x"
    result, _ = run(load("living_room_basic"), "...", [R(out(fixed))])
    assert result.constraints == []
    assert result.dropped[0].code == "no_current_position"
    assert not result.dropped[0].repairable
    # With an initial pose (piano room) it is accepted.
    fixed["a"] = "piano"
    result, _ = run(load("fixed_piano_room"), "...", [R(out(fixed))])
    assert [c.type for c in result.constraints] == ["fixed"]


def test_duplicates_and_existing_constraints_dropped_without_repair():
    problem = load("living_room_basic", keep=True)  # has hard faces(sofa, tv_stand)
    response = out(
        faces("sofa", "tv_stand"),  # already present (same numbers, same strength)
        near("armchair", "bookshelf", 0.1),
        near("bookshelf", "armchair", 0.1),  # symmetric duplicate
    )
    result, reqs = run(problem, "...", [R(response)])
    assert len(reqs) == 1
    assert [d.code for d in result.dropped] == ["already_present", "duplicate"]
    assert [c.id for c in result.constraints] == ["vlm_near_armchair_bookshelf"]


def test_conflicting_requests_kept_and_flagged():
    response = out(
        near("sofa", "tv_stand", 1.0),
        {
            "source_text": "at least 2.5 m away",
            "type": "distance_at_least",
            "a": "tv_stand",
            "b": "sofa",
            "min_distance": 2.5,
            "hard": True,
            "restatement": "x",
        },
    )
    result, _ = run(load("living_room_basic"), "...", [R(response)])
    assert len(result.constraints) == 2
    assert len(result.warnings) == 1 and "within 1 m" in result.warnings[0]


# ------------------------------------------------------------------ repair round


def test_repair_round_is_append_only_and_replaces_answer():
    first = out(near("floor_lamp", "armchair", 0.1, src="lamp beside the armchair"))
    second = out(
        unexpressed=[{"text": "lamp", "kind": "unknown_item", "reason": "No lamp in the room."}]
    )
    result, reqs = run(load("living_room_basic"), "lamp by the armchair", [R(first), R(second)])
    assert len(reqs) == 2
    m1, m2 = reqs[0]["body"]["messages"], reqs[1]["body"]["messages"]
    assert m2[: len(m1)] == m1  # prefix unchanged -> prompt cache + thinking stay valid
    assert [m["role"] for m in m2] == ["user", "assistant", "user"]
    echoed = m2[1]["content"]
    assert [b["type"] for b in echoed] == ["thinking", "text"]
    assert echoed[0]["signature"] == "synthetic-signature-0"
    assert "unknown asset id 'floor_lamp'" in m2[2]["content"][0]["text"]
    assert reqs[1]["body"]["system"] == reqs[0]["body"]["system"]
    assert reqs[1]["body"]["output_config"] == reqs[0]["body"]["output_config"]

    assert result.status == "ok"
    assert result.constraints == [] and result.dropped == []
    assert result.unexpressed[0].kind == "unknown_item"
    assert [a.kind for a in result.attempts] == ["initial", "repair"]
    assert result.attempts[0].dropped[0].code == "unknown_asset"
    assert result.usage.output_tokens == 1200  # summed over both calls


def test_at_most_one_repair_round():
    bad = out(near("floor_lamp", "armchair", 0.1))
    result, reqs = run(load("living_room_basic"), "...", [R(bad), R(bad), R(bad)])
    assert len(reqs) == 2
    assert result.dropped[0].code == "unknown_asset"


def test_repair_disabled():
    result, reqs = run(
        load("living_room_basic"), "...", [R(out(near("lamp", "sofa", 0.1)))], repair=False
    )
    assert len(reqs) == 1 and result.dropped[0].code == "unknown_asset"


def test_failed_repair_keeps_first_answer():
    first = out(faces("sofa", "tv_stand"), near("lamp", "sofa", 0.1))
    result, _ = run(load("living_room_basic"), "...", [R(first), R(None, stop_reason="max_tokens")])
    assert result.status == "ok"
    assert [c.id for c in result.constraints] == ["vlm_faces_sofa_tv_stand"]
    assert "Repair round failed" in result.warnings[0]
    assert result.attempts[1].status == "truncated"


# ------------------------------------------------------------------ stop reasons


def test_refusal_returns_no_constraints_and_no_repair():
    refusal = R(None, stop_reason="refusal", stop_details={"type": "refusal", "category": None})
    result, reqs = run(load("living_room_basic"), "...", [refusal])
    assert len(reqs) == 1
    assert result.status == "refused" and result.constraints == []
    assert "declined" in result.message
    assert result.attempts[0].refusal_category is None


def test_refusal_with_partial_text_is_not_parsed():
    partial = R('{"constraints": [{"source_text": "x"', stop_reason="refusal")
    partial["stop_details"] = {"type": "refusal", "category": "cyber", "explanation": None}
    result, _ = run(load("living_room_basic"), "...", [partial])
    assert result.status == "refused" and result.attempts[0].refusal_category == "cyber"


def test_max_tokens_is_truncated():
    cut = R('{"constraints": [{"source_text": "sofa fa', stop_reason="max_tokens")
    result, _ = run(load("living_room_basic"), "...", [cut])
    assert result.status == "truncated" and result.constraints == []
    assert "max_tokens" in result.message


def test_invalid_json_is_reported():
    result, _ = run(load("living_room_basic"), "...", [R('{"constraints": 3}')])
    assert result.status == "invalid_output" and result.constraints == []


def test_server_side_fallback_is_recorded_and_echo_rule():
    served = R(
        out(faces("sofa", "tv_stand"), near("lamp", "sofa", 0.1)),
        model="claude-opus-5",
        fallback_from="claude-opus-5-5",
    )
    result, reqs = run(
        load("living_room_basic"), "...", [served, R(out(faces("sofa", "tv_stand")))]
    )
    a = result.attempts[0]
    assert a.fallback_used and a.served_by == "claude-opus-5"
    assert a.cost_usd == pytest.approx((900 * 5 + 600 * 25 + 5200 * 0.5) / 1e6)
    # A fallback block first means nothing precedes it to omit.
    assert [b["type"] for b in reqs[1]["body"]["messages"][1]["content"]] == [
        "fallback",
        "thinking",
        "text",
    ]


def test_echo_omits_thinking_before_last_fallback():
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "a"},
        {"type": "fallback", "from": {"model": "x"}, "to": {"model": "y"}},
        {"type": "thinking", "thinking": "", "signature": "b"},
        {"type": "text", "text": "{}"},
    ]
    assert [b.get("signature", b["type"]) for b in echo_content(blocks)] == [
        "fallback",
        "b",
        "text",
    ]


def test_usage_and_cost():
    result, _ = run(load("living_room_basic"), "...", [R(out())])
    assert result.usage.cache_read_input_tokens == 5200
    assert result.cost_usd == pytest.approx((900 * 4 + 600 * 20 + 5200 * 0.2) / 1e6)
    assert result.latency_s >= 0


def test_offline_client_refuses_unplanned_calls():
    client = offline_client([])
    with pytest.raises(OfflineExhausted):
        parse_request(load("living_room_basic"), "x", client=client)


def test_result_round_trips_as_json():
    result, _ = run(load("living_room_basic"), "...", [R(out(faces("sofa", "tv_stand")))])
    again = type(result).model_validate_json(result.model_dump_json())
    assert again.constraints == result.constraints


def test_llm_output_model_rejects_unknown_type_tag():
    with pytest.raises(ValueError):
        LLMOutput.model_validate({"constraints": [{"type": "between"}], "unexpressed": []})
