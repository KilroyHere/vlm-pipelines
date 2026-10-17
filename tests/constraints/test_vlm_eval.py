"""The VLM eval set and scorer (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomlayout.constraints.vlm import eval as vlm_eval
from roomlayout.constraints.vlm.eval import (
    DEFAULT_CASES,
    _matches,
    aggregate,
    load_cases,
    load_problem,
    main,
    replay,
)
from roomlayout.schema import ROOMPLAN_CATEGORIES

FIXTURE = Path(__file__).parent / "vlm_fixtures" / "replay"
CONSTRAINT_TYPES = {
    "faces",
    "against_wall",
    "near",
    "distance_at_least",
    "parallel",
    "keep_clear",
    "walkway_width",
    "fixed",
}
REQUIRED_TAGS = {
    "direct",
    "relative",
    "units",
    "hard",
    "soft",
    "ambiguous",
    "negative",
    "oov",
    "conflict",
    "nonexistent",
}


@pytest.fixture(scope="module")
def cases():
    return load_cases(DEFAULT_CASES)


def test_case_set_size_and_coverage(cases):
    assert len(cases) >= 50
    tags = {t for c in cases for t in c["tags"]}
    assert REQUIRED_TAGS <= tags
    for tag in REQUIRED_TAGS:
        assert sum(tag in c["tags"] for c in cases) >= 3, tag
    assert len({c["problem"] for c in cases}) >= 10  # spread over the example rooms
    assert sum(bool(c["expected"]["unexpressed"]) for c in cases) >= 15


def test_cases_reference_real_ids(cases):
    for case in cases:
        problem = load_problem(case)
        assert problem.constraints == []  # eval starts from the user's words only
        assets = {a.id for a in problem.assets}
        openings = {o.id for o in problem.room.openings}
        walls = {w.id for w in problem.room.walls}
        for e in case["expected"]["constraints"]:
            assert e["type"] in CONSTRAINT_TYPES, case["id"]
            for k in ("a", "b"):
                if k in e:
                    assert e[k] in assets, (case["id"], e[k])
            if e.get("wall_id") is not None:
                assert e["wall_id"] in walls, case["id"]
            if "opening_id" in e:
                assert e["opening_id"] in openings, case["id"]
            if e["type"] == "fixed":
                assert e["a"] in problem.initial, case["id"]
        for u in case["expected"]["unexpressed"]:
            assert u["any"] and all(isinstance(w, str) for w in u["any"])


def test_nonexistent_cases_name_things_not_in_the_room(cases):
    for case in (c for c in cases if "nonexistent" in c["tags"]):
        problem = load_problem(case)
        words = " ".join(u for x in case["expected"]["unexpressed"] for u in x["any"]).lower()
        names = {a.id for a in problem.assets} | {a.label.lower() for a in problem.assets}
        names |= {a.category for a in problem.assets}
        assert not any(w in names for w in words.split()), case["id"]
        assert set(ROOMPLAN_CATEGORIES)  # taxonomy importable from the contract


def test_matching_rules():
    near = {"type": "near", "a": "x", "b": "y", "max_distance": 0.5, "hard": True}
    assert _matches({"type": "near", "a": "y", "b": "x"}, near)  # symmetric
    faces_yx = {**near, "type": "faces", "a": "y", "b": "x"}
    assert not _matches({"type": "faces", "a": "x", "b": "y"}, faces_yx)  # directional
    assert _matches({"type": "near", "a": "x", "b": "y", "max_distance": 0.4}, near)
    assert not _matches({"type": "near", "a": "x", "b": "y", "max_distance": 0.2}, near)
    assert _matches(
        {"type": "near", "a": "x", "b": "y", "max_distance": 0.2, "tol": {"max_distance": 0.3}},
        near,
    )
    wall_any = {"type": "against_wall", "a": "x", "wall_id": None}
    assert _matches(wall_any, {"type": "against_wall", "a": "x", "wall_id": None})
    assert not _matches(wall_any, {"type": "against_wall", "a": "x", "wall_id": "w_n"})
    assert _matches(
        {"type": "against_wall", "a": "x"}, {"type": "against_wall", "a": "x", "wall_id": "w_n"}
    )


def test_scorer_on_replay_fixture(cases):
    """Fixture responses are SYNTHETIC (hand-authored), see the files' "note"."""
    scores, missing, notes = replay(FIXTURE, cases)
    assert sorted(s.case_id for s in scores) == ["lr-01", "lr-02", "lr-06", "lr-11"]
    assert len(missing) == len(cases) - 4 and notes == []
    m = aggregate(scores, timing=True)
    assert m["precision"] == pytest.approx(3 / 4)  # lr-11 adds faces(tv_stand, sofa)
    assert m["recall"] == pytest.approx(1.0)
    assert m["hallucinated_id_rate"] == pytest.approx(1 / 5)  # floor_lamp, before repair
    assert m["unexpressed_recall"] == pytest.approx(2 / 3)  # "light" not reported
    assert m["hard_agreement"] == pytest.approx(1.0)
    assert m["exact_cases"] == 2 and m["repair_rounds"] == 1
    assert m["status"] == {"ok": 4}
    assert m["latency_s"] is None  # synthetic fixture has no recorded latency
    assert m["cost_usd"]["total"] == pytest.approx(5 * (900 * 4 + 600 * 20 + 5200 * 0.2) / 1e6)
    assert m["by_tag"]["no_defaults"]["precision"] == pytest.approx(0.5)
    by_id = {s.case_id: s for s in scores}
    assert by_id["lr-11"].false_positives[0]["a"] == "tv_stand"
    assert by_id["lr-02"].missed_unexpressed == [["light"]]


def test_cli_replay(capsys, tmp_path):
    out = tmp_path / "metrics.json"
    assert main(["--replay", str(FIXTURE), "--json", str(out), "-v"]) == 0
    text = capsys.readouterr().out
    assert "precision" in text and "lr-11" in text
    assert out.exists()


def test_cli_live_without_credentials_exits_cleanly(monkeypatch, tmp_path):
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("must not construct a client without credentials")

    monkeypatch.setattr("anthropic.Anthropic", boom)
    with pytest.raises(SystemExit) as exc:
        main(["--live", "--only", "lr-01", "--out", str(tmp_path / "run")])
    assert "No Anthropic API credentials" in str(exc.value)
    assert not (tmp_path / "run").exists()
    assert vlm_eval.PROMPT_VERSION


def _oracle_answer(case):
    """An answer that does exactly what the case expects (required items only)."""
    nulls = {
        "faces": ("max_angle_deg",),
        "parallel": ("max_angle_deg",),
        "against_wall": ("wall_id", "max_gap"),
        "keep_clear": ("distance",),
        "walkway_width": ("min_width",),
    }
    items = []
    for e in case["expected"]["constraints"]:
        if e.get("optional"):
            continue
        item = {k: v for k, v in e.items() if k not in ("optional", "tol")}
        for k in nulls.get(e["type"], ()):
            item.setdefault(k, None)
        item.setdefault("hard", True)
        item.update(source_text=case["text"], restatement="oracle")
        items.append(item)
    unexpressed = [
        {"text": u["any"][0], "kind": u.get("kind", "out_of_vocabulary"), "reason": "oracle"}
        for u in case["expected"]["unexpressed"]
    ]
    return {"constraints": items, "unexpressed": unexpressed}


def test_oracle_answers_pass_validation_and_score_perfectly(cases):
    """Expectations must be expressible: nothing a correct model says gets dropped."""
    from roomlayout.constraints.vlm import parse_request
    from roomlayout.constraints.vlm.eval import score_case
    from roomlayout.constraints.vlm.offline import offline_client, synthetic_response

    scores = []
    for case in cases:
        client = offline_client([synthetic_response(_oracle_answer(case))])
        result = parse_request(load_problem(case), case["text"], client=client, repair=False)
        assert result.status == "ok" and result.dropped == [], (case["id"], result.dropped)
        scores.append(score_case(case, result, None))
    m = aggregate(scores, timing=False)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["unexpressed_recall"] == 1.0 and m["hallucinated_id_rate"] == 0.0
    assert m["exact_cases"] == len(cases)
