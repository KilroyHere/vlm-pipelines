"""Rule library: roles, every rule family, precedence (ADR-0006) and idempotence."""

from pathlib import Path

import pytest

from roomlayout.constraints.rules import asset_role, assign_roles, implied_constraints
from roomlayout.schema import (
    AgainstWall,
    Asset,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    LayoutProblem,
    Near,
    Parallel,
    WalkwayWidth,
)

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "problems").glob("*.json"))


def rect(x0, y0, x1, y1):
    return {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def room(openings=()):
    return {
        "id": "r",
        "floor": {"points": [[0, 0], [5, 0], [5, 4], [0, 4]]},
        "ceiling_height": 2.5,
        "walls": [
            {"id": "w_s", "start": [0, 0], "end": [5, 0], "height": 2.5},
            {"id": "w_e", "start": [5, 0], "end": [5, 4], "height": 2.5},
            {"id": "w_n", "start": [5, 4], "end": [0, 4], "height": 2.5},
            {"id": "w_w", "start": [0, 4], "end": [0, 0], "height": 2.5},
        ],
        "openings": list(openings),
    }


DOOR = {
    "id": "door",
    "type": "door",
    "wall_id": "w_s",
    "offset": 0.6,
    "width": 0.9,
    "height": 2.1,
    "swing": {"into_room": True, "hinge": "start"},
}
WINDOW = {
    "id": "win",
    "type": "window",
    "wall_id": "w_n",
    "offset": 2.5,
    "width": 1.2,
    "height": 1.2,
    "sill_height": 0.9,
}
PASSAGE = {
    "id": "arch",
    "type": "opening",
    "wall_id": "w_e",
    "offset": 2.0,
    "width": 1.0,
    "height": 2.1,
}


def a(id, category, w=1.0, d=0.5, h=0.8, **kw):
    out = {"id": id, "category": category, "dimensions": {"width": w, "depth": d, "height": h}}
    out.update(kw)
    return out


def problem(assets, constraints=(), openings=(DOOR, WINDOW), initial=None):
    data = {
        "id": "p",
        "room": room(openings),
        "assets": list(assets),
        "constraints": list(constraints),
    }
    if initial:
        data["initial"] = initial
    return LayoutProblem.model_validate(data)


def ids(cs):
    return {c.id for c in cs}


def by_id(cs):
    return {c.id: c for c in cs}


LIVING = [
    a(
        "sofa",
        "sofa",
        2.2,
        0.95,
        clearance_zones=[
            {"name": "legroom", "purpose": "use", "polygon": rect(-1.0, 0.475, 1.0, 0.975)}
        ],
    ),
    a("loveseat", "sofa", 1.4, 0.9),
    a("tv", "television", 1.4, 0.1, 0.8, mount="wall", elevation=1.0),
    a("stand", "storage", 1.6, 0.4, 0.5, subtype="tv_stand"),
    a("ct", "table", 1.1, 0.6, 0.45, label="coffee table"),
    a("arm", "chair", 0.85, 0.85, label="Armchair"),
    a("lamp_table", "table", 0.4, 0.4, label="side table"),
    a("books", "storage", 0.9, 0.35, 1.8, label="bookcase"),
]


# ------------------------------------------------------------------------------ roles


@pytest.mark.parametrize(
    "asset, expected",
    [
        (a("x", "table", subtype="desk", label="coffee table"), ("table", "desk")),  # subtype wins
        (a("x", "couch"), ("sofa", None)),  # category alias
        (a("x", "table", label="Coffee Table"), ("table", "coffee_table")),  # label words
        (a("desk_chair", "chair", label="chair"), ("chair", "office_chair")),  # id words
        (a("tv_console", "storage", label="low console under the TV"), ("storage", "tv_stand")),
        (a("x", "television", label="TV on stand"), ("television", None)),  # not a tv_stand
        (a("x", "plant", label="potted plant"), ("plant", None)),  # unknown stays unknown
    ],
)
def test_asset_role(asset, expected):
    r = asset_role(Asset.model_validate(asset))
    assert (r.category, r.subtype) == expected


def test_roles_living_room():
    r = assign_roles(problem(LIVING))
    assert r.primary_seating == "sofa"  # largest sofa-category asset
    assert r.secondary_seating == ["loveseat", "arm"]
    assert r.tv == "tv" and r.tv_stand == "stand" and r.coffee_table == "ct"
    assert r.side_tables == ["lamp_table"]
    assert set(r.wall_storage) == {"stand", "books"}


def test_tv_stand_stands_in_for_a_missing_television():
    r = assign_roles(problem([a("sofa", "sofa", 2, 0.9), a("st", "storage", label="TV unit")]))
    assert r.tv == "st"
    assert "rule:faces:sofa:st" in ids(
        implied_constraints(
            problem([a("sofa", "sofa", 2, 0.9), a("st", "storage", label="TV unit")])
        )
    )


# ------------------------------------------------------------------------------ families


def test_every_emitted_constraint_is_attributed_and_valid():
    p = problem(LIVING)
    cs = implied_constraints(p)
    assert len(ids(cs)) == len(cs)
    for c in cs:
        assert c.origin == "rule_library" and c.rationale and c.id.startswith("rule:")
    # the result drops straight into the problem
    LayoutProblem.model_validate({**p.model_dump(), "constraints": [c.model_dump() for c in cs]})
    assert implied_constraints(p) == cs  # deterministic


def test_living_rules():
    cs = by_id(implied_constraints(problem(LIVING)))
    for cid in (
        "rule:faces:sofa:tv",
        "rule:faces:tv:sofa",
        "rule:near:sofa:tv",
        "rule:distance_at_least:sofa:tv",
        "rule:faces:sofa:ct",
        "rule:near:ct:sofa",
        "rule:distance_at_least:ct:sofa",
        "rule:parallel:ct:sofa",
        "rule:faces:arm:ct",
        "rule:faces:loveseat:ct",
        "rule:near:stand:tv",
        "rule:near:lamp_table:loveseat",
    ):
        assert cid in cs, cid
        assert not cs[cid].hard
    # legroom-consistent coffee table distances: the sofa's zone is 0.5 m deep
    assert cs["rule:distance_at_least:ct:sofa"].min_distance == pytest.approx(0.5)
    assert cs["rule:near:ct:sofa"].max_distance == pytest.approx(0.65)
    # viewing distance band is ordered
    assert cs["rule:distance_at_least:sofa:tv"].min_distance < cs["rule:near:sofa:tv"].max_distance
    # wall-mounted TV: no implied against_wall (the solver's wall_mount covers it)
    assert "rule:against_wall:tv" not in cs


def test_coffee_table_without_legroom_zone_uses_defaults():
    cs = by_id(
        implied_constraints(
            problem([a("s", "sofa", 2, 0.9), a("c", "table", subtype="coffee_table")])
        )
    )
    assert cs["rule:distance_at_least:c:s"].min_distance == pytest.approx(0.35)
    assert cs["rule:near:c:s"].max_distance == pytest.approx(0.5)


def test_work_rules_pair_desks_and_chairs_by_name():
    p = problem(
        [
            a("chair_b", "chair"),
            a("desk_a", "table", subtype="desk"),
            a("chair_a", "chair"),
            a("desk_b", "table", subtype="desk"),
        ]
    )
    assert assign_roles(p).desk_pairs == [("desk_a", "chair_a"), ("desk_b", "chair_b")]
    cs = ids(implied_constraints(p))
    for d, c in (("desk_a", "chair_a"), ("desk_b", "chair_b")):
        assert {f"rule:near:{c}:{d}", f"rule:faces:{c}:{d}", f"rule:faces:{d}:{c}"} <= cs


def test_dining_rules_are_soft_and_per_chair():
    chairs = [a(f"c{i}", "chair", 0.45, 0.5, label="dining chair") for i in range(4)]
    cs = implied_constraints(problem([a("t", "table", 1.6, 0.9, subtype="dining_table"), *chairs]))
    m = by_id(cs)
    for i in range(4):
        assert not m[f"rule:near:c{i}:t"].hard and not m[f"rule:faces:c{i}:t"].hard
    assert not any(isinstance(c, DistanceAtLeast) for c in cs)  # no fake distribution rule


def test_bedroom_rules():
    p = problem(
        [
            a("bed", "bed", 1.6, 2.0),
            a("n1", "storage", 0.45, 0.4, label="nightstand"),
            a("n2", "storage", 0.45, 0.4, subtype="nightstand"),
        ]
    )
    cs = by_id(implied_constraints(p))
    assert isinstance(cs["rule:against_wall:bed"], AgainstWall)
    for n in ("n1", "n2"):
        assert f"rule:near:{n}:bed" in cs and f"rule:parallel:{n}:bed" in cs
        assert f"rule:against_wall:{n}" in cs
    sides = cs["rule:distance_at_least:n1:n2"]
    assert sides.min_distance == pytest.approx(0.8 * 1.6)


def test_storage_against_wall_except_pinned_fixed_or_wall_mounted():
    p = problem(
        [
            a("shelf", "storage", label="bookshelf"),
            a("ward", "storage", subtype="wardrobe"),
            a("built_in", "storage", subtype="cabinet", movable=False),
            a("dresser", "storage", subtype="dresser"),
            a("wall_shelf", "storage", subtype="bookshelf", mount="wall", elevation=1.2),
        ],
        [{"type": "fixed", "id": "keep", "a": "dresser", "weight": 0.5}],
        initial={"built_in": {"x": 1, "y": 0.25}, "dresser": {"x": 3, "y": 0.25}},
    )
    walls = {c.a for c in implied_constraints(p) if isinstance(c, AgainstWall)}
    assert walls == {"shelf", "ward"}


def test_safety_rules():
    cs = by_id(implied_constraints(problem([a("x", "chair")], openings=(DOOR, WINDOW, PASSAGE))))
    hard_walk, soft_walk = cs["rule:walkway_width"], cs["rule:walkway_width:preferred"]
    assert hard_walk.hard and hard_walk.min_width == pytest.approx(0.8)
    assert not soft_walk.hard and soft_walk.min_width == pytest.approx(0.9)
    assert cs["rule:keep_clear:door"].hard and cs["rule:keep_clear:arch"].hard
    assert cs["rule:keep_clear:door"].distance == pytest.approx(0.9)
    assert not cs["rule:keep_clear:win"].hard


# ------------------------------------------------------------------------------ precedence


def test_user_faces_on_a_subject_suppresses_all_implied_faces_of_that_subject():
    user = {"type": "faces", "id": "u", "a": "sofa", "b": "arm", "origin": "user"}
    cs = ids(implied_constraints(problem(LIVING, [user])))
    assert not any(c.startswith("rule:faces:sofa:") for c in cs)
    assert "rule:faces:tv:sofa" in cs  # other subjects are untouched


def test_user_distance_on_a_pair_suppresses_implied_near_and_distance_on_it():
    user = {
        "type": "distance_at_least",
        "id": "u",
        "a": "sofa",
        "b": "ct",
        "min_distance": 0.6,
        "origin": "vlm",
    }
    cs = ids(implied_constraints(problem(LIVING, [user])))
    assert "rule:near:ct:sofa" not in cs and "rule:distance_at_least:ct:sofa" not in cs
    assert "rule:faces:sofa:ct" in cs and "rule:parallel:ct:sofa" in cs


def test_user_parallel_suppresses_implied_parallel_on_the_pair():
    user = {"type": "parallel", "id": "u", "a": "sofa", "b": "ct"}
    assert "rule:parallel:ct:sofa" not in ids(implied_constraints(problem(LIVING, [user])))


def test_against_wall_already_constrained_or_fixed_is_skipped():
    users = [
        {"type": "against_wall", "id": "u", "a": "books", "wall_id": "w_w"},
        {"type": "fixed", "id": "f", "a": "stand", "hard": True},
    ]
    cs = ids(implied_constraints(problem(LIVING, users, initial={"stand": {"x": 2, "y": 0.2}})))
    assert "rule:against_wall:books" not in cs and "rule:against_wall:stand" not in cs


def test_no_relation_between_two_pinned_assets():
    users = [
        {"type": "fixed", "id": "f1", "a": "sofa", "hard": True},
        {"type": "fixed", "id": "f2", "a": "tv", "hard": True},
    ]
    cs = ids(
        implied_constraints(
            problem(
                LIVING, users, initial={"sofa": {"x": 2.5, "y": 3.5}, "tv": {"x": 2.5, "y": 0.05}}
            )
        )
    )
    assert not any(":sofa:tv" in c or ":tv:sofa" in c for c in cs)
    assert "rule:faces:sofa:ct" in cs  # the coffee table can still move


def test_existing_safety_rules_are_not_duplicated():
    users = [
        {"type": "walkway_width", "id": "w", "min_width": 1.0, "hard": True},
        {"type": "keep_clear", "id": "k", "opening_id": "door", "distance": 1.2},
    ]
    cs = implied_constraints(problem(LIVING, users))
    assert not any(isinstance(c, WalkwayWidth) for c in cs)
    assert [c.opening_id for c in cs if isinstance(c, KeepClear)] == ["win"]


def test_rule_origin_constraints_do_not_count_as_user_intent():
    # a rule-library constraint someone kept with a *different* id does not suppress rules
    kept = {
        "type": "near",
        "id": "old_rule",
        "a": "ct",
        "b": "sofa",
        "max_distance": 0.4,
        "origin": "rule_library",
    }
    assert "rule:near:ct:sofa" in ids(implied_constraints(problem(LIVING, [kept])))


# ------------------------------------------------------------------------------ idempotence


def _with(p: LayoutProblem, extra) -> LayoutProblem:
    return LayoutProblem.model_validate(
        {**p.model_dump(), "constraints": [c.model_dump() for c in [*p.constraints, *extra]]}
    )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("minimal", [False, True], ids=["full", "minimal"])
def test_idempotent_on_examples(path, minimal):
    p = LayoutProblem.model_validate_json(path.read_text())
    if minimal:
        p = LayoutProblem.model_validate(
            {
                **p.model_dump(),
                "constraints": [c.model_dump() for c in p.constraints if isinstance(c, Fixed)],
            }
        )
    first = implied_constraints(p)
    assert implied_constraints(_with(p, first)) == []
    assert all(
        isinstance(
            c, Faces | Near | DistanceAtLeast | Parallel | AgainstWall | WalkwayWidth | KeepClear
        )
        for c in first
    )


def test_zone_that_admits_a_category_invites_the_piano_bench_first():
    """A piano's bench zone admits chairs. The piano bench (taxonomy v0.6 subtype) is
    invited to it ahead of a stool listed before it, even with a chairless desk in the
    room: a bench is never a desk chair. One member per zone, so the stool stays out."""
    bench_zone = {
        "name": "bench_area",
        "purpose": "use",
        "allowed_categories": ["chair"],
        "polygon": rect(-0.75, 0.3, 0.75, 1.2),
    }
    p = problem(
        [
            a("stool", "chair", 0.4, 0.4, label="stool"),
            a("piano", "piano", 1.5, 0.6, 1.2, clearance_zones=[bench_zone]),
            a("desk", "table", 1.2, 0.6, subtype="desk"),
            a("bench", "chair", 0.8, 0.4, label="piano bench"),
        ]
    )
    r = assign_roles(p)
    assert r.by_id["bench"].subtype == "piano_bench"
    assert r.invited == [("piano", "bench_area", "bench")]
    assert r.desk_pairs == []
    cs = ids(implied_constraints(p))
    assert {"rule:near:bench:piano", "rule:faces:bench:piano", "rule:faces:piano:bench"} <= cs
    assert not any("stool" in c or "bench:desk" in c for c in cs)


def test_piano_bench_without_a_zone_goes_to_the_piano():
    p = problem(
        [
            a("upright", "storage", 1.5, 0.6, 1.2, label="Upright piano"),
            a("pb", "chair", 0.8, 0.4, subtype="piano_bench"),
            a("desk", "table", 1.2, 0.6, subtype="desk"),
        ]
    )
    r = assign_roles(p)
    assert r.invited == [("upright", None, "pb")] and r.desk_pairs == []
    assert {"rule:near:pb:upright", "rule:faces:pb:upright", "rule:faces:upright:pb"} <= ids(
        implied_constraints(p)
    )


def test_plain_bench_without_zone_or_piano_gets_no_relation():
    p = problem(
        [
            a("hall_bench", "chair", 1.2, 0.4, subtype="bench"),
            a("desk", "table", 1.2, 0.6, subtype="desk"),
        ]
    )
    r = assign_roles(p)
    assert r.invited == [] and r.desk_pairs == []
    assert not any("hall_bench" in c for c in ids(implied_constraints(p)))


def test_ottoman_is_a_footrest_for_the_main_seating():
    p = problem(
        [
            a("arm", "chair", 0.85, 0.85, subtype="armchair"),
            a("sofa", "sofa", 2.2, 0.95),
            a("pouf", "chair", 0.5, 0.5, label="pouf"),
        ]
    )
    r = assign_roles(p)
    assert r.footrests == [("pouf", "sofa")]
    assert "pouf" not in r.secondary_seating
    assert "rule:near:pouf:sofa" in ids(implied_constraints(p))
    alone = problem(
        [
            a("arm", "chair", 0.85, 0.85, subtype="armchair"),
            a("pouf", "chair", 0.5, 0.5, subtype="ottoman"),
        ]
    )
    assert assign_roles(alone).footrests == [("pouf", "arm")]


def test_meeting_table_takes_the_generic_chairs():
    p = problem(
        [
            a("t", "table", 1.6, 1.0, label="conference table"),
            a("c1", "chair", 0.5, 0.5),
            a("c2", "chair", 0.5, 0.5),
        ]
    )
    r = assign_roles(p)
    assert r.dining == [("t", ["c1", "c2"])]
    assert {"rule:near:c1:t", "rule:faces:c2:t"} <= ids(implied_constraints(p))


def test_desk_chair_zone_does_not_double_invite():
    chair_zone = {
        "name": "chair_zone",
        "purpose": "use",
        "allowed_categories": ["chair"],
        "polygon": rect(-0.6, 0.3, 0.6, 1.0),
    }
    p = problem(
        [
            a("desk", "table", 1.2, 0.6, subtype="desk", clearance_zones=[chair_zone]),
            a("c", "chair", 0.5, 0.5),
        ]
    )
    r = assign_roles(p)
    assert r.desk_pairs == [("desk", "c")] and r.invited == []
