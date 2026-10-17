"""Sensibility checks on hand-placed layouts: a sensible one passes, each defect is named."""

import math

import pytest

from roomlayout.constraints.checks import check_layout, failed
from roomlayout.schema import Layout, LayoutProblem, Placement, Pose2D

N, S, E, W = 0.0, math.pi, -math.pi / 2, math.pi / 2  # yaw facing +y, -y, +x, -x


def room():
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
        "openings": [
            {
                "id": "door",
                "type": "door",
                "wall_id": "w_s",
                "offset": 0.6,
                "width": 0.9,
                "height": 2.1,
            }
        ],
    }


def a(id, category, w, d, h=0.8, **kw):
    return {
        "id": id,
        "category": category,
        "dimensions": {"width": w, "depth": d, "height": h},
        **kw,
    }


def problem(assets):
    return LayoutProblem.model_validate({"id": "p", "room": room(), "assets": assets})


def layout(poses):
    return Layout(
        id="L",
        cost=0,
        feasible=True,
        placements=[
            Placement(asset_id=k, pose=Pose2D(x=x, y=y, yaw=t)) for k, (x, y, t) in poses.items()
        ],
    )


LIVING = problem(
    [
        a("sofa", "sofa", 2.2, 0.95),
        a("tv", "television", 1.4, 0.1, 0.8, mount="wall", elevation=1.0),
        a("stand", "storage", 1.6, 0.4, 0.5, subtype="tv_stand"),
        a("ct", "table", 1.1, 0.6, 0.45, subtype="coffee_table"),
        a("arm", "chair", 0.85, 0.85, subtype="armchair"),
        a("books", "storage", 0.9, 0.35, 1.8, subtype="bookshelf"),
    ]
)
GOOD_LIVING = {
    "sofa": (2.5, 3.5, S),
    "tv": (2.5, 0.05, N),
    "stand": (2.5, 0.2, N),
    "ct": (2.5, 2.2, N),
    "arm": (0.8, 2.2, E),
    "books": (4.825, 2.0, W),
}


def test_sensible_living_room_passes_everything():
    res = check_layout(LIVING, layout(GOOD_LIVING))
    names = {r.name for r in res}
    assert {
        "seating_faces_tv[sofa:tv]",
        "tv_faces_seating[tv:sofa]",
        "coffee_table_between_seating_and_tv[ct]",
        "tv_line_of_sight[sofa:tv]",
        "secondary_seating_oriented[arm]",
        "storage_against_wall[books]",
        "storage_against_wall[stand]",
        "door_unblocked[door]",
    } <= names
    assert failed(res) == []


@pytest.mark.parametrize(
    "change, expected_failures",
    [
        (
            {"ct": (1.0, 2.6, N)},
            {
                "coffee_table_between_seating_and_tv[ct]",
                "coffee_table_in_front_of_seating[ct:sofa]",
            },
        ),
        ({"tv": (2.5, 0.05, S)}, {"tv_faces_seating[tv:sofa]"}),
        (
            {"sofa": (2.5, 3.5, E)},
            {"seating_faces_tv[sofa:tv]", "coffee_table_in_front_of_seating[ct:sofa]"},
        ),
        ({"books": (2.5, 1.3, N)}, {"tv_line_of_sight[sofa:tv]", "storage_against_wall[books]"}),
        ({"arm": (0.8, 2.2, W)}, {"secondary_seating_oriented[arm]"}),
        ({"arm": (0.6, 0.45, N)}, {"door_unblocked[door]"}),
    ],
)
def test_each_living_room_defect_is_named(change, expected_failures):
    res = check_layout(LIVING, layout({**GOOD_LIVING, **change}))
    assert {r.name for r in failed(res)} == expected_failures


def test_viewing_distance():
    p = problem([a("sofa", "sofa", 2.0, 0.9), a("tv", "television", 1.2, 0.4, 0.7)])
    res = {r.name: r for r in check_layout(p, layout({"sofa": (2.5, 1.0, N), "tv": (2.5, 1.8, S)}))}
    assert not res["tv_viewing_distance[sofa:tv]"].passed  # 0.15 m: far too close
    assert not res["tv_against_wall[tv]"].passed


def test_desk_chair_at_front_not_at_the_end():
    p = problem(
        [
            a("desk", "table", 1.2, 0.6, subtype="desk"),
            a("desk_chair", "chair", 0.5, 0.5, label="chair"),
        ]
    )
    good = check_layout(p, layout({"desk": (2.5, 0.3, N), "desk_chair": (2.5, 0.85, S)}))
    assert failed(good) == []
    bad = check_layout(p, layout({"desk": (2.5, 0.3, N), "desk_chair": (3.4, 0.3, W)}))
    assert {r.name for r in failed(bad)} == {"desk_chair_at_desk_front[desk:desk_chair]"}


BEDROOM = problem(
    [
        a("bed", "bed", 1.6, 2.0, 0.5),
        a("n1", "storage", 0.45, 0.4, 0.55, subtype="nightstand"),
        a("n2", "storage", 0.45, 0.4, 0.55, label="bedside table"),
    ]
)


def test_bedroom_checks():
    good = {"bed": (2.5, 3.0, S), "n1": (1.45, 3.8, S), "n2": (3.55, 3.8, S)}
    assert failed(check_layout(BEDROOM, layout(good))) == []
    same_side = {**good, "n2": (0.95, 3.8, S)}
    assert {r.name for r in failed(check_layout(BEDROOM, layout(same_side)))} == {
        "nightstands_on_both_sides[bed]"
    }
    at_foot = {**good, "n1": (1.45, 2.2, S)}
    assert {r.name for r in failed(check_layout(BEDROOM, layout(at_foot)))} == {
        "nightstand_beside_bed_head[n1]"
    }
    floating = {**good, "bed": (2.5, 2.5, S)}
    assert "bed_head_against_wall[bed]" in {
        r.name for r in failed(check_layout(BEDROOM, layout(floating)))
    }


def test_dining_chairs_at_the_table():
    p = problem(
        [
            a("t", "table", 1.6, 0.9, subtype="dining_table"),
            a("c1", "chair", 0.45, 0.5, label="dining chair"),
            a("c2", "chair", 0.45, 0.5, label="dining chair"),
        ]
    )
    good = {"t": (2.5, 2.0, N), "c1": (2.5, 2.75, S), "c2": (2.5, 1.25, N)}
    assert failed(check_layout(p, layout(good))) == []
    far = {**good, "c1": (2.5, 3.5, S)}
    assert {r.name for r in failed(check_layout(p, layout(far)))} == {"dining_chair_at_table[c1]"}


def test_side_table_by_seating():
    p = problem(
        [
            a("arm", "chair", 0.85, 0.85, subtype="armchair"),
            a("st", "table", 0.4, 0.4, subtype="side_table"),
        ]
    )
    ok = {r.name: r.passed for r in check_layout(p, layout({"arm": (1, 2, E), "st": (1, 2.7, N)}))}
    assert ok["side_table_by_seating[st]"]
    far = {r.name: r.passed for r in check_layout(p, layout({"arm": (1, 2, E), "st": (4, 3, N)}))}
    assert not far["side_table_by_seating[st]"]


def test_checks_only_cover_present_furniture():
    p = problem([a("plant", "plant", 0.4, 0.4)])
    assert [r.name for r in check_layout(p, layout({"plant": (2, 2, 0)}))] == [
        "door_unblocked[door]"
    ]


def test_invited_member_in_zone():
    zone = {
        "name": "bench_area",
        "purpose": "use",
        "allowed_categories": ["chair"],
        "polygon": {"points": [[-0.75, 0.3], [0.75, 0.3], [0.75, 1.2], [-0.75, 1.2]]},
    }
    p = problem(
        [
            a("piano", "piano", 1.5, 0.6, 1.2, clearance_zones=[zone]),
            a("bench", "chair", 0.8, 0.4, 0.5),
        ]
    )
    at = {
        r.name: r.passed
        for r in check_layout(p, layout({"piano": (2, 3.7, S), "bench": (2, 2.9, N)}))
    }
    assert at["invited_at_zone[piano:bench]"]
    away = {
        r.name: r.passed
        for r in check_layout(p, layout({"piano": (2, 3.7, S), "bench": (4, 1, N)}))
    }
    assert not away["invited_at_zone[piano:bench]"]
