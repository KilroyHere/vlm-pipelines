"""Product knowledge per furniture category: RoomPlan mapping, front heuristic weights,
default clearance zones, mount and movability.

Category names follow Apple RoomPlan's ``CapturedRoom.Object.Category`` where possible
(verified 2026-10-16 against developer.apple.com: bathtub, bed, chair, dishwasher,
fireplace, oven, refrigerator, sink, sofa, stairs, storage, stove, table, television,
toilet, washerDryer). Finer types (sectional, dining_chair, desk, bookshelf, ...) map onto
one of those and keep their own clearance defaults. ``Asset.category`` gets the RoomPlan
category; the fine type goes into ``Asset.label`` and the normalization report.

Clearance depths are conventional interior-design rules of thumb, not measurements:

- Sofa/armchair legroom 0.40 m: the usual advice is 14-18 in (35-45 cm) between the seat
  front and a coffee table. We use the middle, so a coffee table may sit just outside it.
- Bed access 0.60 m on each long side and at the foot: the minimum usually quoted is
  24 in (~60 cm). 30-36 in (75-90 cm) is the comfortable value.
- Dining table 0.75 m on every side for seated diners and pulling chairs out. Common
  advice is 30-36 in (75-90 cm) from the table edge (NKBA kitchen planning guidelines
  give 32-36 in from table edge to wall).
- Desk 0.90 m in front for the chair: roughly 36 in of chair space.
- Dining/office chair 0.45 m behind: the room a chair needs to slide back. A dining chair
  takes about half of the 36 in table-edge allowance.
- Bookshelf / open storage 0.60 m in front to stand and reach.
- Dresser 0.90 m and wardrobe 0.90 m in front: drawer pull-out or door swing plus a
  standing person (commonly 36 in).
- Kitchen/laundry appliances 0.90 m in front for doors (dishwasher and oven doors open
  about 0.5-0.6 m, plus standing room). Toilet 0.60 m in front (residential codes ask
  for ~21 in / 53 cm, NKBA recommends 30 in). Sink/bathtub 0.75 m. Fireplace 0.50 m of
  hearth.
- Television: no clearance zone. The viewing zone is not a keep-out area for other
  footprints (a coffee table may stand in it). Viewing is the ``faces`` constraint's job.

Sources: commonly cited designer guidance and NKBA Kitchen & Bath Planning Guidelines.
These were *not* re-verified against the primary texts in this session. Treat them as
defaults to tune, which is why they live in one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from roomlayout.schema import Mount

ROOMPLAN_CATEGORIES = (
    "bathtub",
    "bed",
    "chair",
    "dishwasher",
    "fireplace",
    "oven",
    "refrigerator",
    "sink",
    "sofa",
    "stairs",
    "storage",
    "stove",
    "table",
    "television",
    "toilet",
    "washerDryer",
)

Side = Literal["front", "back", "left", "right"]
Purpose = Literal["access", "swing", "use"]


@dataclass(frozen=True)
class FrontWeights:
    """Weights of the per-side features in ``front.choose_front`` (see that module).

    low:    the side whose half of the top surface is lower (seat side, foot of a bed).
    recess: the side that is open or recessed when seen from outside (open shelves,
            knee space). Its opposite is a flat back panel.
    long:   the side is a long side of the footprint (negative prefers short sides).
    dims:   the resulting width/depth ratio matches the measured W/D.
    """

    low: float = 0.0
    recess: float = 0.0
    long: float = 0.3
    dims: float = 1.0


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    side: Side
    depth: float  # meters, measured outward from the footprint
    purpose: Purpose = "access"


@dataclass(frozen=True)
class CategoryProfile:
    key: str
    roomplan: str
    label: str
    front: FrontWeights
    front_rule: str
    zones: tuple[ZoneSpec, ...] = ()
    mount: Mount = Mount.floor
    movable: bool = True


SEATING = FrontWeights(low=1.0, recess=0.5, long=0.2, dims=1.0)
CHAIR = FrontWeights(low=1.0, recess=0.5, long=0.0, dims=1.0)
BED = FrontWeights(low=1.0, recess=0.3, long=-0.2, dims=1.0)
STORAGE = FrontWeights(recess=1.0, long=0.5, dims=1.0)
DESK = FrontWeights(recess=0.5, long=0.5, dims=1.0)
FLAT = FrontWeights(long=0.5, dims=1.0)
TABLE = FrontWeights(long=0.3, dims=1.0)

_SEAT_RULE = "seating: front = side where the top surface is lower (seat, not backrest)"
_BED_RULE = "bed: front = foot end, i.e. the side opposite the higher headboard"
_STORAGE_RULE = "storage: front = long side that is open/recessed (back panel is flat)"
_FLAT_RULE = "long side; 180-degree ambiguity is unresolved by geometry (low confidence)"
_TABLE_RULE = "table: front is arbitrary; long side chosen for a stable width/depth"
_DESK_RULE = "desk: front = long side with knee space (modesty panel is at the back)"


def _z(name: str, depth: float, purpose: Purpose = "access") -> ZoneSpec:
    """Zone in front of the asset."""
    return ZoneSpec(name, "front", depth, purpose)


LEGROOM = ZoneSpec("legroom", "front", 0.40, "use")
PULL_OUT = ZoneSpec("pull_out", "back", 0.45, "use")


def _profile(key, roomplan, label, front, rule, zones=(), mount=Mount.floor, movable=True):
    return CategoryProfile(key, roomplan, label, front, rule, tuple(zones), mount, movable)


_AROUND_075 = [ZoneSpec(f"chairs_{s}", s, 0.75, "use") for s in ("front", "back", "left", "right")]
_FRONT_090 = [ZoneSpec("door_access", "front", 0.90, "swing")]

PROFILES: dict[str, CategoryProfile] = {
    p.key: p
    for p in [
        # seating
        _profile("sofa", "sofa", "sofa", SEATING, _SEAT_RULE, [LEGROOM]),
        _profile("sectional", "sofa", "sectional sofa", SEATING, _SEAT_RULE, [LEGROOM]),
        _profile("loveseat", "sofa", "loveseat", SEATING, _SEAT_RULE, [LEGROOM]),
        # Generic RoomPlan "chair" gets lounge-chair semantics (legroom in front). Use
        # dining_chair / office_chair for chairs that tuck under a table or desk.
        _profile("chair", "chair", "chair", CHAIR, _SEAT_RULE, [LEGROOM]),
        _profile("armchair", "chair", "armchair", CHAIR, _SEAT_RULE, [LEGROOM]),
        _profile("dining_chair", "chair", "dining chair", CHAIR, _SEAT_RULE, [PULL_OUT]),
        _profile("office_chair", "chair", "office chair", CHAIR, _SEAT_RULE, [PULL_OUT]),
        _profile("stool", "chair", "stool", TABLE, _TABLE_RULE),
        # beds
        _profile(
            "bed",
            "bed",
            "bed",
            BED,
            _BED_RULE,
            [
                ZoneSpec("access_left", "left", 0.60),
                ZoneSpec("access_right", "right", 0.60),
                ZoneSpec("access_foot", "front", 0.60),
            ],
        ),
        # tables. Generic "table" has no zone: it is often a coffee table, and a zone
        # would push the sofa away. Pick dining_table / desk explicitly.
        _profile("table", "table", "table", TABLE, _TABLE_RULE),
        _profile("coffee_table", "table", "coffee table", TABLE, _TABLE_RULE),
        _profile("side_table", "table", "side table", TABLE, _TABLE_RULE),
        _profile("dining_table", "table", "dining table", TABLE, _TABLE_RULE, _AROUND_075),
        _profile("round_table", "table", "round table", TABLE, _TABLE_RULE, _AROUND_075),
        _profile(
            "desk",
            "table",
            "desk",
            DESK,
            _DESK_RULE,
            [ZoneSpec("chair_pull_out", "front", 0.90, "use")],
        ),
        # storage
        _profile("storage", "storage", "storage", STORAGE, _STORAGE_RULE, [_z("access", 0.6)]),
        _profile("bookshelf", "storage", "bookshelf", STORAGE, _STORAGE_RULE, [_z("access", 0.6)]),
        _profile("cabinet", "storage", "cabinet", STORAGE, _STORAGE_RULE, [_z("access", 0.6)]),
        _profile("sideboard", "storage", "sideboard", STORAGE, _STORAGE_RULE, [_z("access", 0.6)]),
        _profile("tv_stand", "storage", "TV stand", STORAGE, _STORAGE_RULE, [_z("access", 0.5)]),
        _profile("dresser", "storage", "dresser", STORAGE, _STORAGE_RULE, [_z("drawers", 0.9)]),
        _profile(
            "wardrobe",
            "storage",
            "wardrobe",
            STORAGE,
            _STORAGE_RULE,
            [ZoneSpec("doors", "front", 0.90, "swing")],
        ),
        _profile("nightstand", "storage", "nightstand", STORAGE, _STORAGE_RULE),
        # media
        _profile("television", "television", "television", FLAT, _FLAT_RULE),
        # fixtures and appliances (RoomPlan categories; rarely generated, not movable)
        _profile("refrigerator", "refrigerator", "refrigerator", FLAT, _FLAT_RULE, _FRONT_090),
        _profile("oven", "oven", "oven", FLAT, _FLAT_RULE, _FRONT_090, movable=False),
        _profile("stove", "stove", "stove", FLAT, _FLAT_RULE, _FRONT_090, movable=False),
        _profile(
            "dishwasher", "dishwasher", "dishwasher", FLAT, _FLAT_RULE, _FRONT_090, movable=False
        ),
        _profile("washerDryer", "washerDryer", "washer/dryer", FLAT, _FLAT_RULE, _FRONT_090),
        _profile("sink", "sink", "sink", FLAT, _FLAT_RULE, [_z("use", 0.75)], movable=False),
        _profile("toilet", "toilet", "toilet", FLAT, _FLAT_RULE, [_z("use", 0.6)], movable=False),
        _profile(
            "bathtub", "bathtub", "bathtub", FLAT, _FLAT_RULE, [_z("use", 0.75)], movable=False
        ),
        _profile(
            "fireplace",
            "fireplace",
            "fireplace",
            FLAT,
            _FLAT_RULE,
            [_z("hearth", 0.5)],
            movable=False,
        ),
        _profile(
            "stairs", "stairs", "stairs", FLAT, _FLAT_RULE, [_z("landing", 0.9)], movable=False
        ),
    ]
}

ALIASES: dict[str, str] = {
    "couch": "sofa",
    "settee": "loveseat",
    "sectional_sofa": "sectional",
    "l_sofa": "sectional",
    "l_shaped_sofa": "sectional",
    "corner_sofa": "sectional",
    "lounge_chair": "armchair",
    "accent_chair": "armchair",
    "recliner": "armchair",
    "kitchen_chair": "dining_chair",
    "desk_chair": "office_chair",
    "swivel_chair": "office_chair",
    "bar_stool": "stool",
    "dining": "dining_table",
    "kitchen_table": "dining_table",
    "end_table": "side_table",
    "bookcase": "bookshelf",
    "shelf": "bookshelf",
    "shelving": "bookshelf",
    "chest_of_drawers": "dresser",
    "closet": "wardrobe",
    "armoire": "wardrobe",
    "credenza": "sideboard",
    "buffet": "sideboard",
    "media_console": "tv_stand",
    "tv_unit": "tv_stand",
    "tv_console": "tv_stand",
    "bedside_table": "nightstand",
    "tv": "television",
    "fridge": "refrigerator",
    "washer": "washerDryer",
    "dryer": "washerDryer",
    "washer_dryer": "washerDryer",
    "washerdryer": "washerDryer",
}


def normalize_key(category: str) -> str:
    return category.strip().lower().replace("-", "_").replace(" ", "_")


def lookup(category: str) -> CategoryProfile | None:
    """Profile for a RoomPlan category, a fine type, or an alias (case-insensitive)."""
    key = normalize_key(category)
    key = ALIASES.get(key, key)
    if key == "washerdryer":
        key = "washerDryer"
    return PROFILES.get(key)


def profile_for(category: str) -> CategoryProfile:
    """Like ``lookup`` but falls back to a generic profile (no zones) for unknown names."""
    found = lookup(category)
    if found is not None:
        return found
    key = normalize_key(category)
    return CategoryProfile(
        key=key,
        roomplan=key,
        label=category.strip(),
        front=FrontWeights(),
        front_rule="unknown category: long side, low confidence",
    )
