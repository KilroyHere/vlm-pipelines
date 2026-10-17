"""Shared furniture vocabulary (contract, ADR-0002).

Coarse ``category`` follows Apple RoomPlan's ``CapturedRoom.Object.Category``; ``subtype``
is a finer type that maps onto exactly one category. ``ALIASES`` map everyday words (as a
user or VLM might write them) onto a subtype or category. Stage packages keep their own
per-type knowledge (zones, front heuristics, implied relations) keyed by these names.
"""

from __future__ import annotations

ROOMPLAN_CATEGORIES: tuple[str, ...] = (
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

# subtype -> RoomPlan category
SUBTYPES: dict[str, str] = {
    "sectional": "sofa",
    "loveseat": "sofa",
    "armchair": "chair",
    "dining_chair": "chair",
    "office_chair": "chair",
    "stool": "chair",
    "coffee_table": "table",
    "side_table": "table",
    "dining_table": "table",
    "round_table": "table",
    "desk": "table",
    "bookshelf": "storage",
    "cabinet": "storage",
    "sideboard": "storage",
    "tv_stand": "storage",
    "dresser": "storage",
    "wardrobe": "storage",
    "nightstand": "storage",
}

# everyday word -> subtype or category
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


def resolve(name: str) -> tuple[str, str | None] | None:
    """Map a category, subtype or alias to ``(category, subtype)``; None if unknown."""
    key = name.strip().replace(" ", "_").replace("-", "_")
    for candidate in (key, key.lower()):
        candidate = ALIASES.get(candidate, candidate)
        if candidate in SUBTYPES:
            return SUBTYPES[candidate], candidate
        if candidate in ROOMPLAN_CATEGORIES:
            return candidate, None
    return None
