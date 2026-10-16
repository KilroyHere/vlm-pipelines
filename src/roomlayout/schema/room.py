"""The room: floor, walls, openings and fixed obstacles, in world coordinates."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from .geometry import Polygon2D, Vec2


class Wall(BaseModel):
    """A straight wall segment. The room interior is on the left when walking
    from ``start`` to ``end`` (i.e. walls run counter-clockwise around the floor)."""

    id: str
    start: Vec2
    end: Vec2
    height: PositiveFloat
    thickness: PositiveFloat = 0.1

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


class OpeningType(StrEnum):
    door = "door"
    window = "window"
    opening = "opening"  # doorless passage


class DoorSwing(BaseModel):
    """Door leaf swing. The swept quarter-disc (radius = opening width) must stay clear."""

    into_room: bool = True
    hinge: Literal["start", "end"] = "start"  # which jamb, relative to the wall direction


class Opening(BaseModel):
    id: str
    type: OpeningType
    wall_id: str
    offset: float = Field(
        description="Distance along the wall from its start to the opening centre"
    )
    width: PositiveFloat
    height: PositiveFloat
    sill_height: float = Field(0.0, ge=0.0, description="Height of the opening's bottom edge")
    swing: DoorSwing | None = None


class Obstacle(BaseModel):
    """Immovable thing the layout must avoid (column, radiator, built-in)."""

    id: str
    label: str
    footprint: Polygon2D  # world frame
    height: PositiveFloat


class Room(BaseModel):
    id: str
    floor: Polygon2D
    ceiling_height: PositiveFloat
    walls: list[Wall]
    openings: list[Opening] = []
    obstacles: list[Obstacle] = []

    @model_validator(mode="after")
    def _check_refs(self) -> Room:
        walls = {w.id: w for w in self.walls}
        if len(walls) != len(self.walls):
            raise ValueError("duplicate wall ids")
        for o in self.openings:
            wall = walls.get(o.wall_id)
            if wall is None:
                raise ValueError(f"opening {o.id!r} references unknown wall {o.wall_id!r}")
            if o.offset - o.width / 2 < -1e-6 or o.offset + o.width / 2 > wall.length + 1e-6:
                raise ValueError(f"opening {o.id!r} does not fit on wall {o.wall_id!r}")
        return self
