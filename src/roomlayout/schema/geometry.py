"""Geometric primitives shared by every stage.

Frame conventions (ADR-0001): right-handed, meters, Z-up, floor at z = 0.
Yaw is a rotation about +Z in radians, counter-clockwise when viewed from above.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, field_validator

Vec2 = tuple[float, float]


class Polygon2D(BaseModel):
    """Simple polygon in the XY plane. Stored counter-clockwise, not closed
    (the first point is not repeated at the end)."""

    model_config = ConfigDict(frozen=True)

    points: list[Vec2]

    @field_validator("points")
    @classmethod
    def _valid_ccw(cls, pts: list[Vec2]) -> list[Vec2]:
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 distinct points")
        area = _signed_area(pts)
        if abs(area) < 1e-9:
            raise ValueError("polygon has zero area")
        return list(pts) if area > 0 else list(reversed(pts))

    @property
    def area(self) -> float:
        return _signed_area(self.points)

    @classmethod
    def rectangle(cls, width: float, depth: float) -> Polygon2D:
        """Axis-aligned rectangle centred on the origin (width along X, depth along Y)."""
        w, d = width / 2, depth / 2
        return cls(points=[(-w, -d), (w, -d), (w, d), (-w, d)])


class Pose2D(BaseModel):
    """Placement of an asset's local frame on the floor."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    yaw: float = 0.0

    def transform(self, p: Vec2) -> Vec2:
        """Map a point from asset-local coordinates to world coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1])

    @property
    def front(self) -> Vec2:
        """World-space unit vector of the asset's front (local +Y)."""
        return (-math.sin(self.yaw), math.cos(self.yaw))


def _signed_area(pts: list[Vec2]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True)
    )
