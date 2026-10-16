"""Data contract shared by every pipeline stage. See docs/contracts.md.

Changes here affect every workstream: they go through the tech lead and bump
SCHEMA_VERSION, and `python -m roomlayout.schema.export` must be re-run.
"""

from .asset import Asset, AssetSource, ClearanceZone, Dimensions, MeshRef, Mount
from .constraints import (
    AgainstWall,
    Constraint,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    Near,
    Parallel,
    WalkwayWidth,
    referenced_assets,
)
from .geometry import Polygon2D, Pose2D, Vec2
from .layout import (
    SCHEMA_VERSION,
    ConstraintResult,
    Layout,
    LayoutProblem,
    Placement,
    SolveResult,
    SolveStats,
)
from .room import DoorSwing, Obstacle, Opening, OpeningType, Room, Wall

__all__ = [
    "SCHEMA_VERSION",
    "AgainstWall",
    "Asset",
    "AssetSource",
    "ClearanceZone",
    "Constraint",
    "ConstraintResult",
    "Dimensions",
    "DistanceAtLeast",
    "DoorSwing",
    "Faces",
    "Fixed",
    "KeepClear",
    "Layout",
    "LayoutProblem",
    "MeshRef",
    "Mount",
    "Near",
    "Obstacle",
    "Opening",
    "OpeningType",
    "Parallel",
    "Placement",
    "Polygon2D",
    "Pose2D",
    "Room",
    "SolveResult",
    "SolveStats",
    "Vec2",
    "Wall",
    "WalkwayWidth",
    "referenced_assets",
]
