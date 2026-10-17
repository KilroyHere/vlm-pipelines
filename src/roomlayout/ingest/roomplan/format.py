"""Parse Apple RoomPlan ``CapturedRoom`` JSON (the ``Codable`` encoding) into plain records.

What is verified, and how (details and links in docs/ingest.md):

- **Apple docs** (developer.apple.com/documentation/roomplan, read 2026-10-17):
  ``CapturedRoom`` is ``Codable``. It has ``walls``/``doors``/``windows``/``openings``/
  ``floors`` (``[Surface]``; ``floors`` iOS 17+), ``objects`` (``[Object]``), ``sections``
  (iOS 17+), ``story`` (iOS 17+), ``version``. ``Surface`` has ``identifier``,
  ``parentIdentifier: UUID?``, ``category`` (floor, door(isOpen:), opening, wall, window),
  ``confidence``, ``transform: simd_float4x4``, ``dimensions: simd_float3``,
  ``polygonCorners: [simd_float3]`` ("local plane coordinates", iOS 17+),
  ``curve: Curve?``, ``completedEdges``, ``story``. ``Object`` has the same identity and
  pose fields plus ``attributes``. Attribute cases: ChairType dining/stool/swivel/
  unidentified, TableType coffee/dining/unidentified, StorageType cabinet/shelf, SofaType
  lShaped/lShapedExtension/rectangular/singleSeat/unidentified.
- **Apple sample code**: scans are written and read with plain ``JSONEncoder``/
  ``JSONDecoder`` on ``CapturedRoom`` (WWDC23 sample "Providing custom models for captured
  rooms and structure exports"). Merged multi-room scans are a different type,
  ``CapturedStructure``, whose JSON has a ``rooms`` array of CapturedRoom objects.
- **Public device exports** (10 CapturedRoom and 2 CapturedStructure JSON files from
  public GitHub repos, 173 surfaces/objects; surveyed, not committed): key names as
  above; ``transform`` is a flat array of 16 floats in **column-major** order
  (translation at indices 12-14, last row 0 0 0 1); ``dimensions`` is a 3-array;
  ``category`` is ``{"wall": {}}`` / ``{"door": {"isOpen": false}}``; ``confidence`` is
  ``{"high": {}}``; ``identifier`` is an uppercase UUID string; ``attributes`` is
  ``{"ChairType": "swivel", ...}``; ``version`` was 2 and ``story`` 0 in every file.
  Extra top-level keys ``coreModel`` and ``referenceOriginTransform`` appear and are
  ignored. No top-level ``identifier`` was present even though the docs list one.

Unverified (handled defensively): nested 4x4 transform arrays, a string ``category``,
the ``curve`` encoding (never non-null in the samples), exports from iOS 16 (no floors,
polygonCorners, story or sections), and any ``version`` other than 2.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

SURFACE_KINDS = ("walls", "doors", "windows", "openings", "floors")


class IngestError(ValueError):
    """The capture can't be turned into a single-room LayoutProblem."""


@dataclass
class Element:
    """A Surface or Object, with its transform already decoded to a 4x4 matrix."""

    kind: str  # walls | doors | windows | openings | floors | objects
    identifier: str
    category: str
    category_payload: dict
    dimensions: np.ndarray  # (3,)
    transform: np.ndarray  # (4, 4), ARKit world from local
    parent: str | None = None
    polygon_corners: np.ndarray | None = None  # (N, 3) local
    curve: object | None = None
    story: int | None = None
    confidence: str | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3]

    @property
    def center(self) -> np.ndarray:
        return self.transform[:3, 3]


@dataclass
class Capture:
    elements: dict[str, list[Element]]
    version: object
    story: int | None
    sections: list[dict]
    skipped: list[str]  # human-readable reasons for elements we couldn't parse


def parse_transform(raw) -> np.ndarray:
    """4x4 matrix from RoomPlan's encoding.

    Verified: flat list of 16 numbers, column-major (simd column order). Accepted but
    unverified: a nested list of 4 columns of 4 numbers.
    """
    arr = np.asarray(raw, dtype=float)
    if arr.shape == (16,):
        m = arr.reshape(4, 4).T
    elif arr.shape == (4, 4):
        m = arr.T  # nested list of columns (unverified)
    else:
        raise ValueError(f"transform has shape {arr.shape}, expected 16 numbers")
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-3):
        raise ValueError("transform's last row isn't 0 0 0 1 (not column-major?)")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-3) or np.linalg.det(r) < 0:
        raise ValueError("transform rotation isn't orthonormal and right-handed")
    return m


def _enum_case(raw) -> tuple[str, dict]:
    """Swift enum Codable: ``{"door": {"isOpen": false}}`` -> ("door", {...})."""
    if isinstance(raw, dict) and len(raw) == 1:
        ((name, payload),) = raw.items()
        return str(name), payload if isinstance(payload, dict) else {}
    if isinstance(raw, str):  # unverified
        return raw, {}
    raise ValueError(f"unexpected enum encoding {raw!r}")


def _element(kind: str, raw: dict) -> Element:
    ident = raw.get("identifier")
    if not ident:
        # Deterministic fallback so ids stay stable across runs.
        ident = str(uuid.uuid5(uuid.NAMESPACE_OID, repr(sorted(raw.items(), key=str)))).upper()
    category, payload = _enum_case(raw["category"])
    dims = np.asarray(raw["dimensions"], dtype=float)
    if dims.shape != (3,):
        raise ValueError(f"dimensions has shape {dims.shape}, expected 3 numbers")
    corners = raw.get("polygonCorners") or None
    conf = raw.get("confidence")
    return Element(
        kind=kind,
        identifier=str(ident),
        category=category,
        category_payload=payload,
        dimensions=dims,
        transform=parse_transform(raw["transform"]),
        parent=raw.get("parentIdentifier"),
        polygon_corners=np.asarray(corners, dtype=float) if corners else None,
        curve=raw.get("curve"),
        story=raw.get("story"),
        confidence=_enum_case(conf)[0] if conf else None,
        attributes=raw.get("attributes") or {},
    )


def select_room(data: dict, room: int | None) -> tuple[dict, list[str]]:
    """Unwrap a CapturedStructure (``rooms`` array) to one CapturedRoom."""
    notes: list[str] = []
    if not isinstance(data, dict):
        raise IngestError("capture JSON must be an object (a CapturedRoom)")
    rooms = data.get("rooms")
    if rooms is not None:
        if room is None:
            if len(rooms) != 1:
                raise IngestError(
                    f"this is a merged CapturedStructure with {len(rooms)} rooms; v1 ingests "
                    "one room. Pick one with --room INDEX (0-based) or export a single "
                    "CapturedRoom."
                )
            room = 0
        if not 0 <= room < len(rooms):
            raise IngestError(f"--room {room} is out of range (structure has {len(rooms)} rooms)")
        notes.append(f"CapturedStructure with {len(rooms)} room(s): using rooms[{room}]")
        return rooms[room], notes
    if room not in (None, 0):
        raise IngestError("--room only applies to a CapturedStructure (a file with 'rooms')")
    if "walls" not in data and "objects" not in data:
        raise IngestError("not a RoomPlan CapturedRoom: no 'walls' or 'objects' arrays")
    return data, notes


def parse_capture(data: dict) -> Capture:
    elements: dict[str, list[Element]] = {}
    skipped: list[str] = []
    for kind in (*SURFACE_KINDS, "objects"):
        items = []
        for i, raw in enumerate(data.get(kind) or []):
            try:
                items.append(_element(kind, raw))
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append(f"{kind}[{i}] skipped: {exc}")
        elements[kind] = items
    stories = {e.story for items in elements.values() for e in items if e.story is not None}
    top = data.get("story")
    if top is not None:
        stories.add(top)
    if len(stories) > 1:
        raise IngestError(
            f"capture spans several stories {sorted(stories)}; v1 ingests a single room on "
            "one story"
        )
    return Capture(
        elements=elements,
        version=data.get("version"),
        story=next(iter(stories)) if stories else None,
        sections=list(data.get("sections") or []),
        skipped=skipped,
    )
