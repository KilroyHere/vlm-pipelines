"""Apple RoomPlan ``CapturedRoom`` JSON -> ``LayoutProblem`` skeleton (WS-E, M4a).

    from roomlayout.ingest.roomplan import load, convert
    result = load("Room.json")            # IngestResult
    result.problem                        # Room + box Assets + initial poses, no constraints
    result.warnings                       # things the user should check in the scan

CLI: ``python -m roomlayout.ingest.roomplan Room.json --out problem.json``.
The output has no clearance zones or constraints; ``roomlayout.pipeline`` adds the
category defaults and the rule library. Mapping and evidence: docs/ingest.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from .convert import IngestResult, convert
from .format import IngestError

__all__ = ["IngestError", "IngestResult", "convert", "load"]


def load(
    path: str | Path, *, room: int | None = None, problem_id: str | None = None
) -> IngestResult:
    """Read a RoomPlan JSON export from disk and convert it."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path} is not JSON: {exc}") from exc
    return convert(data, room=room, problem_id=problem_id or path.stem)
