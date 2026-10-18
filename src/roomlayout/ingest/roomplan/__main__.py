"""``python -m roomlayout.ingest.roomplan CAPTURE.json --out problem.json``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import IngestError, load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RoomPlan CapturedRoom JSON -> LayoutProblem")
    p.add_argument("capture", type=Path)
    p.add_argument("--out", type=Path, required=True, help="problem JSON to write")
    p.add_argument("--room", type=int, help="room index in a merged CapturedStructure")
    p.add_argument("--id", help="problem id (default: the file name)")
    p.add_argument("--report", type=Path, help="also write notes/warnings/id mapping JSON")
    a = p.parse_args(argv)
    try:
        res = load(a.capture, room=a.room, problem_id=a.id)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(res.problem.model_dump_json(indent=1))
    if a.report:
        a.report.write_text(json.dumps(res.report(), indent=1))
    r = res.problem.room
    print(f"{a.out}: {len(r.walls)} walls, {len(r.openings)} openings, "
          f"{len(r.obstacles)} obstacles, {len(res.problem.assets)} assets")  # fmt: skip
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
