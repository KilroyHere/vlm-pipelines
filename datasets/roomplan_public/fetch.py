"""Download the public RoomPlan exports listed in manifest.json and verify them.

    python datasets/roomplan_public/fetch.py [--out data/roomplan_public]

Each file comes from raw.githubusercontent.com at the repo's HEAD and must match the
git blob SHA-1 recorded in the manifest (the file as surveyed on the retrieval date).
A mismatch means the upstream file changed; it is reported and not saved. The scans are
not committed: most source repos carry no license, so they are used for research only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/roomplan_public"))
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    args = ap.parse_args(argv)
    rooms = json.loads(args.manifest.read_text())["rooms"]
    args.out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for r in rooms:
        dest = args.out / f"{r['id']}.json"
        if dest.exists() and git_blob_sha1(dest.read_bytes()) == r["git_blob_sha1"]:
            continue
        url = "https://raw.githubusercontent.com/{}/HEAD/{}".format(
            r["repo"], urllib.parse.quote(r["path"])
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except OSError as e:
            print(f"FAIL {r['id']}: {e}", file=sys.stderr)
            bad += 1
            continue
        if git_blob_sha1(data) != r["git_blob_sha1"]:
            print(
                f"CHANGED {r['id']}: upstream file differs from the surveyed one", file=sys.stderr
            )
            bad += 1
            continue
        dest.write_bytes(data)
    print(f"{len(rooms) - bad}/{len(rooms)} scans in {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
