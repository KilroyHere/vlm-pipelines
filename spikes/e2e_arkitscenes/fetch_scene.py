#!/usr/bin/env python
"""Fetch the parts of an ARKitScenes capture the end-to-end test needs, without the
multi-GB archives.

    python fetch_scene.py 44358518 --out data            # annotation, poses, mesh, intrinsics
    python fetch_scene.py 44358518 --out data --frames 401.234 402.034   # + full-res frames

Per capture (raw/Validation/<video_id>/ on Apple's CDN) we download:

- ``<id>_3dod_annotation.json`` (~20 KB): oriented furniture boxes, world frame, meters, Z-up
- ``lowres_wide.traj`` (~200 KB): 10 Hz camera poses (timestamp, axis-angle, translation;
  world -> camera, see ARKitScenes threedod/benchmark_scripts/utils/tenFpsDataLoader.py)
- ``<id>_3dod_mesh.ply`` (~40 MB): reconstructed scene mesh, same world frame
- ``wide_intrinsics/*.pincam``: per-frame intrinsics of the 1920x1440 ``wide`` frames

``wide.zip`` (~450 MB, ~1 frame / 0.4 s) is never downloaded whole: its central directory
and the chosen frames are read with HTTP range requests.

ARKitScenes is Apple's dataset under a non-commercial license (see its LICENSE); keep
the data out of git (``data/`` is ignored) and use it for internal evaluation only.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import requests

CDN = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SESSION = requests.Session()


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = SESSION.head(url, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        r = SESSION.get(self.url, headers={"Range": rng}, timeout=300)
        r.raise_for_status()
        data = r.content
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20))


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with SESSION.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dst)


def base_url(video_id: str, fold: str = "Validation") -> str:
    return f"{CDN}/raw/{fold}/{video_id}"


def fetch_scene(video_id: str, out: Path, fold: str = "Validation") -> Path:
    scene = out / video_id
    scene.mkdir(parents=True, exist_ok=True)
    url = base_url(video_id, fold)
    for name in (
        f"{video_id}_3dod_annotation.json",
        "lowres_wide.traj",
        f"{video_id}_3dod_mesh.ply",
    ):
        _download(f"{url}/{name}", scene / name)

    index = scene / "wide_index.json"
    if not index.exists():
        intr = {}
        with remote_zip(f"{url}/wide_intrinsics.zip") as z:
            for n in z.namelist():
                if n.endswith(".pincam"):
                    ts = n.rsplit("_", 1)[1][: -len(".pincam")]
                    intr[ts] = [float(v) for v in z.read(n).split()]
        with remote_zip(f"{url}/wide.zip") as z:
            frames = sorted(
                n.rsplit("_", 1)[1][: -len(".png")] for n in z.namelist() if n.endswith(".png")
            )
        index.write_text(json.dumps({"frames": frames, "intrinsics": intr}, indent=1))
    return scene


def fetch_frames(
    video_id: str, scene: Path, stamps: list[str], fold: str = "Validation"
) -> list[Path]:
    """Download the given ``wide`` frames (1920x1440 PNG) into ``scene/wide/``."""
    want = [s for s in stamps if not (scene / "wide" / f"{s}.png").exists()]
    (scene / "wide").mkdir(exist_ok=True)
    if want:
        with remote_zip(f"{base_url(video_id, fold)}/wide.zip") as z:
            for s in want:
                (scene / "wide" / f"{s}.png").write_bytes(z.read(f"wide/{video_id}_{s}.png"))
    return [scene / "wide" / f"{s}.png" for s in stamps]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("video_ids", nargs="+")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--fold", default="Validation")
    p.add_argument("--frames", nargs="*", default=[], help="wide-frame timestamps to fetch")
    a = p.parse_args()
    _download(f"{CDN}/raw/metadata.csv", a.out / "metadata.csv")  # sky_direction per capture
    for vid in a.video_ids:
        scene = fetch_scene(vid, a.out, a.fold)
        if a.frames:
            fetch_frames(vid, scene, a.frames, a.fold)
        print(vid, "->", scene)


if __name__ == "__main__":
    main()
