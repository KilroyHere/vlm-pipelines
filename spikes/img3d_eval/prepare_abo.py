"""Build the image-to-3D test set from ABO (Amazon Berkeley Objects, CC BY-NC 4.0).

    python spikes/img3d_eval/prepare_abo.py --out data/abo [--gt]

For each item in ``abo_selection.json``: download the listing's main photo (and with
``--gt`` the ground-truth 3D model), make ``<out>/eval/<id>/`` with ``image.jpg`` +
``meta.json`` via ``spikes/sam3d/make_mask.py prepare`` (dims = the model's true
extent, as "manufacturer" dims), then white-background masks (``white_mask.py``).
``bed_03`` is photographed in a room: its mask comes from rembg on the pod
(``pod_job.sh``). Deterministic, so the pod and the laptop build identical inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
# our asset category per item (roomlayout.assets profiles)
CATEGORY = {
    "sectional_01": "sectional", "sectional_02": "sectional", "sofa_01": "sofa",
    "sofa_02": "sectional", "chair_01": "armchair", "chair_02": "dining_chair",
    "chair_03": "armchair", "chair_04": "office_chair", "table_01": "console_table",
    "table_02": "tv_stand", "table_03": "coffee_table", "table_04": "nightstand",
    "desk_01": "desk", "desk_02": "desk", "bed_01": "bed", "bed_02": "bed", "bed_03": "bed",
    "bookshelf_01": "bookshelf", "bookshelf_02": "bookshelf", "dresser_01": "dresser",
    "cabinet_01": "cabinet", "ottoman_01": "ottoman", "bench_01": "bench",
}  # fmt: skip
IN_ROOM = ["bed_03"]
WHITE_ON_WHITE = ["desk_01", "table_04", "bookshelf_02"]  # tighter background tolerance
LARGEST_ONLY = ["chair_02"]  # two chairs in the photo: mask one


def fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        subprocess.run(["curl", "-sS", "-f", "-o", str(dest), url], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "abo")
    ap.add_argument("--gt", action="store_true", help="also download the ground-truth GLBs")
    args = ap.parse_args()
    sel = json.loads((HERE / "abo_selection.json").read_text())
    py = sys.executable
    for s in sel:
        raw = args.out / "items" / s["id"]
        raw.mkdir(parents=True, exist_ok=True)
        fetch(f"{ABO}/images/original/{s['image_path']}", raw / "photo.jpg")
        if args.gt:
            fetch(f"{ABO}/3dmodels/original/{s['model_path']}", raw / "gt.glb")
        w, h, d = s["extent_xyz"]  # glTF Y-up: x = width, y = height, z = depth
        item = args.out / "eval" / s["id"]
        if not (item / "meta.json").exists():
            cond = ["in_room"] if s["id"] in IN_ROOM else ["product"]
            subprocess.run(
                [py, str(ROOT / "spikes/sam3d/make_mask.py"), "prepare", str(raw / "photo.jpg"),
                 str(item), "--category", CATEGORY[s["id"]], "--dims", f"{w:.4f}", f"{d:.4f}",
                 f"{h:.4f}", "--dims-provenance", "manufacturer", "--source-url",
                 f"abo:{s['model_id']}", "--condition", *cond],
                check=True,
            )  # fmt: skip
    ev = args.out / "eval"
    plain = [str(ev / s["id"]) for s in sel if s["id"] not in IN_ROOM + WHITE_ON_WHITE]
    mask = [py, str(HERE / "white_mask.py")]
    subprocess.run([*mask, *plain, "--largest", *LARGEST_ONLY], check=True)
    subprocess.run([*mask, *(str(ev / i) for i in WHITE_ON_WHITE), "--tol", "5"], check=True)
    print(f"{len(sel)} items in {ev} (bed_03 still needs its rembg mask)")


if __name__ == "__main__":
    main()
