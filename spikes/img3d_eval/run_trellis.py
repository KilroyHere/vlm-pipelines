"""TRELLIS (microsoft/TRELLIS-image-large, MIT) on the same items and masks as SAM 3D.

Run inside the ``trellis`` env built by ``trellis_setup.sh`` (GPU), never imported by
roomlayout (ADR-0004):

    python spikes/img3d_eval/run_trellis.py --data data/abo/eval --out OUT --trellis-repo DIR

Per item: the photo with our mask as alpha (TRELLIS crops to the alpha box and skips its
own background removal when alpha is present), ``pipeline.run(seed=42)``, then
``postprocessing_utils.to_glb(simplify=0.95, texture_size=1024)``, the README's example
settings. Writes ``OUT/<id>/trellis.glb`` (textured, glTF Y-up) and ``OUT/summary.json``
(seconds, peak VRAM, faces, errors). One failing item never stops the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trellis-repo", type=Path, required=True)
    ap.add_argument("--model", default="microsoft/TRELLIS-image-large")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--items", help="comma-separated ids (default: all)")
    args = ap.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "xformers")  # no flash-attn build needed
    os.environ.setdefault("SPCONV_ALGO", "native")  # README: recommended for single runs
    sys.path.insert(0, str(args.trellis_repo.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    t0 = time.perf_counter()
    pipe = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()
    print(f"[trellis] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    wanted = set(args.items.split(",")) if args.items else None
    items = sorted(d for d in args.data.iterdir() if (d / "mask.png").exists())
    summary: dict[str, dict] = {}
    for d in items:
        if wanted and d.name not in wanted:
            continue
        out = args.out / d.name
        out.mkdir(parents=True, exist_ok=True)
        rec: dict = {"status": "pending"}
        try:
            rgb = Image.open(next(d.glob("image.*"))).convert("RGB")
            mask = Image.open(d / "mask.png").convert("L")
            alpha = np.where(np.asarray(mask) > 127, 255, 0).astype(np.uint8)
            rgba = rgb.copy()
            rgba.putalpha(Image.fromarray(alpha))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = pipe.run(rgba, seed=args.seed, formats=["gaussian", "mesh"])
            glb = postprocessing_utils.to_glb(
                res["gaussian"][0], res["mesh"][0], simplify=0.95, texture_size=1024
            )
            torch.cuda.synchronize()
            rec["generation_s"] = round(time.perf_counter() - t, 2)
            rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            glb.export(str(out / "trellis.glb"))
            rec["faces"] = int(len(glb.faces)) if hasattr(glb, "faces") else None
            rec["status"] = "generated"
            print(f"[trellis] {d.name}: {rec['generation_s']}s", flush=True)
            del res, glb
        except Exception as exc:  # one bad item must not stop the batch
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            (out / "generation_error.txt").write_text(traceback.format_exc())
            print(f"[trellis] {d.name}: FAILED {rec['error']}", flush=True)
            torch.cuda.empty_cache()
        summary[d.name] = rec
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    ok = sum(r["status"] == "generated" for r in summary.values())
    print(f"[trellis] done: {ok}/{len(summary)} generated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
