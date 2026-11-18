#!/usr/bin/env bash
# Phase 2 image-to-3D comparison, the whole GPU part in one unattended job.
#
#   bash pod_job.sh BRANCH          (on a Runpod pod: template runpod-torch-v220, A40)
#
# Serves /workspace/img3d_out on port 8000 from the start (status.json, logs, results),
# so the laptop can follow progress and fetch results.tgz through the Runpod HTTP proxy.
# Steps: clone repo -> Miniforge -> build SAM 3D and TRELLIS envs in parallel (CPU, the
# slow part) while the test set is prepared -> bed_03 mask (rembg) -> SAM 3D (textured)
# -> TRELLIS -> results.tgz -> "done". The GPU is only busy in the two generation steps.
# Stopping the pod is the operator's job (the pod's own key can't stop it).
set -uo pipefail
BRANCH=${1:-main}
WS=/workspace
REPO=$WS/vlm-pipelines
OUT=$WS/img3d_out
DATA=$WS/abo
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
st() { printf '{"step": "%s", "time": "%s"}\n' "$1" "$(date -u +%FT%TZ)" > "$OUT/status.json"; echo "== $1 $(date -u +%T) =="; }

pgrep -f "http.server 8000" >/dev/null || (cd "$OUT" && nohup python3 -m http.server 8000 >/dev/null 2>&1 &)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"

st clone
[ -d "$REPO/.git" ] || git clone --depth 1 -b "$BRANCH" https://github.com/KilroyHere/vlm-pipelines "$REPO"
git -C "$REPO" log -1 --format='repo commit %h %s'

st miniforge
if [ ! -x "$WS/miniforge3/bin/mamba" ]; then
    curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/mf.sh -b -p "$WS/miniforge3"
fi

st build_envs
( bash "$REPO/spikes/sam3d/runpod_bootstrap.sh" > "$OUT/sam3d_setup.log" 2>&1; echo $? > "$OUT/sam3d_setup.rc" ) &
SAM_PID=$!
( bash "$REPO/spikes/img3d_eval/trellis_setup.sh" > "$OUT/trellis_setup.log" 2>&1; echo $? > "$OUT/trellis_setup.rc" ) &
TRE_PID=$!

st prepare_data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$WS/prepenv/bin/python" ] || { uv venv -q -p 3.12 "$WS/prepenv" && uv pip install -q --python "$WS/prepenv/bin/python" numpy pillow scipy; }
"$WS/prepenv/bin/python" "$REPO/spikes/img3d_eval/prepare_abo.py" --out "$DATA"

st wait_envs
wait $SAM_PID $TRE_PID
echo "sam3d setup rc=$(cat "$OUT/sam3d_setup.rc")  trellis setup rc=$(cat "$OUT/trellis_setup.rc")"

set +u; source "$WS/miniforge3/bin/activate"; set -u

st bed_03_mask
if [ ! -f "$DATA/eval/bed_03/mask.png" ]; then
    set +u; conda activate "$WS/envs/trellis"; set -u
    python - "$DATA/eval/bed_03" <<'PY'
import sys, numpy as np
from pathlib import Path
from PIL import Image
from rembg import new_session, remove
d = Path(sys.argv[1])
im = Image.open(d / "image.jpg").convert("RGB")
cut = remove(im, session=new_session("u2net"))  # same remover TRELLIS uses without alpha
alpha = np.asarray(cut)[:, :, 3] > 127
Image.fromarray((alpha * 255).astype(np.uint8)).save(d / "mask.png")
print(f"bed_03 mask: {alpha.mean():.1%} of the image")
PY
    conda deactivate
fi

st sam3d
if [ "$(cat "$OUT/sam3d_setup.rc")" = 0 ]; then
    set +u; conda activate "$WS/envs/sam3d-objects"; set -u
    python "$REPO/spikes/sam3d/run_sam3d.py" --data "$DATA/eval" --out "$OUT/sam3d" \
        --sam3d-repo "$WS/sam-3d-objects" --textured --skip-normalize 2>&1 | tee "$OUT/sam3d_run.log"
    conda deactivate
else
    echo "SAM 3D env failed: see sam3d_setup.log"
fi

st trellis
if [ "$(cat "$OUT/trellis_setup.rc")" = 0 ]; then
    set +u; conda activate "$WS/envs/trellis"; set -u
    python "$REPO/spikes/img3d_eval/run_trellis.py" --data "$DATA/eval" --out "$OUT/trellis" \
        --trellis-repo "$WS/TRELLIS" 2>&1 | tee "$OUT/trellis_run.log"
    conda deactivate
else
    echo "TRELLIS env failed: see trellis_setup.log"
fi

st package
cp "$DATA/eval/bed_03/mask.png" "$OUT/bed_03_mask.png" 2>/dev/null || true
(cd "$OUT" && tar czf results.tgz --exclude=results.tgz sam3d trellis bed_03_mask.png gpu.txt ./*.log 2>/dev/null; ls -la results.tgz)
st done
