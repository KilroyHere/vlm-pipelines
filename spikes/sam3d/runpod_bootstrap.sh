#!/usr/bin/env bash
# One-command SAM 3D setup on a Runpod pod (template runpod-torch-v220, CUDA 12.1 devel).
#
#   bash runpod_bootstrap.sh            # build env + download checkpoints (idempotent)
#   bash runpod_bootstrap.sh --smoke    # ...then run Meta's demo.py as a smoke test
#
# Everything lives under /workspace (the pod's volume disk), so a stop/start keeps it;
# terminating the pod deletes it. Needs HF_TOKEN in the environment for the gated
# checkpoints: set the pod env var HF_TOKEN={{ RUNPOD_SECRET_hf_token }} after creating
# a Runpod secret named hf_token in the console. The token is never printed or logged.
set -euo pipefail

SMOKE=${1:-}
set --   # conda's activate script reads "$@": don't let it see our flags

WS=/workspace
SAM3D_COMMIT=f91db41            # the commit spikes/sam3d/README.md documents
CONDA_DIR=$WS/miniforge3
ENV_DIR=$WS/envs/sam3d-objects
SAM3D_DIR=$WS/sam-3d-objects
REPO_DIR=$WS/vlm-pipelines
LOG=$WS/sam3d_bootstrap.log
exec > >(tee -a "$LOG") 2>&1
step() { printf '\n== %s ==\n' "$*"; }

step "GPU check"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$cap" -ge 10 ]; then
    echo "Blackwell-class GPU (compute capability $cap): SAM 3D pins torch 2.5.1+cu121," \
         "which has no kernels for it. Recreate the pod on an A40/L40S/A100." >&2
    exit 1
fi
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 32000 ] || { echo "Need >= 32 GB VRAM, found ${mem} MiB" >&2; exit 1; }

step "Miniforge"
if [ ! -x "$CONDA_DIR/bin/mamba" ]; then
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
set +u; source "$CONDA_DIR/bin/activate"; set -u   # conda hooks read unset vars

step "SAM 3D Objects source @ $SAM3D_COMMIT"
if [ ! -d "$SAM3D_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_DIR"
fi
git -C "$SAM3D_DIR" checkout -q -f "$SAM3D_COMMIT"
# Their pyproject.toml also names pypi.ngc.nvidia.com, which no longer resolves (2026-10-17).
sed -i 's#https://pypi.ngc.nvidia.com#https://pypi.nvidia.com#g' "$SAM3D_DIR/pyproject.toml"

step "Conda env (doc/setup.md; slow: pytorch3d compiles from source)"
cd "$SAM3D_DIR"
if [ ! -f "$ENV_DIR/.roomlayout_ready" ]; then
    [ -d "$ENV_DIR" ] || mamba env create -f environments/default.yml -p "$ENV_DIR"
    set +u; conda activate "$ENV_DIR"; set -u
    # pypi.ngc.nvidia.com (in SAM 3D's doc/setup.md) no longer resolves (2026-10-17); use NVIDIA's current index
    export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e '.[dev]'
    pip install -e '.[p3d]'
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e '.[inference]'
    ./patching/hydra
    touch "$ENV_DIR/.roomlayout_ready"
else
    set +u; conda activate "$ENV_DIR"; set -u
fi
python -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"

step "Checkpoints (facebook/sam-3d-objects, gated)"
if [ -d "$SAM3D_DIR/checkpoints/hf" ]; then
    echo "already present"
elif [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not set: skipping. Set it (see header) and re-run this script." >&2
else
    pip install -q 'huggingface-hub[cli]<1.0'
    hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 \
        facebook/sam-3d-objects   # reads HF_TOKEN from the environment
    mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
fi

step "roomlayout (CPU side, its own venv; ADR-0004)"
if [ -f "$REPO_DIR/pyproject.toml" ]; then
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    (cd "$REPO_DIR" && uv venv -q && uv pip install -q -e ".[assets]")
    echo "roomlayout ready: $REPO_DIR/.venv/bin/python"
else
    echo "$REPO_DIR not found: clone or upload the repo there, then re-run." >&2
fi

if [ "$SMOKE" = "--smoke" ] && [ -d "$SAM3D_DIR/checkpoints/hf" ]; then
    step "Smoke test (Meta's demo.py)"
    python demo.py
fi

step "Done"
cat <<EOF
Next (in this terminal):
  source $CONDA_DIR/bin/activate && conda activate $ENV_DIR
  cd $REPO_DIR/spikes/sam3d
  python run_sam3d.py --data data --out outputs --sam3d-repo $SAM3D_DIR \\
      --roomlayout-python $REPO_DIR/.venv/bin/python --save-splat 2>&1 | tee outputs_log.txt
Log of this script: $LOG
EOF
