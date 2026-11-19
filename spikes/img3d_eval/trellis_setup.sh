#!/usr/bin/env bash
# TRELLIS env on a Runpod pod (template runpod-torch-v220: CUDA 12.1 devel toolkit).
# Idempotent. Uses the Miniforge that runpod_bootstrap.sh installs under /workspace.
#   bash spikes/img3d_eval/trellis_setup.sh
set -euo pipefail
set --   # conda's activate script reads "$@"
WS=/workspace
CONDA_DIR=$WS/miniforge3
ENV_DIR=$WS/envs/trellis
TRELLIS_DIR=$WS/TRELLIS
LOG=$WS/trellis_setup.log
exec > >(tee -a "$LOG") 2>&1
set +u; source "$CONDA_DIR/bin/activate"; set -u

if [ ! -d "$TRELLIS_DIR/.git" ]; then
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$TRELLIS_DIR"
fi
echo "TRELLIS commit: $(git -C "$TRELLIS_DIR" rev-parse HEAD)"

if [ ! -f "$ENV_DIR/.ready" ]; then
    [ -d "$ENV_DIR" ] || mamba create -y -p "$ENV_DIR" python=3.10
    set +u; conda activate "$ENV_DIR"; set -u
    # torch 2.4.0 + cu121 matches the image's CUDA 12.1 toolkit (README: tested 11.8/12.2)
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    cd "$TRELLIS_DIR"
    # CUDA extensions (nvdiffrast, diff-gaussian-rasterization, ...) import torch in their
    # setup.py: pip's build isolation hides it ("Cannot compile nvdiffrast", 2026-11-19).
    # The image's pip config also names pypi.ngc.nvidia.com, which no longer resolves.
    export PIP_NO_BUILD_ISOLATION=1
    export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
    pip install -q setuptools wheel ninja
    # xformers attention (no flash-attn compile); extensions needed for to_glb
    set +u
    . ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    set -u
    python -c "import torch, xformers, spconv, nvdiffrast, kaolin; print('trellis env ok', torch.__version__)"
    touch "$ENV_DIR/.ready"
fi
echo "trellis env ready: $ENV_DIR"
