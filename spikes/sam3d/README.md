# SAM 3D Objects spike (M2b)

Goal: run Meta's SAM 3D Objects on ~20 of *our* furniture photos on a rented GPU, push
every mesh through `roomlayout.assets`, and score the results against
[EVAL.md](EVAL.md) to make the **go/no-go** call. Everything except generation runs on
CPU and is tested (`tests/assets/test_asset_spike_harness.py`). **The generation step
(`run_sam3d.py` without `--skip-generation`) has never been run: there was no GPU.**

| file | what |
|---|---|
| `run_sam3d.py` | batch runner: validate inputs, run SAM 3D (GPU), normalize (subprocess), summaries |
| `make_mask.py` | prepare item folders (upright image + meta.json), draw/derive masks, QA overlays |
| `score.py` | footprint IoU vs. true footprints, merge manual ratings, go/no-go table |
| `EVAL.md` | test set spec, rubric, thresholds, results template |
| `photo_sources.md` | where each case's photo comes from (web listings, own photos) + data-handling note |

`data/` and `outputs/` are gitignored. Never commit photos or meshes.

## 1. What we verified (2026-10-16)

Sources: the repo at commit `f91db41` (2026-06-02), its `doc/setup.md`, `LICENSE`, `demo.py`,
`notebook/inference.py` and `sam3d_objects/` source; the Hugging Face model page; the
paper (arXiv 2511.16624v2).

| topic | fact | source |
|---|---|---|
| OS | "A linux 64-bits architecture (i.e. `linux-64` platform in `mamba info`)". ARM hosts such as GH200 don't qualify. | [doc/setup.md](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) |
| GPU | "A NVIDIA GPU with at least 32 Gb of VRAM." Build steps may need a GPU node ("`RuntimeError: Not compiled with GPU support`" otherwise). | doc/setup.md |
| CUDA / Python / torch | conda env pins `cuda-version=12.1`, `python=3.11.0`; pip pins `torch*==2.5.1+cu121`, `kaolin==0.17.0` (torch-2.5.1_cu121 wheels), `spconv-cu121`, `xformers 0.0.28.post3`, `flash_attn 2.8.3`, pytorch3d and gsplat from pinned git commits; hydra 1.3.2 must be patched (`./patching/hydra`). | `environments/default.yml`, `requirements*.txt`, setup.md |
| Checkpoints | Hugging Face `facebook/sam-3d-objects`, **gated with manual approval** (`"gated":"manual"` on the model page). Request access with full legal name, date of birth and full organization name. Not available in comprehensively sanctioned jurisdictions. Then `hf auth login` and `hf download`. | [HF page](https://huggingface.co/facebook/sam-3d-objects), setup.md |
| License | "SAM License" (last updated Nov 19, 2025) covers code **and** weights. It grants a "non-exclusive, worldwide, non-transferable and royalty-free limited license ... to use, reproduce, distribute, copy, create derivative works of, and make modifications". The text contains **no non-commercial or research-only clause** (the word "commercial" does not appear). Conditions: ship the license with any redistribution, acknowledge SAM in publications, trade-control/ITAR and military-use bans, no reverse engineering, termination on patent/IP litigation against Meta, and Meta may amend the terms. You own derivative works you make (5a). | [LICENSE](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) |
| Input | RGB image + binary object mask. The wrapper merges the mask into the alpha channel. A depth/pointmap model (MoGe) runs internally unless a pointmap is passed. | `notebook/inference.py`, `pipeline/inference_pipeline_pointmap.py` |
| Entry point | `sys.path.append("notebook"); from inference import Inference; inference = Inference("checkpoints/hf/pipeline.yaml", compile=False); output = inference(image, mask, seed=42)` | README, `demo.py` |
| Output | dict with `glb`: a `trimesh.Trimesh` (with the wrapper's defaults `with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True`, a raw mesh with **vertex colours**), `gs`: a Gaussian splat (`save_ply`), and `rotation` (quaternion), `translation`, `scale`: the object's pose in the camera frame, plus `pointmap`. The demo only saves the splat; we `output["glb"].export(...)`. | `inference.py`, `inference_pipeline.py::postprocess_slat_output` |
| **Up axis** | `to_glb` ends with `vertices @ [[1,0,0],[0,0,-1],[0,1,0]]` commented "rotate mesh (from z-up to y-up)", i.e. (x, y, z) -> (x, z, -y). **The returned mesh is Y-up (glTF convention)**, so our CLI's default is `--up y`. SAM 3D's own layout code undoes exactly this rotation before applying the pose (`layout_post_optimization_utils.get_mesh`), which confirms it. | `tdfy_dit/utils/postprocessing_utils.py:665` |
| Scale | Shape is generated in a normalized canonical space (64^3 voxel grid, `coords / 64 - 0.5`, i.e. a unit cube). Size only comes from the predicted `scale` relative to a monocular pointmap, which is **not metric enough** for layouts. We ignore it (ADR-0004) and fit to measured dims. | `inference_pipeline_pointmap.py`, paper sec. 2.1 |
| Known limits | Coarse geometry is 64^3: "thin structures ... can lead to noticeable distortions or loss of details". Objects are predicted one at a time with no physical reasoning. Textures of rotationally symmetric objects can come out rotated. | paper, Appendix F |

### Not verified (measure or ask)
- **Runtime per object and peak VRAM** on our images. The paper mentions distillation for
  "sub-second shape and layout from the Geometry model", but the public `Inference` wrapper
  runs without distillation and includes texture/decoding. `run_sam3d.py` measures both.
- **Whether the canonical shape is upright** (gravity-aligned) for real photos. Likely, since
  pretraining uses artist assets, but there's no statement. Our pipeline levels tilts up to 8°
  and `up_check` flags a wrong up axis.
- **Yaw of the canonical shape** relative to the object's front: no evidence. We treat it as
  arbitrary (front heuristic + renders).
- **Commercial use**: our reading of the license text above is that it's allowed, but that's
  not a legal opinion. **Product owner / counsel must confirm** (risk register). Also unknown:
  whether Meta's online demo terms differ (don't use the demo for product data).
- Whether 48 GB cards (L40S / RTX 6000 Ada / A6000 / A40) work. They exceed the stated 32 GB,
  but Meta doesn't name tested GPUs.
- **Do not use Blackwell GPUs** (RTX 5090, RTX PRO 6000 Blackwell, B200). SAM 3D pins
  torch 2.5.1+cu121, which has no sm_120/sm_100 kernels; Blackwell needs a CUDA 12.8+ build
  ([PyTorch forum](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090/218954)).
  Ampere (A6000, A40, A100), Ada (L40S, 6000 Ada) and Hopper (H100) are fine.

## 2. GPU and cost

Prices change weekly; the Runpod plan below is from live reads. Market snapshot (Sept–Oct 2026):

| GPU (VRAM) | Vast.ai (marketplace) | RunPod Community | RunPod Secure | Lambda | fit |
|---|---|---|---|---|---|
| RTX A6000 (48 GB) | from ~$0.28/h | ~$0.49/h | ~$0.76/h | $1.09/h | out of stock on Runpod 2026-10-17 |
| A40 (48 GB) | varies | no stock | **$0.59/h (live)** | – | **chosen** |
| L40S (48 GB) | from ~$0.47/h | ~$0.86–0.89/h | ~$1.09–1.19/h | – | good |
| A100 (80 GB) | varies | | $1.79/h | | fallback if 48 GB OOMs |
| RTX 5090 / Blackwell | | | | | **incompatible** (see above) |

Sources: [Hostinger comparison, Sept 2026](https://www.hostinger.com/tutorials/?p=158956),
[RunPod pricing](https://www.runpod.io/gpu-cloud/pricing),
[RunPod price changes, Sept 2026](https://pricingsaas.com/companies/runpod/diffs/2026W37).

### Runpod plan (live catalog reads via the Runpod connector, 2026-10-17)

| Choice | Value | Why (read) |
|---|---|---|
| GPU | **NVIDIA A40, 48 GB, Secure Cloud, $0.59/h** | cheapest ≥ 32 GB card in stock (`list-gpu-types`): CA-MTL-1 medium, EU-SE-1 low. Its $0.35 Community price had no Community stock. RTX A6000: out of stock. |
| Data center | CA-MTL-1 | highest A40 stock |
| Template / image | `runpod-torch-v220` = `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` | official template (`list-public-templates`): CUDA 12.1 toolkit for the pytorch3d build, allowed host CUDA 12.1–12.9, Jupyter on 8888 for a browser terminal + uploads |
| CUDA floor | `gpu.minCudaVersion = "12.1"` | the image's CUDA line |
| Storage | pod volume disk 100 GB at `/workspace` (+ 50 GB container disk) | survives stop/start, deleted on terminate. **No network volume:** Runpod network volumes attach to Secure Cloud pods only ([docs](https://docs.runpod.io/storage/network-volumes)), and neither A40 data center offers them. A network volume would mean RTX 6000 Ada ($0.99/h) or L40S ($1.09/h) in US-IL-1. Not worth it for a 3–4 h run. |
| Ports | `8888/http` only | Jupyter terminal + file upload; no SSH needed |
| Env | `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` | create a Runpod secret `hf_token` in the console first (never paste the token into chat or code) |

Fallbacks if A40 stock is gone: L40S (US-IL-1, $1.09/h Secure / $0.79 Community),
RTX 6000 Ada ($0.99/h Secure), A100 80 GB PCIe ($1.79/h Secure). **Never** a Blackwell card
(RTX 5090, RTX PRO 6000/4500, B200/B300); the bootstrap script refuses them.

Setup on the pod is one command: `bash runpod_bootstrap.sh --smoke` (this folder). It builds
the env, downloads checkpoints, sets up the roomlayout venv and runs Meta's demo, all under
`/workspace`, idempotently.

**Cheapest alternative provider:** Vast.ai A6000 (verified hosts, reliability ≥ 99 %,
≥ 100 GB disk; destroy when done, stopped instances still bill disk).

Budget: env build ~45–60 min (pytorch3d compile), checkpoint download, then ~20 items.
About **3–4 GPU-hours ≈ $2–2.50 on the A40**, plus volume disk while stopped. Prepare photos
and masks on your laptop first (section 3) so no GPU time is spent on prep. **Terminate the
pod when done** (after copying `outputs/` home).

## 3. Before renting the GPU (laptop, CPU)

1. **Request checkpoint access** on Hugging Face (manual approval can take time). Use an
   HF account in the company's name, with the full legal org name as required.
   *(Status 2026-10-16: requested by the product owner, awaiting approval.)*
2. **Collect the test set** per [EVAL.md](EVAL.md): photos (web listings per
   [photo_sources.md](photo_sources.md), or the owner's own), W/D/H with their provenance,
   condition tags. Follow the data-handling note in photo_sources.md: images go into the
   gitignored `data/` only.
3. **Prepare items and masks** with `make_mask.py` (needs numpy, Pillow, matplotlib):

   ```bash
   cd spikes/sam3d
   python make_mask.py prepare ~/photos/IMG_0412.jpg data/sofa_02 --category sofa \
       --dims 2.10 0.92 0.83 --dims-provenance tape_measured --condition in_room occluded \
       --notes "pillows, side table"
   python make_mask.py prepare kivik.jpg data/sofa_01 --category sofa --condition product \
       --dims W D H --dims-provenance manufacturer \
       --source-url https://www.ikea.com/us/en/p/kivik-sofa-gunnared-light-green-s19499706/
   python make_mask.py polygon data/sofa_01      # click the outline, Enter, Enter
   python make_mask.py check data                # look at data/*/mask_check.png
   ```

   - `--dims-provenance` (required with `--dims`): `tape_measured`, `manufacturer`,
     `seller_reported` or `estimated`. It's informational: `manufacturer`/`seller_reported`
     become the contract's `dims_from = listing`, the others `manual`, and the finer value
     shows per item in the summary and results. `--source-url` strips `utm_*` tracking
     parameters. Without known dims leave out `--dims` (the asset is marked `mesh`).
   - Product photos on white backgrounds: `make_mask.py sam2 --box` (a box around the
     product is usually enough) or `polygon`.
   - Export iPhone HEIC photos as JPEG first (Pillow can't read HEIC without a plugin).
   - `prepare` applies the EXIF rotation, so the image is upright, caps it at 2048 px and
     writes `meta.json`. A mask drawn on a sideways image is the most likely silent failure,
     and `run_sam3d.py` refuses mismatched sizes.
   - **Mask policy**: draw the visible object only (not occluders), including legs, with a
     tight outline (a few px of background is fine). One polygon per visible part; Enter
     closes each. 1-2 minutes per item.
   - Optional: box-prompted **SAM 2** (Apache-2.0, ungated): `pip install sam2`, then
     `make_mask.py sam2 data/sofa_01 --box X0 Y0 X1 Y1`. It uses
     `SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")` and
     `predict(box=...)` as in the SAM 2 README. **Untested here.** It should run on CPU,
     slowly. SAM 3 (text prompts like "sofa") also works but needs torch >= 2.7 / CUDA 12.6,
     so it can't share SAM 3D's env (torch 2.5.1 / cu121).
   - Masks from cropped cutouts (iOS "Copy Subject") can't be registered to the photo, so
     `make_mask.py alpha` only accepts same-size RGBA cutouts.
4. Fill `meta.json` `"true_footprint"` (`"rect"`, `"round"`, or `{"points": [...]}`) for the
   IoU metric where the shape is known (EVAL.md section 1). Web items without one are fine:
   G2 just skips them, and the report says how many items G2 covered.
5. Dry run: `python run_sam3d.py --data data --out outputs --dry-run` must report 0 invalid.

## 4. Runbook on the rented GPU

### Runpod quickstart (the plan in section 2)
1. Console → Settings → Secrets: create `hf_token` with your Hugging Face token (needs the
   approved access to `facebook/sam-3d-objects`).
2. Create the pod per section 2 (or ask Claude to create it via the Runpod connector), then
   open Jupyter (`https://<podId>-8888.proxy.runpod.net`) → Terminal.
3. Put this repo at `/workspace/vlm-pipelines`: `git clone <repo URL> /workspace/vlm-pipelines`
   (a private repo needs a GitHub token), or upload a tarball through Jupyter and extract it.
4. `bash /workspace/vlm-pipelines/spikes/sam3d/runpod_bootstrap.sh --smoke` (~1 h first time).
5. Upload your prepared `spikes/sam3d/data/` folder through Jupyter into the same path.
6. Run the command the script prints. Then zip `outputs/`, download it through Jupyter, and
   **terminate the pod**.

The manual equivalent of steps 2–4 on any provider:

```bash
# 0. Instance: Ubuntu 22.04, 1x 48 GB GPU (A6000/L40S/6000 Ada), >= 100 GB disk,
#    NVIDIA driver >= 530 (CUDA 12.1 capable). Check:
nvidia-smi

# 1. Mamba + SAM 3D Objects env (doc/setup.md, verbatim apart from paths)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && source ~/miniforge3/bin/activate
git clone https://github.com/facebookresearch/sam-3d-objects.git ~/sam-3d-objects
cd ~/sam-3d-objects && git checkout f91db41   # the commit we read; drop to try newer
mamba env create -f environments/default.yml
mamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://download.pytorch.org/whl/cu121"  # pypi.ngc.nvidia.com no longer resolves
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra

# 2. Checkpoints (access must already be approved)
pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download

# 3. Smoke test with Meta's own demo (writes splat.ply)
python demo.py

# 4. Our repo + roomlayout venv (CPU side; separate interpreter, ADR-0004)
git clone <our repo> ~/vlm-pipelines && cd ~/vlm-pipelines
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv pip install -e ".[assets]"
# copy the prepared data/ folder: scp -r spikes/sam3d/data gpu:~/vlm-pipelines/spikes/sam3d/

# 5. Run (inside the sam3d-objects env; roomlayout runs via its own python)
cd ~/vlm-pipelines/spikes/sam3d
mamba activate sam3d-objects
python run_sam3d.py --data data --out outputs --sam3d-repo ~/sam-3d-objects \
    --roomlayout-python ~/vlm-pipelines/.venv/bin/python --save-splat 2>&1 | tee outputs_log.txt

# 6. Copy results home, then shut the instance down
tar czf sam3d_outputs.tgz outputs outputs_log.txt     # scp it back
```

Back on the laptop:

```bash
cd spikes/sam3d
python run_sam3d.py --data data --out outputs --skip-generation   # optional re-normalize
python score.py --data data --out outputs    # 1st run writes outputs/scores.csv template
# rate each item (EVAL.md section 2) using outputs/<id>/normalized/<id>.front_candidates.png
# and a GLB viewer on outputs/<id>/sam3d.glb, then:
python score.py --data data --out outputs    # -> outputs/results.md with the verdict
```

`results.md` has one decision over all items, how many items G2 (footprint IoU) covered,
the dims-provenance mix, descriptive stats (failure tags, mismatch, runtime/VRAM) and a
per-item table with a `dims_provenance` column. `results.json` holds the same numbers in
machine-readable form.

To normalize a single mesh by hand (this is what the harness runs):

```bash
.venv/bin/python -m roomlayout.assets outputs/sofa_01/sam3d.glb --category sofa \
    --dims 2.10 0.92 0.83 --up y --geometry-from sam3d --id sofa_01 --out outputs/sofa_01/normalized
```

### If something breaks
- `KeyError: 'CONDA_PREFIX'`: the env isn't activated (`notebook/inference.py` reads it).
- Mask/image size mismatch: re-run `make_mask.py prepare` and redraw the mask.
- CUDA OOM: the harness logs it, frees the cache and continues. Move to an 80 GB A100.
- One item failing never stops the batch. See `outputs/<id>/generation_error.txt` and
  `outputs/<id>/normalized/cli_stderr.txt`.
