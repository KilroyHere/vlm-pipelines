# Spike: end-to-end on real rooms (ARKitScenes)

The whole pipeline (room scan + furniture boxes + video frames → SAM 3D assets → user
request → constraints → layouts → checks) on real LiDAR captures from Apple's
[ARKitScenes](https://github.com/apple/ARKitScenes), standing in for the user's own
scan until we have one. Results: [RESULTS.md](RESULTS.md).

ARKitScenes is under Apple's non-commercial license (see its LICENSE). Data stays in
`data/` and `out/` (git-ignored) and is for internal evaluation only.

## Why ARKitScenes

Each capture is an iPad Pro LiDAR scan, the same hardware class as RoomPlan, with:
a reconstructed mesh, oriented 3D boxes for 17 furniture categories (≈ RoomPlan objects),
10 Hz camera poses, and 1920×1440 video frames (for the ~40% of captures in the
"upsampling" subset). We use validation-split living rooms that have a sofa and a table,
no kitchen appliances, and full-resolution frames.

## Runbook

All commands from this directory, with the repo venv (`../../.venv/bin/python`; it
needs `requests`).

```bash
# 1. data: annotation, poses, mesh, frame index (~40-60 MB per room; frames on demand)
python fetch_scene.py 42445441 47204607 48018379 --out data

# 2. LayoutProblem with the real arrangement as `initial` (+ real_layout.png, outline_check.png)
python to_problem.py data/42445441 data/47204607 data/48018379

# 3. best frame per object -> SAM 3D item folders (image + meta.json with prompt_box)
python pick_frames.py data/42445441 data/47204607 data/48018379 --items data/items

# 4. masks: SAM 2 prompted by the projected scan box (CPU is fine, ~1 min/item)
(cd ../sam3d && $MASKENV/bin/python make_mask.py auto ../e2e_arkitscenes/data/items/*/)
#    eyeball data/items/*/mask_check.png; re-mask a bad one with the text detector by
#    renaming its meta "prompt_box" and setting "prompt" (see 47204607__table_1)

# 5. SAM 3D + normalization on the GPU pod (../sam3d/runpod_bootstrap.sh, then)
python ../sam3d/run_sam3d.py --data data/items --out data/sam3d_outputs \
    --sam3d-repo /workspace/sam-3d-objects --roomlayout-python /workspace/vlm-pipelines/.venv/bin/python

# 6. end to end: assets + request -> constraints -> solve -> checks -> compare with real
python run_e2e.py 42445441 47204607 48018379 --assets data/sam3d_outputs --out out/sam3d
python run_e2e.py 42445441 47204607 48018379 --assets data/no_assets --out out/boxes  # baseline
```

`requests.json` holds one request per room in the user's words plus the VLM's
structured answer. No Anthropic API key is configured here, so the answers are written
by hand in the VLM output format and served through the real SDK offline (as in
`tests/test_integration.py`); swap in a live client once a key exists.

## What stands in for what

| product | here |
|---|---|
| RoomPlan walls/floor | outline rebuilt from the scan mesh (`to_problem.py`): floor points + furniture footprints, closed, doorways cut, rectilinear |
| RoomPlan objects (+ TableType) | ARKitScenes oriented boxes; coffee/side table from size |
| object front | heuristic: back to the nearest wall, else sofas face the TV stand, chairs a table |
| user's video | the capture's `wide` frames + ARKit poses |
| VLM call | hand-written structured answers, served offline |

Known gaps: no doors or windows (not annotated), so door rules are not exercised; a TV
standing on furniture is folded into that furniture as the `tv_stand`.
