# Image-to-3D evaluation (Phase 2): SAM 3D vs TRELLIS on ABO

Status, 2026-11-25: **SAM 3D scored; TRELLIS never produced output.** So this is not the
head-to-head comparison it was meant to be. It does show whether SAM 3D's geometry is good
enough for our pipeline.

## Setup

- 23 ABO items (`spikes/img3d_eval/abo_selection.json`): sofas, sectionals, chairs, tables,
  beds, storage. 22 are white-background product photos and 1 (`bed_03`) is an in-room
  photo masked with rembg. ABO ships a GLB for each one, which serves as ground truth.
- Each generator mesh goes through `roomlayout.assets.normalize` and is fitted to the true
  W x D x H, as the product pipeline does with listing sizes. Then
  `spikes/img3d_eval/score_abo.py` scores it:
  - Chamfer distance, and F-score at 2 % and 5 % of the true bbox diagonal;
  - the footprint IoU the solver would see;
  - whether our front heuristic chose the right yaw;
  - the raw aspect mismatch (shape error before we force the size).
- GPU: one Runpod A40 (48 GB), run unattended by `spikes/img3d_eval/pod_job.sh`.

## SAM 3D results (run 1, vertex colours)

Per-item numbers: `spikes/img3d_eval/results/sam3d_scores.json`.

| metric | median | worst |
|---|---|---|
| F-score @ 5 % | 0.996 | 0.90 (sofa_01, chair_04) |
| F-score @ 2 % | 0.85 | 0.60 (chair_04 office chair), 0.63 (bookshelf_02) |
| Chamfer (% of diagonal) | 1.17 | 2.27 (chair_04) |
| Footprint IoU | 0.97 | 0.80 (chair_04), 0.84 (chair_02) |
| Front picked right | 22 / 23 | ottoman_01 off by 180° (it is symmetric, so harmless) |
| Generation time | 16 s | 28 s; peak VRAM 20 GB |

Raw aspect mismatch is above 15 % on two items, `bed_01` and `dresser_01`. Fitting to the
listing box hides it, and the normalize report flags it.

Reading the numbers:
- F@5 % saturates because the fit to the true box already fixes the overall size. F@2 %
  and the footprint IoU are the informative columns.
- Ground truth scored against itself is the ceiling: F@5 % 1.00, IoU ~1.0, front right
  on 5 of 6 checked items.
- Thin, open structures score worst: the office chair's star base and the open
  bookshelf. Boxy furniture (tables, dressers, benches, beds) gets F@2 % ≥ 0.9.

**Verdict on SAM 3D: go for footprints and layout.** Its footprints reach a median IoU of
0.97, and only one item falls below 0.84. That is accurate enough for the solver, whose
real input is the listing box anyway. For display, its geometry is fine but it has only
vertex colours: texture baking never ran (below), so the look is soft and blurry next to
the ABO models.

**TRELLIS: no verdict.** It produced nothing, so we can't say whether it beats SAM 3D.

## Known failures

1. **TRELLIS did not run.** Its environment built, but `import trellis` loads open3d,
   which needs `libEGL.so.1`. The Runpod `pytorch:2.2.0` image doesn't have it. Fix:
   `apt-get install -y libegl1 libgl1 libgomp1` in `trellis_setup.sh`.
2. **SAM 3D texture baking failed on all 23 items.** The error was
   `name 'GaussianRasterizationSettings' is not defined`: SAM 3D's renderer needs
   `diff_gaussian_rasterization`, and its environment doesn't have it. `run_sam3d.py` fell
   back to vertex colours, so the rerun's SAM 3D meshes match run 1. Fix: build
   mip-splatting's `diff-gaussian-rasterization` into the sam3d env with
   `--no-build-isolation`.
3. **Setup bugs, fixed in the rerun:**
   - pip reads `PIP_NO_BUILD_ISOLATION` inverted (`=1` keeps isolation on);
   - conda's `activate` inherited the job script's arguments;
   - TRELLIS's `setup.sh` compares `torch.__version__` exactly and skipped
     xformers/kaolin for `2.4.0+cu121`;
   - a newer uv refuses an existing venv.

## Cost

Pod `wrqvvcc2yy8pal`, from Runpod billing on 2026-11-25:
- GPU: **$10.46**;
- disk: **$3.16**. The 120 GB volume costs about $0.80 a day while the pod is stopped.
- total: **$13.62**, against an $8 cap.

How it went over: the rerun job finished at about 03:09 UTC on 11-19. The safety stop
scheduled for 04:00 fired on time, but its stop call sat on a permission prompt. The pod
was only stopped at about 16:58, so it idled about 13 h at $0.59/h (~$7.70 for nothing).

**Lesson:** a GPU job has to stop its own pod from inside the container. Ways to do that:
- `runpodctl stop pod $RUNPOD_POD_ID` at the end of `pod_job.sh`;
- a `sleep N && runpodctl stop pod ...` guard started first thing.

An agent check-in that needs approval to run is not a safety stop. The pod can't stop
itself with the API key it has now, so the next run needs a pod-scoped key or
`runpodctl` set up before it launches.

## Next run (if TRELLIS is still wanted)

1. Apply both fixes above.
2. Add the in-container stop.
3. Run `ONLY=trellis` and then `ONLY=sam3d` (textured) with a 1 h hard limit, about $0.60.
4. Score with `score_abo.py --methods trellis`.
5. Make side-by-side shots with `compare.html` / `compare_shots.js`.
