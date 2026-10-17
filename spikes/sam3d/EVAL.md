# SAM 3D spike: evaluation protocol

The question: **are SAM 3D Objects meshes of users' own furniture, made from one phone photo,
good enough to place in layouts and show back to the user?** Layout correctness doesn't
depend on SAM 3D's scale (measured dims win, ADR-0004). It depends on **shape** (footprint,
proportions) and on users accepting the look.

The numbers come from `run_sam3d.py` (timings, VRAM, mismatch) and `score.py` (IoU,
aggregation, verdict). The ratings come from two people scoring independently, then
reconciling.

## 1. Test set (20 items)

Every item: one photo, a mask (README section 3), and **tape-measured W x D x H in meters**
(W = side to side seen from the front, D = front to back, H = floor to highest point; for
L-sofas the overall bounding box). Note how the measurement was taken in `meta.json` `notes`.

| # | category (`meta.category`) | condition | why |
|---|---|---|---|
| 1 | sofa | product photo | baseline |
| 2 | sofa | in-room, phone, 3/4 view | core case |
| 3 | sectional | in-room, phone | non-rectangular footprint (L) |
| 4 | sectional | product photo | L from a clean view |
| 5 | armchair | in-room, partial occlusion (side table / person) | occlusion |
| 6 | armchair | in-room, low light (evening, lamps only) | low light |
| 7 | dining_chair | product photo | thin legs and slats (64^3 limit) |
| 8 | dining_chair | in-room, partly under a table | heavy occlusion |
| 9 | office_chair | in-room | star base, thin parts |
| 10 | coffee_table | in-room, cluttered top | clutter on the object |
| 11 | round_table | in-room | round footprint |
| 12 | dining_table | in-room, chairs around | occlusion by chairs |
| 13 | desk | in-room, cluttered background | clutter |
| 14 | bed | in-room, made bed with pillows | soft, large, partly out of frame |
| 15 | bookshelf | in-room, filled with books | open front vs. contents |
| 16 | dresser | product photo | drawers, flat front |
| 17 | wardrobe | in-room, straight-on photo | front-only view (back is hallucinated) |
| 18 | tv_stand | in-room, TV on it | object on object |
| 19 | sofa | in-room, back view only | hallucinated front |
| 20 | armchair | in-room, top-down-ish phone angle | unusual viewpoint |

Rules: at least 14 in-room items (the product promise) and at least 4 product photos.
Photos from a recent iPhone or Android at default settings. No retouching. In-room photos
taken from where a user would naturally stand. Tag conditions in `meta.condition` with
`product`, `in_room`, `occluded`, `low_light`, `clutter`, `back_view`, `odd_angle`.

**True footprint** (`meta.true_footprint`): `"rect"` for boxy items, `"round"` for round or
oval tops, otherwise `{"points": [[x, y], ...]}` in meters in the asset frame (origin at the
centre of the W x D box, +Y = front, CCW), from tape measurements. Example, a 2.8 x 1.6 L
with a 0.8 m wide chaise at the right front and 0.9 m deep main seats:
`[[-1.4,-0.8],[1.4,-0.8],[1.4,0.8],[0.6,0.8],[0.6,0.1],[-1.4,0.1]]`.
For a rectangle the IoU is almost guaranteed by the fit to dims. The informative items are
the L-sofas, round/oval tables and chairs with splayed legs. Report them separately.

### Dev set (web-sourced): not part of the go/no-go set

A separate warm-up set of web photos (retailer pages, resale/estate-sale listings) mapped
1:1 to the 20 cases above; sources in [dev_web_sources.md](dev_web_sources.md).
- **Purpose:** exercise masks, generation, normalization and scoring as soon as GPU access
  arrives, and catch failure modes cheaply.
- **Limits:** dims are manufacturer, seller-reported or estimated (unverified). There are
  usually no true footprints, and phone/capture settings are unknown. **These items can't
  inform go/no-go.**
- **Mechanics:** `meta.json` `"set": "dev_web"` (default `"go_no_go"`), plus
  `"dims_provenance"` (`tape_measured` | `manufacturer` | `seller_reported` | `estimated`)
  and `"source_url"`. Dev items are rated with the same rubric (section 2). `score.py`
  reports them in their own section and **never** in a G1-G5 denominator.
- Go/no-go items must be tape-measured (`dims_provenance: tape_measured`, the default for
  `make_mask.py prepare`). A dev photo that becomes a properly captured, measured item
  (e.g. case #6 shot by the product owner) moves to `go_no_go`.

## 2. Rubric (manual, per item, in `outputs/scores.csv`)

**Shape fidelity, 1-5** (look at the raw `sam3d.glb` in a viewer and the contact sheet;
compare with the photo, and for hidden sides with what a person would expect):

| score | anchor |
|---|---|
| 5 | Could pass for a scan: every major part present and correctly proportioned; small details (legs, arm shape, cushions) right. |
| 4 | Clearly the same object: all major parts present, proportions right; minor detail lost or blobby (thin legs merged, cushion seams gone). |
| 3 | Recognizably the object and *usable for layout*: footprint and height profile right, but one notable defect (a missing leg, a fused part, a soft/blobby back). |
| 2 | Recognizable category but wrong in a way users would notice in a layout: wrong topology (L became straight), wrong proportions (>20 % off after fitting), large hallucinated parts. |
| 1 | Unusable: wrong object, fragment, merged with background or occluder, not upright. |

**Texture plausibility, 1-5**: 5 = matches the photo's colours/materials on visible sides and
plausible on hidden ones; 3 = right base colours, smeared detail; 1 = wrong colours,
background baked in.

**Front** (`front_correct_letter`): which letter on `normalized/<id>.front_candidates.png` is
the true front. `score.py` compares it with the heuristic's choice. This measures *our* front
heuristic, not SAM 3D.

**Failure tags** (`failure_tags`, `;`-separated, from this closed set):
`missing_parts` (leg, arm, backrest missing), `fused_parts` (legs merged into a slab, object
fused with an occluder), `wrong_topology` (L became straight, holes closed/opened),
`hallucinated_back` (implausible unseen side), `wrong_proportions`, `not_upright`, `floaters`
(debris that survived cleanup), `background_leak` (floor/wall pieces in the mesh),
`thin_parts_lost`, `texture_wrong`, `other` (explain in `comments`).

**Automatic metrics** (no rating needed):
- *Footprint IoU*: our extracted footprint (after fit to measured dims) vs. the true
  footprint. We report it with the heuristic's front and best-of-4 yaws; the go/no-go uses
  best-of-4 so front mistakes don't count twice.
- *Aspect mismatch*: from our report, `max_i |s_i / s_geo - 1|` over the per-axis scale
  factors. It's how far SAM 3D's proportions are from the tape measure.
- *Generation time* (GPU seconds per item, model loaded once) and *peak VRAM*.
- *Normalization warnings*: debris share, up-axis suspicion, tilt corrected, footprint rule.

## 3. Cost

`cost/item = median generation seconds x GPU $/h / 3600`. Note the instance type and price
in the results. The model-load time (once per process) is reported separately in the log.

## 4. Results template

`score.py` writes `outputs/results.md` in this form. Paste it into the PR / status report.

| item | category | condition | status | gen s | VRAM GB | shape | texture | IoU (front / best) | mismatch | front auto/true (conf) | failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sofa_01 | sofa | in_room | ok | | | | | / | | / ( ) | |
| ... | | | | | | | | | | | |

Summary block: the five criteria below with value, threshold and pass/FAIL, computed over
`go_no_go` items only (the report states the count), then texture pass rate and
front-heuristic accuracy. The `dev_web` items follow in their own section with the same
per-item columns and descriptive statistics, but no verdict. Also write a short paragraph on the
dominant failure tags and one contact-sheet example each of the best and worst item.

## 5. Go / no-go thresholds

All rates below are over **`go_no_go` items only**. Dev-set items are excluded.

Blocking (all must pass for GO):

| id | criterion | threshold |
|---|---|---|
| G1 | in-room items with shape score >= 3 | **>= 70 %** |
| G2 | items with footprint IoU >= 0.80 (best-of-4, items with a true footprint) | **>= 80 %** |
| G3 | items that produce a mesh and normalize without error | **>= 90 %** |

Non-blocking (failing these gives **CONDITIONAL GO** with a mitigation plan):

| id | criterion | threshold | mitigation if failed |
|---|---|---|---|
| G4 | median generation time per item | <= 120 s | distillation flags, smaller GPU batch, async UX |
| G5 | items with aspect mismatch <= 15 % | >= 75 % | warn the user, offer retake / catalog fallback |

Reported, not gating: texture >= 3 on >= 60 % of items (visual appeal; can improve later
with texture baking), front-heuristic accuracy (decides whether the VLM front check is
mandatory or optional).

**Decision rules**
- **GO**: G1-G5 pass. Build M3/M4 on SAM 3D.
- **CONDITIONAL GO**: G1-G3 pass, G4 or G5 fail. Proceed, with the mitigation in the plan.
- **GO for product photos only**: G1/G2 fail on in-room items but pass on product photos.
  Ship the feature for "add a product from a listing" and keep in-room capture on
  boxes/catalog until a better generator lands.
- **NO-GO**: any of G1-G3 fails on both subsets. Fall back to catalog retrieval /
  Object Capture (risk register), keeping `roomlayout.assets` unchanged (ADR-0004).
- Independently of quality: **if counsel says the SAM License doesn't allow our
  commercial use, it's NO-GO** regardless of scores.
