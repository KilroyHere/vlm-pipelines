# SAM 3D pilot: 7 web-sourced items (2026-10-17)

**Verdict: promising, not yet a go/no-go.** Every item produced a usable mesh; shapes are
recognisable and proportions match manufacturer dimensions. 7 items is too few, with no
true footprints and none of the hard phone conditions (low light, back-only, top-down), for
the formal decision (EVAL.md expects 20).

Pipeline (all automatic): listing photo → OWLv2 box → SAM 2 mask (checked by eye, all usable) →
SAM 3D on an A40 → `roomlayout.assets` normalization (fit to listing dims where known).

| item | photo | shape (1–5) | front | aspect mismatch | notes |
|---|---|---|---|---|---|
| sofa_01 (IKEA KIVIK) | product | 3 | A ✓ | 6.5% | **2-seat instead of 3-seat**; footprint/proportions fine |
| sectional_04 (KIVIK + chaise) | product | 4 | A ✓ | 8.0% | L preserved (bbox fill 0.68 vs ~0.73 ideal) |
| dining_chair_07 (STEFAN) | product | 5 | B ✓ | 0.6% | thin legs, stretchers, 3 slats correct |
| dresser_16 (HEMNES 8-drawer) | product | 4 | A ✓ (low conf.) | 10.3% | drawers + knobs, legs |
| armchair_05 (wingback) | in-room | 4 | A ✓ | – (no dims) | wings + cabriole legs; back hallucinated plausibly |
| coffee_table_10 | in-room, clutter | 4 | n/a (symmetric) | – | clutter removed, base simplified |
| desk_13 | in-room, clutter, cut off | 4 | A ✓ (low conf.) | – | clutter removed, cut-off side completed |

Ratings are the lead's first pass from renders (`outputs/*/normalized/*.front_candidates.png`,
local only, not in git); the owner confirms. G1 (shape ≥ 3 on in-room items) 3/3; G3 7/7;
G5 4/4; G2 no data (no true footprints); G4: generation **14–20 s/item** on an A40 (pod log;
the local re-normalization overwrote the timing columns).

Cost: the pilot run ≈ $0.60 of A40 time (≈ 2 min of generation; the rest is env setup).
Front heuristic: 6/6 non-symmetric items correct (2 with low confidence).

## Lessons
- 12/20 listing sites block scripted downloads (AWS WAF, Cloudflare, Vercel); not worked around.
- IKEA JSON-LD `depth` is the *minimum* depth; use the Measurements table's Max. depth (fixed).
- Pod venv: the template's default Python is 3.10; roomlayout needs ≥ 3.11, so normalization
  failed on the pod and was run locally. Fix before relying on pod-side normalization.
- A stopped pod can lose its GPU to another renter; budget a re-setup (~45 min, now reliable).

## Next
Full 20-item set (owner saves blocked images or uses own room photos) for the formal
go/no-go; then an end-to-end test in the owner's room (see PROJECT_PLAN).
