# "Will it fit?" from listing sizes (`roomlayout.fit`)

The v1 question (ADR-0009), answered from the manufacturer's dimensions alone, with no 3D
model and no GPU (ADR-0010).

```bash
python -m roomlayout.fit data/roomplan_public/apple_myhome_bedroom2.json \
    --products examples/products/ikea_2026-11.json \
    --ids malm_bed_queen,micke_desk,billy_bookcase --out out/fit_demo [--request "..."]
# -> out/fit_demo/{problem,result,fit}.json; prints "fits" or the advice
```

```python
from roomlayout.assets import load_products
from roomlayout.fit import check_fit
from roomlayout.pipeline import empty_room, problem_from_roomplan

room, _ = empty_room(problem_from_roomplan("Room.json")[0])
report = check_fit(
    room, load_products("examples/products/ikea_2026-11.json", ["soderhamn_sectional"])
)
report.fits, report.layouts, report.advice
```

## Products (`roomlayout.assets.listing`)
A product file lists `ProductSpec`s: id, name, url, category/subtype (shared taxonomy),
`width_m`, `depth_m`, `height_m`, an optional `l_shape` (seat depth, chaise width, chaise
side) and `count` (e.g. 4 chairs). `box_assets(spec)` makes `Asset`s with
`dimensions_from="listing"`, `geometry_from="box"`, an L-shaped footprint for sectionals,
and the category's default clearance zones.

`examples/products/ikea_2026-11.json` holds the five test pieces: SÖDERHAMN
sectional with chaise, LISABO table + 4 chairs, MALM queen bed, MICKE desk and BILLY
bookcase. Sizes are from the IKEA US pages (retrieved 2026-11-18), with the page's own
wording kept in `listing`. One value is estimated, the SÖDERHAMN chaise width; the entry
says so in `estimated`.

## When it doesn't fit
`check_fit` runs the same problem as `pipeline.plan` (request + rule library + solver). If
no layout is feasible it tries variants of the *built* problem: the same constraints,
minus those that refer to removed pieces.

| Variant | Runs | Advice |
|---|---|---|
| drop one product (all of a multi-piece product together) | one per product | "Without the gray L-shaped sectional, the rest fits." |
| narrower biggest piece (sofa, bed, table or storage; depth kept, chaise kept) | 1 + 4 (binary search from 50% to 100% of the listing width) | "A gray L-shaped sectional up to 2.37 m wide fits (this one is 2.91 m)." / "Even at 50% of its width ... its depth is the problem." |

Variants run with `k=1` and a third of the time budget (at least 2 s). The solver's own
notes (crowding percentage, most-violated rules) stay in `report.reasons`.

Timing: a "fits" answer takes about 4 s on a real scan (`--seconds 5`). A "doesn't fit"
answer with advice takes 10-20 s.

## Limits
- Box footprints: chairs can't tuck under the table (ADR-0003 backlog), so dining sets in
  tight rooms are reported as not fitting when they would.
- Only one piece is resized, the biggest one.
- One change at a time: no "drop the desk *and* take the smaller sofa".
