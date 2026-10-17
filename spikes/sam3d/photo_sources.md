# Photo sources for the evaluation set

Where the photos for the 20 cases in [EVAL.md](EVAL.md) come from: retailer product pages,
resale and estate-sale listings, mapped 1:1 to the cases, plus the owner's own photos where
no listing fits (#6). All of them are part of the one evaluation and the one go/no-go
decision. Web listings bring their own limits: dims are manufacturer- or seller-reported or
estimated, capture settings are unknown, and a true footprint exists only where it's known.
Each item records this in `meta.json` (`dims_provenance`, `source_url`, `true_footprint` only
when known), and `score.py` shows it per item.

## Data handling
- Download images **by hand** into the gitignored `data/` directory (e.g.
  `data/sectional_03/`). Never commit photos, masks or outputs.
- Internal evaluation only. Listing photos show private homes, so outputs (meshes, renders,
  contact sheets) derived from them aren't published or shared outside the team.
- Record the page each photo came from in `meta.json` `source_url` (`make_mask.py prepare
  --source-url` strips `utm_*`/`fbclid`/`gclid` tracking parameters), and note in `notes`
  which image of the gallery was used.
- Retailer/listing images belong to their owners. Use them only for this internal evaluation.

Item ids: `<category>_<NN>` with NN the case number, e.g. `sofa_01`, `sectional_03`. Prepare
each item with `make_mask.py prepare ... --dims-provenance <source> --source-url <URL>`
(README section 3).

## Sources

| # | EVAL case | source URL | status | priority | dimension source |
|---|---|---|---|---|---|
| 1 | sofa, product photo | https://www.ikea.com/us/en/p/kivik-sofa-gunnared-light-green-s19499706/ | ready-ish: pick the isolated product image | START | manufacturer |
| 4 | sectional, product photo | https://www.ikea.com/us/en/p/kivik-sofa-with-chaise-gunnared-blue-s29499701/ | ready-ish: pick the isolated product image | START | manufacturer |
| 7 | dining_chair, product photo | https://www.ikea.com/us/en/p/stefan-chair-brown-black-00211088/ | ready-ish: pick the isolated product image | START | manufacturer |
| 16 | dresser, product photo | https://www.ikea.com/us/en/p/hemnes-8-drawer-dresser-black-brown-30576190/ | ready-ish: pick the isolated product image | START | manufacturer |
| 3 | sectional, in-room | https://offerup.com/item/detail/bddf17b5-f600-3968-b34c-d758a524f837 | strong scene match | START | check listing: seller_reported if stated, else estimated |
| 10 | coffee_table, cluttered top | https://thearchitecturedesigns.com/remove-things-into-living-room/ | strong scene match | START | none listed: estimated, or no dims |
| 13 | desk, cluttered background | https://www.estatesales.net/MN/Henderson/56044/4900365 | strong scene match | START | check listing: seller_reported if stated, else estimated |
| 15 | bookshelf, filled | https://www.aptdeco.com/product/blu-dot-open-plan-large-low-bookcase | strong scene match | START | check listing: seller_reported if stated, else estimated |
| 18 | tv_stand with TV | https://offerup.com/item/detail/2315be82-4d43-3387-9831-c1a1e8c0ab3d | strong scene match | START | seller_reported (W/D/H in listing) |
| 19 | sofa, back view only | https://www.whoppah.com/products/flexform-italian-design-sofa-cGGdJU99j | strong scene match: feed **only the rear-view image** | START | check listing: seller_reported if stated, else estimated |
| 20 | armchair, elevated view | https://www.deesup.com/en/shop/lc3-armchair-in-tubular-steel-and-leather-cassina-279099 | strong scene match | START | check listing: seller_reported if stated, else estimated |
| 2 | sofa, in-room 3/4 view | https://www.aptdeco.com/product/room-board-metro-sofa-76 | candidate gallery: needs image-level selection | later | check listing: seller_reported if stated, else estimated |
| 5 | armchair, partial occlusion | https://www.estatesales.net/CA/Rocklin/95765/4772826 | candidate gallery: check the occlusion is substantial | later | check listing: seller_reported if stated, else estimated |
| 8 | dining_chair, partly under table | https://www.aptdeco.com/product/a-dining-table-and-chairs-set/ | candidate gallery: needs image-level selection | later | check listing: seller_reported if stated, else estimated |
| 9 | office_chair | https://www.aptdeco.com/product/herman-miller-office-chair-6 | candidate gallery: is the star base visible? | later | check listing: seller_reported if stated, else estimated |
| 11 | round_table | https://www.aptdeco.com/product/round-glass-dining-table-11 | candidate gallery: glass top (extra challenge) | later | check listing: seller_reported if stated, else estimated |
| 12 | dining_table, chairs around | https://www.aptdeco.com/product/wood-dining-table-set-with-6-matching-chairs | candidate gallery: needs image-level selection | later | check listing: seller_reported if stated, else estimated |
| 14 | bed, made with pillows | https://www.aptdeco.com/product/custom-queen-bed-frame-and-pillows | candidate gallery: needs image-level selection | later | check listing: seller_reported if stated, else estimated |
| 17 | wardrobe, straight-on | https://www.aptdeco.com/product/ikea-pax-wardrobe-with-white-paneled-doors | candidate gallery: needs image-level selection | later | check listing: seller_reported if stated, else estimated |
| 6 | armchair, low light | needs sourcing | **needs sourcing** | – | – |

**#6 recommendation:** have the product owner photograph their own armchair at night
(lamps only) with a phone at default settings and tape-measure it. That gives an
item with tape-measured dims, a known capture and a true footprint
(`--dims-provenance tape_measured`, no `source_url`).

"check listing" means nobody has checked the page for dimensions yet. Whoever downloads the
image records what the page states (`seller_reported`), or uses `estimated` and says how in
`notes` when nothing is given. Only #1/#4/#7/#16 (IKEA, manufacturer) and #18 (seller
W/D/H) are known to state dimensions.
