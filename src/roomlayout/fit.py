""" "Will it fit?" from listing sizes, with an actionable answer when it doesn't (ADR-0009).

    from roomlayout.fit import check_fit
    report = check_fit(room, specs)           # room: an emptied scan (pipeline.empty_room)
    report.fits, report.layouts, report.advice

Box assets come straight from the listing (``roomlayout.assets.listing``), so this needs
no 3D model and no GPU. When nothing fits, it tries cheap variants of the same problem
(same constraints, rule library included) and says which change would make it fit:

- drop one product (all of a multi-piece product, e.g. the 4 chairs, count as one), and
- the widest version of the biggest piece that fits (binary search on its width, depth
  kept, L-shapes keep their chaise).

Orchestration level, like ``roomlayout.pipeline``: uses the solver, assets and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roomlayout.assets import ProductSpec, box_assets, with_default_zones
from roomlayout.constraints import BuildResult
from roomlayout.pipeline import furnish, plan
from roomlayout.schema import LayoutProblem, SolveResult
from roomlayout.solver import solve

# Pieces worth resizing ("a sofa up to 2.40 m wide fits"); small items are better dropped.
RESIZABLE = ("sofa", "bed", "table", "storage")
MIN_WIDTH_FRACTION = 0.5  # don't suggest anything narrower than half the listing width


@dataclass
class DropOption:
    product_id: str
    label: str
    fits: bool


@dataclass
class WidthOption:
    product_id: str
    label: str
    listing_width_m: float
    max_width_m: float | None  # None: not even at MIN_WIDTH_FRACTION of the listing width


@dataclass
class FitReport:
    fits: bool
    built: BuildResult
    result: SolveResult
    free_floor_m2: float
    furniture_m2: float
    reasons: list[str] = field(default_factory=list)  # solver notes when it doesn't fit
    drops: list[DropOption] = field(default_factory=list)
    width: WidthOption | None = None

    @property
    def layouts(self):
        return [lay for lay in self.result.layouts if lay.feasible]

    @property
    def advice(self) -> list[str]:
        """Plain-English suggestions, best first (empty when it fits)."""
        if self.fits:
            return []
        out = [f"Without the {d.label}, the rest fits." for d in self.drops if d.fits]
        w = self.width
        if w is not None and w.max_width_m is not None:
            out.append(
                f"A {w.label} up to {w.max_width_m:.2f} m wide fits "
                f"(this one is {w.listing_width_m:.2f} m)."
            )
        elif w is not None:
            out.append(
                f"Even at {MIN_WIDTH_FRACTION:.0%} of its width the {w.label} doesn't fit: "
                "its depth is the problem."
            )
        if not out:
            out.append("No single change found: try a smaller set of pieces.")
        return out


def free_floor_area(problem: LayoutProblem) -> float:
    """Floor area minus obstacles and the footprints of fixtures that stay (approximate:
    overlaps between them are not subtracted twice)."""
    taken = sum(o.footprint.area for o in problem.room.obstacles)
    taken += sum(a.effective_footprint().area for a in problem.assets if not a.movable)
    return problem.room.floor.area - taken


def _without(problem: LayoutProblem, ids: set[str]) -> LayoutProblem:
    data = problem.model_dump(mode="json")
    data["assets"] = [a for a in data["assets"] if a["id"] not in ids]
    data["initial"] = {k: v for k, v in data["initial"].items() if k not in ids}
    data["constraints"] = [c for c in data["constraints"] if not ({c.get("a"), c.get("b")} & ids)]
    return LayoutProblem.model_validate(data)


def _resized(problem: LayoutProblem, spec: ProductSpec, width: float) -> LayoutProblem:
    """The problem with ``spec``'s pieces at ``width`` (zones rebuilt for the new size)."""
    shape = spec.l_shape
    if shape is not None:  # keep the chaise; it can't be wider than the piece
        shape = shape.model_copy(update={"chaise_width_m": min(shape.chaise_width_m, width)})
    new = {
        a.id: a for a in box_assets(spec.model_copy(update={"width_m": width, "l_shape": shape}))
    }
    data = problem.model_dump(mode="json")
    for i, a in enumerate(problem.assets):
        if a.id in new:
            b = new[a.id].model_copy(update={"clearance_zones": [], "label": a.label})
            data["assets"][i] = with_default_zones(b).model_dump(mode="json")
    return LayoutProblem.model_validate(data)


def _feasible(problem: LayoutProblem, seconds: float, seed: int) -> bool:
    res = solve(problem, k=1, seed=seed, time_budget_s=seconds)
    return any(lay.feasible for lay in res.layouts)


def _ids(spec: ProductSpec) -> set[str]:
    return {a.id for a in box_assets(spec)}


def check_fit(
    room: LayoutProblem,
    specs: list[ProductSpec],
    text: str | None = None,
    *,
    client: Any = None,
    k: int = 3,
    seed: int = 0,
    seconds: float = 10.0,
    advise: bool = True,
    variant_seconds: float | None = None,
) -> FitReport:
    """Solve ``room`` furnished with ``specs`` as listing-size boxes; when nothing fits
    and ``advise``, try drop-one and narrower-piece variants.

    Product ids must be unique in the room (``furnish`` would rename clashes, and the
    advice refers to products by id)."""
    assets = [a for s in specs for a in box_assets(s)]
    taken = {a.id for a in room.assets}
    clash = sorted(a.id for a in assets if a.id in taken)
    if clash:
        raise ValueError(f"product ids already used in the room: {clash}")
    problem = furnish(room, assets)
    built, result = plan(problem, text, client=client, k=k, seed=seed, seconds=seconds)
    fits = any(lay.feasible for lay in result.layouts)
    furniture = sum(a.effective_footprint().area for a in assets)
    report = FitReport(
        fits=fits,
        built=built,
        result=result,
        free_floor_m2=free_floor_area(room),
        furniture_m2=furniture,
        reasons=[] if fits else list(result.notes),
    )
    if fits or not advise:
        return report
    vs = variant_seconds if variant_seconds is not None else max(2.0, seconds / 3)
    full = built.problem
    for s in specs:
        if len(specs) == 1:
            break
        report.drops.append(
            DropOption(s.id, s.label or s.name, _feasible(_without(full, _ids(s)), vs, seed))
        )
    big = [s for s in specs if s.category in RESIZABLE]
    if big:
        spec = max(big, key=lambda s: s.width_m * s.depth_m)
        lo, hi = spec.width_m * MIN_WIDTH_FRACTION, spec.width_m
        best = None
        if _feasible(_resized(full, spec, lo), vs, seed):
            best = lo
            for _ in range(4):  # ~3% of the listing width
                mid = (lo + hi) / 2
                if _feasible(_resized(full, spec, mid), vs, seed):
                    best, lo = mid, mid
                else:
                    hi = mid
        report.width = WidthOption(spec.id, spec.label or spec.name, spec.width_m, best)
    return report


def report_json(report: FitReport) -> dict[str, Any]:
    """JSON-ready summary (the app/viewer reads this next to the solve result)."""
    return {
        "fits": report.fits,
        "advice": report.advice,
        "free_floor_m2": round(report.free_floor_m2, 2),
        "furniture_m2": round(report.furniture_m2, 2),
        "reasons": report.reasons,
        "drops": [d.__dict__ for d in report.drops],
        "width": None if report.width is None else report.width.__dict__,
        "feasible_layouts": len(report.layouts),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from roomlayout.assets import load_products
    from roomlayout.pipeline import empty_room, problem_from_roomplan

    ap = argparse.ArgumentParser(description="Will these products fit in this scanned room?")
    ap.add_argument("capture", type=Path, help="RoomPlan CapturedRoom JSON")
    ap.add_argument("--products", type=Path, required=True, help="products JSON")
    ap.add_argument("--ids", required=True, help="comma-separated product ids")
    ap.add_argument("--request", help="what the user wants, in their words (needs an API key)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    scanned, _ = problem_from_roomplan(args.capture)
    room, _ = empty_room(scanned)
    specs = load_products(args.products, args.ids.split(","))
    report = check_fit(room, specs, args.request, seconds=args.seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "problem.json").write_text(report.built.problem.model_dump_json(indent=1))
    (args.out / "result.json").write_text(report.result.model_dump_json(indent=1))
    summary = report_json(report)
    (args.out / "fit.json").write_text(json.dumps(summary, indent=1))
    print("fits" if report.fits else "does not fit", *report.advice, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
