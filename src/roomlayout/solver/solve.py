"""``solve()``: restarts (optionally in parallel), diversity selection, infeasibility notes.

Determinism: restart ``i`` uses RNG seed ``seed * 1_000_003 + i``; results are consumed
in restart-index order and the stopping rule only looks at that index-ordered prefix,
so the output depends only on ``(problem, k, seed)`` and the restart/iteration limits,
not on the number of workers or scheduling. The wall-clock ``time_budget_s`` is a
safety net: if it, rather than the stopping rule, ends the search, the result depends on
machine speed (``SolveDiagnostics.time_limited`` is then True).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass

from roomlayout.schema import (
    AgainstWall,
    DistanceAtLeast,
    Layout,
    LayoutProblem,
    Near,
    Placement,
    Pose2D,
    SolveResult,
    SolveStats,
)

from .anneal import RestartResult, default_iterations, run_restart
from .diversity import MIN_LAYOUT_DISTANCE, interchangeable_groups, select_diverse
from .model import Model, Pose, State

DEFAULT_MIN_RESTARTS = 16
DEFAULT_MAX_RESTARTS = 512
_POSTPROCESS_RESERVE = 0.05  # fraction of the budget kept for selection + reporting


@dataclass
class SolveDiagnostics:
    restarts: int  # restarts whose results were used
    restarts_feasible: int
    time_to_first_feasible_s: float | None
    feasible_layouts: int  # feasible layouts in the returned (diverse) top-k
    time_limited: bool  # True if the wall-clock budget, not the stopping rule, ended search
    seconds: float
    evaluations: int


@dataclass
class SolveOutcome:
    result: SolveResult
    diagnostics: SolveDiagnostics


# ---------------------------------------------------------------------------- workers

_WORKER: dict = {}


def _worker_init(problem_json: str) -> None:
    _WORKER["model"] = Model(LayoutProblem.model_validate_json(problem_json))


def _worker_run(seed: int, index: int, iterations: int, deadline: float, use_initial: bool):
    return run_restart(_WORKER["model"], seed, index, iterations, deadline, use_initial)


def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


# ---------------------------------------------------------------------------- solve


def solve(
    problem: LayoutProblem,
    k: int = 10,
    seed: int = 0,
    time_budget_s: float = 10.0,
    **kwargs,
) -> SolveResult:
    """Find up to ``k`` mutually diverse layouts, best first. See :func:`solve_detailed`
    for the keyword-only tuning arguments."""
    return solve_detailed(problem, k=k, seed=seed, time_budget_s=time_budget_s, **kwargs).result


def solve_detailed(
    problem: LayoutProblem,
    k: int = 10,
    seed: int = 0,
    time_budget_s: float = 10.0,
    *,
    workers: int | None = None,
    min_restarts: int = DEFAULT_MIN_RESTARTS,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    iterations: int | None = None,
) -> SolveOutcome:
    """Like :func:`solve`, plus diagnostics.

    Search stops at the first restart index ``i >= min_restarts - 1`` at which the
    results ``0..i`` already contain ``k`` mutually diverse feasible layouts, after
    ``max_restarts`` restarts, or when ``time_budget_s`` runs out.
    ``workers`` defaults to ``min(4, cpu_count)``; ``iterations`` (annealing steps per
    restart) defaults to ``300 + 250 * movable assets``.
    """
    t0 = time.monotonic()
    deadline = t0 + time_budget_s * (1 - _POSTPROCESS_RESERVE)
    model = Model(problem)
    groups = interchangeable_groups(model)
    iters = iterations if iterations is not None else default_iterations(model)
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, max_restarts))
    has_initial = any(p is not None for i, p in enumerate(model.initial) if not model.pinned[i])

    results: dict[int, RestartResult] = {}
    prefix: list[RestartResult] = []
    stop = False
    time_limited = False

    def consume() -> bool:
        """Extend the index-ordered prefix; return True when the stopping rule fires."""
        while len(prefix) in results:
            prefix.append(results.pop(len(prefix)))
            if len(prefix) >= min_restarts and _enough(prefix, k, groups):
                return True
        return len(prefix) >= max_restarts

    if workers == 1:
        i = 0
        while not stop and i < max_restarts:
            if time.monotonic() > deadline:
                time_limited = True
                break
            results[i] = run_restart(model, seed, i, iters, deadline, has_initial and i == 0)
            time_limited |= results[i].truncated
            stop = consume()
            i += 1
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp_context(),
            initializer=_worker_init,
            initargs=(problem.model_dump_json(),),
        ) as pool:
            pending: dict[Future, int] = {}
            next_index = 0
            while not stop:
                while (
                    len(pending) < 2 * workers
                    and next_index < max_restarts
                    and time.monotonic() < deadline
                ):
                    fut = pool.submit(
                        _worker_run,
                        seed,
                        next_index,
                        iters,
                        deadline,
                        has_initial and next_index == 0,
                    )
                    pending[fut] = next_index
                    next_index += 1
                if not pending:
                    time_limited = next_index < max_restarts
                    break
                done, _ = wait(
                    pending,
                    timeout=max(0.0, deadline - time.monotonic()) + 0.5,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    time_limited = True
                    break
                for fut in done:
                    pending.pop(fut)
                    r = fut.result()
                    time_limited |= r.truncated
                    results[r.index] = r
                stop = consume()
            for fut in pending:
                fut.cancel()

    return _finish(problem, model, prefix, groups, k, seed, t0, time_limited)


def _enough(prefix: list[RestartResult], k: int, groups) -> bool:
    cands = _candidates(prefix, feasible_only=True)
    if len(cands) < k:
        return False
    return len(select_diverse(cands, k, groups)) >= k


def _candidates(prefix: list[RestartResult], feasible_only: bool = False):
    out = []
    for r in prefix:
        if r.feasible or not feasible_only:
            out.append((r.poses, r.cost, r.feasible))
        for poses, cost in r.extra:
            out.append((poses, cost, True))
    return out


def _finish(
    problem: LayoutProblem,
    model: Model,
    prefix: list[RestartResult],
    groups,
    k: int,
    seed: int,
    t0: float,
    time_limited: bool,
) -> SolveOutcome:
    cands = _candidates(prefix)
    feasible = [c for c in cands if c[2]]
    pool = feasible if feasible else cands
    chosen = select_diverse(pool, k, groups) if pool else []
    layouts = []
    for rank, ci in enumerate(chosen):
        poses = pool[ci][0]
        res, ok, cost = model.report(poses)
        layouts.append(
            Layout(
                id=f"{problem.id}-{rank:02d}",
                placements=[
                    Placement(asset_id=t.id, pose=Pose2D(x=p[0], y=p[1], yaw=p[2]))
                    for t, p in zip(model.templates, poses, strict=True)
                ],
                cost=cost,
                feasible=ok,
                results=res,
            )
        )
    notes = list(model.notes)
    if not feasible:
        reasons, proven = explain_infeasibility(model, prefix)
        notes += reasons
        if not proven:
            notes.append(
                f"no static contradiction was found; searched {len(prefix)} restarts"
                + (
                    " until the time budget ran out; a larger budget may help"
                    if time_limited
                    else ""
                )
            )
    elif len(layouts) < k:
        notes.append(
            f"only {len(layouts)} mutually distinct feasible layouts found "
            f"(min layout distance {MIN_LAYOUT_DISTANCE} m)"
        )
    firsts = [r.first_feasible_at for r in prefix if r.first_feasible_at is not None]
    seconds = time.monotonic() - t0
    evaluations = sum(r.evaluations for r in prefix)
    diag = SolveDiagnostics(
        restarts=len(prefix),
        restarts_feasible=sum(bool(r.feasible) for r in prefix),
        time_to_first_feasible_s=(min(firsts) - t0) if firsts else None,
        feasible_layouts=sum(bool(lay.feasible) for lay in layouts),
        time_limited=time_limited,
        seconds=seconds,
        evaluations=evaluations,
    )
    result = SolveResult(
        problem_id=problem.id,
        layouts=layouts,
        stats=SolveStats(seed=seed, seconds=seconds, evaluations=evaluations),
        notes=notes,
    )
    return SolveOutcome(result, diag)


# ---------------------------------------------------------------------------- notes


def explain_infeasibility(model: Model, prefix: list[RestartResult]) -> tuple[list[str], bool]:
    """Human-readable reasons: cheap static checks + which hard constraints the best
    layout of each restart most often violates. Returns ``(notes, proven)`` where
    ``proven`` means a static check shows the problem cannot be solved."""
    notes = ["no feasible layout found"]
    proven = False
    room = model.room
    fp_area = sum(t.area for t in model.templates)
    ob_area = sum(ob.poly.area for ob in room.obstacles)
    free = room.area - ob_area
    if fp_area > free:
        proven = True
        notes.append(
            f"infeasible: furniture footprints total {fp_area:.2f} m², more than the free "
            f"floor area {free:.2f} m²"
        )
    else:
        need = sum(t.area + t.zone_area for t in model.templates)
        if need > 0.85 * free:
            notes.append(
                f"footprints plus clearance zones need about {need:.2f} m² of {free:.2f} m² "
                f"free floor ({100 * need / free:.0f}%): the room is very crowded"
            )
    x0, y0, x1, y1 = room.bounds
    for t in model.templates:
        w, d = t.xmax - t.xmin, t.ymax - t.ymin
        if not ((w <= x1 - x0 and d <= y1 - y0) or (d <= x1 - x0 and w <= y1 - y0)):
            proven = True
            notes.append(
                f"infeasible: asset {t.id!r} ({w:.2f} x {d:.2f} m) does not fit in the "
                "room's bounding box"
            )
    for c in model.problem.constraints:
        if isinstance(c, AgainstWall) and c.hard and c.wall_id:
            t = model.templates[model.index[c.a]]
            wall = room.wall_by_id[c.wall_id]
            spans = wall.blocked_spans(t.height, t.wall_mounted)
            longest = _longest_free_run(wall.length, spans)
            if longest < t.xmax - t.xmin:
                proven = True
                notes.append(
                    f"infeasible: {c.id}: {t.id!r} is {t.xmax - t.xmin:.2f} m wide but wall "
                    f"{c.wall_id!r} has at most {longest:.2f} m of usable wall between openings"
                )
    hard_near = {
        (c.a, c.b): c.max_distance
        for c in model.problem.constraints
        if isinstance(c, Near) and c.hard
    }
    for c in model.problem.constraints:
        if isinstance(c, DistanceAtLeast) and c.hard:
            for key in ((c.a, c.b), (c.b, c.a)):
                if key in hard_near and hard_near[key] < c.min_distance:
                    proven = True
                    notes.append(
                        f"infeasible: {c.id} requires {c.a}–{c.b} distance ≥ {c.min_distance} m "
                        f"but a hard near constraint requires ≤ {hard_near[key]} m"
                    )
    pinned = [model.initial[i] if model.pinned[i] else None for i in range(model.n)]
    if any(p is not None for p in pinned):
        st = State(model, pinned)
        if st.hard > 0:
            proven = True
            notes.append(
                "infeasible: pinned (fixed / immovable) assets already violate hard rules at their "
                "initial poses: " + "; ".join(sorted(set(st.hard_terms()))[:5])
            )
    if prefix:
        counts: Counter[str] = Counter()
        rule_counts: Counter[str] = Counter()
        for r in prefix:
            counts.update(set(r.violated))
            rule_counts.update({v.split(" ")[0] for v in r.violated})
        n = len(prefix)
        top_rules = ", ".join(
            f"{cid} ({100 * c / n:.0f}%)" for cid, c in rule_counts.most_common(4)
        )
        notes.append(
            f"hard constraints most often violated by the best layout of each of "
            f"{n} restarts: {top_rules}"
        )
        top_terms = "; ".join(f"{lab} ({100 * c / n:.0f}%)" for lab, c in counts.most_common(5))
        notes.append(f"most frequent specific conflicts: {top_terms}")
    return notes, proven


def _longest_free_run(length: float, spans: list[tuple[float, float]]) -> float:
    pos, best = 0.0, 0.0
    for s0, s1 in sorted(spans):
        best = max(best, s0 - pos)
        pos = max(pos, s1)
    return max(best, length - pos)


def evaluate(problem: LayoutProblem, poses: dict[str, Pose2D] | None = None):
    """Evaluate a hand-specified layout. Assets missing from ``poses`` use ``initial``.
    Returns ``(results, feasible, cost)``."""
    model = Model(problem)
    merged = dict(problem.initial)
    merged.update(poses or {})
    missing = [t.id for t in model.templates if t.id not in merged]
    if missing:
        raise KeyError(f"no pose for assets {missing}")
    ps: list[Pose] = [(merged[t.id].x, merged[t.id].y, merged[t.id].yaw) for t in model.templates]
    return model.report(ps)
