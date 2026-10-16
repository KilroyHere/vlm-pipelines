"""One simulated-annealing restart: greedy randomized construction + annealing.

Moves (per step, one movable asset is picked first):

- ``translate``: Gaussian step in x/y; sigma shrinks from 0.6 m to 0.03 m.
- ``slide``: Gaussian step along the asset's own left-right axis (keeps wall contact).
- ``rotate``: jump to another allowed yaw (multiples of 90 deg + wall-aligned yaws).
- ``jump``: resample a position anywhere in the room, or next to a related asset
  (``near``/``faces`` partner), with a random or partner-facing yaw.
- ``snap``: put the asset's back against a random wall, facing into the room.
- ``swap``: exchange the poses of two movable assets (wall-bound ones re-snap).
- ``back``: an asset with a soft ``fixed`` jumps back to its exact initial pose.

Group moves: with probability ``P_GROUP`` a non-swap move of an asset that has
*followers* (smaller assets tied to it by a tight hard ``near``, e.g. desk -> chair,
bed -> nightstands) applies the same rigid transform to the followers.
Focus: once past ``FOCUS_AFTER`` of the schedule, with probability ``P_FOCUS`` the moved
asset is drawn from those involved in a violated hard term.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .geometry import wrap_angle, yaw_facing
from .model import Model, Pose, State

T_START = 3.0
T_END = 1e-4
SIGMA_START = 0.6
SIGMA_END = 0.03
INIT_CANDIDATES = 24
WALL_GAP = 0.001  # snap leaves 1 mm between back edge and wall (robust to float noise)
P_GROUP = 0.5
P_BACK_TO_INITIAL = 0.1  # soft-`fixed` assets: jump straight back to the initial pose
P_FOCUS = 0.5
FOCUS_AFTER = 0.3


def default_iterations(model: Model) -> int:
    return 300 + 250 * len(model.movable)


@dataclass
class RestartResult:
    index: int
    poses: list[Pose]
    cost: float
    feasible: bool
    evaluations: int
    first_feasible_at: float | None  # time.monotonic() when this restart first hit feasible
    truncated: bool  # stopped by the deadline
    extra: list[tuple[list[Pose], float]] = field(default_factory=list)  # more feasible states
    violated: list[str] = field(default_factory=list)  # hard terms violated in `poses`


class Annealer:
    def __init__(self, model: Model, rng: random.Random):
        self.m = model
        self.rng = rng
        self.room = model.room
        self.wall_lengths = [w.length for w in model.room.walls]

    # ------------------------------------------------------------------ pose samplers

    def _walls_for(self, i: int) -> list[int]:
        allowed = self.m.wall_walls[i]
        return list(range(len(self.room.walls))) if allowed is None else allowed

    def wall_pose(self, i: int, wall: int | None = None, s: float | None = None) -> Pose:
        t = self.m.templates[i]
        width = t.xmax - t.xmin
        if wall is None:
            cands = self._walls_for(i)
            fits = [k for k in cands if self.room.walls[k].length >= width] or cands
            weights = [self.room.walls[k].length for k in fits]
            wall = self.rng.choices(fits, weights)[0]
        w = self.room.walls[wall]
        lo, hi = -t.xmin, w.length - t.xmax
        if s is None:
            s = self.rng.uniform(lo, hi) if hi >= lo else w.length / 2
        else:
            s = min(max(s, lo), hi) if hi >= lo else w.length / 2
        x, y = w.point(s, -t.ymin + WALL_GAP)
        return (x, y, w.yaw_in)

    def resnap(self, i: int, pose: Pose) -> Pose:
        """Move a wall-bound asset onto the nearest allowed wall, keeping its position
        along that wall."""
        best, best_d = None, math.inf
        for k in self._walls_for(i):
            w = self.room.walls[k]
            t, d = w.to_local(pose[0], pose[1])
            dd = abs(d) + max(0.0, -t) + max(0.0, t - w.length)
            if dd < best_d:
                best, best_d = k, dd
        assert best is not None
        s, _ = self.room.walls[best].to_local(pose[0], pose[1])
        return self.wall_pose(i, best, s)

    def free_pose(self, i: int) -> Pose:
        x, y = self.room.random_point(self.rng)
        return (x, y, self.rng.choice(self.room.yaws))

    def partner_pose(self, i: int, j: int, pj: Pose) -> Pose:
        ti, tj = self.m.templates[i], self.m.templates[j]
        r = self.rng.uniform(0.3, 1.0) * (ti.reach + tj.reach) + self.rng.uniform(0.0, 0.5)
        a = self.rng.uniform(-math.pi, math.pi)
        x, y = pj[0] + r * math.cos(a), pj[1] + r * math.sin(a)
        if j in self.m.faces_targets[i]:
            yaw = self.room.snap_yaw(yaw_facing(pj[0] - x, pj[1] - y))
        else:
            yaw = self.rng.choice(self.room.yaws)
        return (x, y, yaw)

    def sample_pose(self, i: int, poses: list[Pose | None]) -> Pose:
        if self.m.wall_bound[i]:
            return self.wall_pose(i)
        u = self.rng.random()
        placed = [j for j in self.m.partners[i] if poses[j] is not None]
        if placed and u < 0.45:
            j = self.rng.choice(placed)
            return self.partner_pose(i, j, poses[j])  # type: ignore[arg-type]
        if u < (0.75 if self.m.wall_soft[i] else 0.6):
            return self.wall_pose(i)
        return self.free_pose(i)

    # ------------------------------------------------------------------ construction

    def construct(self, use_initial: bool) -> State:
        m = self.m
        state = State(m, [m.initial[i] if m.pinned[i] else None for i in range(m.n)])
        rest = [i for i in m.movable]
        if use_initial:
            hinted = {i: m.initial[i] for i in rest if m.initial[i] is not None}
            if hinted:
                state.commit(state.propose(hinted))  # type: ignore[arg-type]
                rest = [i for i in rest if i not in hinted]
        # wall-bound first, then by decreasing footprint+zone area; light shuffling of ties
        rest.sort(
            key=lambda i: (
                not m.wall_bound[i],
                -(m.templates[i].area + m.templates[i].zone_area) * self.rng.uniform(0.85, 1.15),
            )
        )
        for i in rest:
            best = None
            for _ in range(INIT_CANDIDATES):
                p = state.propose({i: self.sample_pose(i, state.poses)})
                if best is None or p.cost < best.cost:
                    best = p
            assert best is not None
            state.commit(best)
        return state

    # ------------------------------------------------------------------ moves

    def propose_move(self, state: State, sigma: float, focus: bool = False) -> dict[int, Pose]:
        m, rng = self.m, self.rng
        i = -1
        if focus and rng.random() < P_FOCUS:
            bad = [j for j in state.violators() if not m.pinned[j]]
            if bad:
                i = rng.choice(bad)
        if i < 0:
            i = rng.choice(m.movable)
        move = self._move(i, state, sigma)
        if len(move) == 1 and m.followers[i] and rng.random() < P_GROUP:
            old, new = state.poses[i], move[i]
            assert old is not None
            dyaw = new[2] - old[2]
            c, s = math.cos(dyaw), math.sin(dyaw)
            for j in m.followers[i]:
                pj = state.poses[j]
                assert pj is not None
                rx, ry = pj[0] - old[0], pj[1] - old[1]
                move[j] = (
                    new[0] + c * rx - s * ry,
                    new[1] + s * rx + c * ry,
                    self.room.snap_yaw(pj[2] + dyaw),
                )
        return move

    def _move(self, i: int, state: State, sigma: float) -> dict[int, Pose]:
        m, rng = self.m, self.rng
        x, y, yaw = state.poses[i]  # type: ignore[misc]
        if m.prefers_initial[i] and rng.random() < P_BACK_TO_INITIAL:
            return {i: m.initial[i]}  # type: ignore[dict-item]
        u = rng.random()
        if m.wall_bound[i]:
            if u < 0.45:
                return {i: self._slide(i, x, y, yaw, sigma)}
            if u < 0.75:
                return {i: self.wall_pose(i)}
            if u < 0.85:
                return {i: (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), yaw)}
            return self._swap(i, state)
        if u < 0.40:
            return {i: (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), yaw)}
        if u < 0.52:
            return {i: self._slide(i, x, y, yaw, sigma)}
        if u < 0.67:
            others = [yw for yw in self.room.yaws if abs(wrap_angle(yw - yaw)) > 1e-6]
            return {i: (x, y, rng.choice(others))} if others else {}
        if u < 0.80:
            placed = m.partners[i]
            if placed and rng.random() < 0.6:
                j = rng.choice(placed)
                return {i: self.partner_pose(i, j, state.poses[j])}  # type: ignore[arg-type]
            return {i: self.free_pose(i)}
        if u < 0.90:
            return {i: self.wall_pose(i)}
        return self._swap(i, state)

    def _slide(self, i: int, x: float, y: float, yaw: float, sigma: float) -> Pose:
        d = self.rng.gauss(0, 1.5 * sigma)
        return (x + d * math.cos(yaw), y + d * math.sin(yaw), yaw)

    def _swap(self, i: int, state: State) -> dict[int, Pose]:
        others = [j for j in self.m.movable if j != i]
        if not others:
            return {}
        j = self.rng.choice(others)
        pi, pj = state.poses[i], state.poses[j]
        assert pi is not None and pj is not None
        ni, nj = pj, pi
        if self.m.wall_bound[i]:
            ni = self.resnap(i, ni)
        if self.m.wall_bound[j]:
            nj = self.resnap(j, nj)
        return {i: ni, j: nj}

    # ------------------------------------------------------------------ annealing

    def run(self, index: int, iterations: int, deadline: float, use_initial: bool) -> RestartResult:
        m, rng = self.m, self.rng
        state = self.construct(use_initial)
        first_feasible = time.monotonic() if state.feasible else None
        best_poses, best_key = list(state.poses), (not state.feasible, state.cost)
        extra: list[tuple[list[Pose], float]] = []
        truncated = False
        if m.movable:
            log_ratio = math.log(T_END / T_START)
            for it in range(iterations):
                if it % 32 == 0 and time.monotonic() > deadline:
                    truncated = True
                    break
                frac = it / iterations
                temp = T_START * math.exp(log_ratio * frac)
                sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * frac
                changes = self.propose_move(
                    state, sigma, focus=frac > FOCUS_AFTER and not state.feasible
                )
                if not changes:
                    continue
                prop = state.propose(changes)
                delta = prop.cost - state.cost
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    was_feasible = state.feasible
                    state.commit(prop)
                    if state.feasible:
                        if first_feasible is None:
                            first_feasible = time.monotonic()
                        if not was_feasible and frac > 0.5:
                            extra.append((list(state.poses), state.cost))  # type: ignore[arg-type]
                    key = (not state.feasible, state.cost)
                    if key < best_key:
                        best_key, best_poses = key, list(state.poses)
        final = State(m, best_poses)  # exact re-evaluation of the best state
        violated = [] if final.feasible else final.hard_terms()
        return RestartResult(
            index=index,
            poses=best_poses,  # type: ignore[arg-type]
            cost=final.cost,
            feasible=bool(final.feasible),
            evaluations=state.evaluations,
            first_feasible_at=first_feasible,
            truncated=truncated,
            extra=extra[-3:],
            violated=violated,
        )


def run_restart(
    model: Model, seed: int, index: int, iterations: int, deadline: float, use_initial: bool
) -> RestartResult:
    rng = random.Random(seed * 1_000_003 + index)
    return Annealer(model, rng).run(index, iterations, deadline, use_initial)
