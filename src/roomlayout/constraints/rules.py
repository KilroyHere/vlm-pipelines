"""Rule library (M3a): the relations people take for granted, as constraints.

``implied_constraints(problem)`` returns only the constraints to *add*: every one has
``origin="rule_library"``, a human-readable ``rationale`` and a deterministic id
(``rule:<type>:<subject>[:<object>]``). The full rule table, with weights and the
reasoning behind them, is in docs/constraints.md.

Design choices:

- **Soft by default.** Only the safety rules (walkway, doors) are hard. A rule that is
  impossible in a particular room must never make the whole problem infeasible.
- **Precedence (ADR-0006).** User/VLM constraints win: no implied relation is emitted on
  a subject the existing non-rule constraints already speak about (see ``_Precedence``).
- **Idempotent.** Ids are deterministic and existing ids are skipped, so running the
  library on its own output adds nothing.

Roles come from ``roomlayout.schema.taxonomy``: ``subtype`` first, then ``category``, then
words of the label and finally of the id (``"coffee table"``, ``"desk_chair"``), accepting
a word-derived subtype only if it belongs to the asset's category.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from roomlayout.schema import (
    AgainstWall,
    Asset,
    Constraint,
    DistanceAtLeast,
    Faces,
    Fixed,
    KeepClear,
    LayoutProblem,
    Mount,
    Near,
    OpeningType,
    Parallel,
    WalkwayWidth,
)
from roomlayout.schema.taxonomy import ROOMPLAN_CATEGORIES, resolve

ORIGIN = "rule_library"
# All soft weights below are multiplied by this. The solver weighs hard violations by 100
# per unit; relations weighted ~1 get settled only after the hard rules and can be frozen
# out in crowded rooms, so rule weights live on a ~5-30 scale (docs/constraints.md).
WEIGHT_SCALE = 5.0
# Soft weight of a user/VLM request: as strong as the strongest default relation
# (W_SEAT_TV), so a soft request is never outvoted by a rule of the library (ADR-0007
# amendment: at weight 1 an armchair's default pulled the user's coffee table away).
USER_SOFT_WEIGHT = 3.0 * WEIGHT_SCALE

# --- safety -------------------------------------------------------------------------
WALKWAY_WIDTH = 0.9  # m, preferred: common clear-circulation guidance (~36 in)
WALKWAY_MIN = 0.8  # m, hard minimum: ~32 in, the usual minimum clear width of a doorway
W_WALKWAY = 2.0  # soft weight of the 0.9 m preference
DOOR_CLEAR = 0.9  # m: a person standing in front of a door / passage
WINDOW_CLEAR = 0.4  # m: reach a window, keep tall furniture from covering it
WINDOW_WEIGHT = 0.5

# --- living -------------------------------------------------------------------------
W_SEAT_TV = 3.0  # mutual faces primary seating <-> TV
SEAT_TV_ANGLE = 20.0
TV_SEAT_ANGLE = 30.0  # TV screens are watchable off-axis
W_VIEW = 1.0  # viewing-distance band
W_COFFEE_FACES = 2.0
W_COFFEE_NEAR = 1.5
W_COFFEE_DIST = 1.0
W_COFFEE_PARALLEL = 0.5
COFFEE_NEAR = 0.5  # m, seat front to table: reachable from the seat
COFFEE_LEGROOM = 0.35  # m minimum knee/leg room when the seat declares no legroom zone
W_SECONDARY = 1.0
SECONDARY_NEAR = 1.0  # m: armchair within a conversation distance of the coffee table
W_SIDE_TABLE = 1.0
W_TV_STAND = 1.0
# --- work / dining / bedroom / storage ----------------------------------------------
W_DESK_CHAIR = 2.0
DESK_CHAIR_NEAR = 0.15
W_DESK_WALL = 0.5
W_INVITED = 1.5  # zone invitation (asset admitted by another asset's clearance zone)
W_OTTOMAN = 1.0
OTTOMAN_NEAR = 0.6  # m: within leg reach of the seat
ZONE_INVITE_NEAR = 0.3
W_DINING = 1.0
DINING_NEAR = 0.15
W_BED_WALL = 3.0
W_NIGHTSTAND = 1.5
W_NIGHTSTAND_WALL = 1.0
W_NIGHTSTAND_PARALLEL = 0.5
W_NIGHTSTAND_SIDES = 1.0
W_STORAGE_WALL = 2.0
WALL_STORAGE = ("bookshelf", "wardrobe", "dresser", "sideboard", "cabinet", "tv_stand")

SECONDARY_SEATING_SUBTYPES = ("armchair", "loveseat", "sectional")


# ============================================================================ roles


@dataclass(frozen=True)
class Role:
    category: str
    subtype: str | None


def _words_subtype(text: str, category: str) -> Role | None:
    tokens = re.findall(r"[a-z]+", text.lower())
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            r = resolve("_".join(tokens[i : i + n]))
            if r is None:
                continue
            cat, sub = r
            known = category in ROOMPLAN_CATEGORIES
            if (sub is not None and (cat == category or not known)) or (sub is None and not known):
                return Role(cat, sub)
    return None


def asset_role(asset: Asset) -> Role:
    """(category, subtype) of an asset via the shared taxonomy; see the module doc."""
    if asset.subtype:
        r = resolve(asset.subtype)
        if r is not None:
            return Role(*r)
    r = resolve(asset.category)
    role = Role(*r) if r is not None else Role(asset.category, None)
    if role.subtype is None:
        for text in (asset.label, asset.id):
            found = _words_subtype(text, role.category) if text else None
            if found is not None:
                return found
    return role


def _id_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+", s.lower()))


@dataclass
class Roles:
    """Who is who in the room, as used by the rules and the sensibility checks."""

    by_id: dict[str, Role]
    primary_seating: str | None = None
    secondary_seating: list[str] = field(default_factory=list)
    tv: str | None = None  # television, or the TV stand if there is no television
    tv_stand: str | None = None
    coffee_table: str | None = None
    side_tables: list[str] = field(default_factory=list)
    desk_pairs: list[tuple[str, str]] = field(default_factory=list)  # (desk, chair)
    dining: list[tuple[str, list[str]]] = field(default_factory=list)  # (table, chairs)
    beds: list[tuple[str, list[str]]] = field(default_factory=list)  # (bed, nightstands)
    wall_storage: list[str] = field(default_factory=list)
    desks: list[str] = field(default_factory=list)
    # (zone owner, zone name, member): a clearance zone that admits a category invites one
    # otherwise-unassigned asset of that category (a piano's bench area -> the bench).
    # zone name None: a piano bench paired with a piano-labelled asset (no zone).
    invited: list[tuple[str, str | None, str]] = field(default_factory=list)
    footrests: list[tuple[str, str]] = field(default_factory=list)  # (ottoman, seat)


def _area(a: Asset) -> float:
    return a.effective_footprint().area


def assign_roles(problem: LayoutProblem) -> Roles:
    assets = problem.assets
    roles = {a.id: asset_role(a) for a in assets}
    by = {a.id: a for a in assets}
    R = Roles(by_id=roles)

    def of(pred) -> list[str]:
        return [a.id for a in assets if pred(roles[a.id])]

    sofas = of(lambda r: r.category == "sofa")
    if sofas:
        R.primary_seating = max(sofas, key=lambda i: (_area(by[i]), -sofas.index(i)))
    R.secondary_seating = [
        i
        for i in of(lambda r: r.category == "sofa" or r.subtype in SECONDARY_SEATING_SUBTYPES)
        if i != R.primary_seating
    ]
    tvs = of(lambda r: r.category == "television")
    stands = of(lambda r: r.subtype == "tv_stand")
    R.tv_stand = stands[0] if stands else None
    if tvs:
        R.tv = max(tvs, key=lambda i: (by[i].dimensions.width, -tvs.index(i)))
    elif stands:
        R.tv = stands[0]
    coffee = of(lambda r: r.subtype == "coffee_table")
    R.coffee_table = max(coffee, key=lambda i: _area(by[i])) if coffee else None
    R.side_tables = of(lambda r: r.subtype == "side_table")

    desks = of(lambda r: r.subtype == "desk")
    R.desks = desks
    # tables that take a set of chairs (meeting tables behave like dining tables)
    dining_tables = of(lambda r: r.subtype in ("dining_table", "round_table", "meeting_table"))
    office = of(lambda r: r.subtype == "office_chair")
    dining_chairs = of(lambda r: r.subtype == "dining_chair")
    generic = of(lambda r: r.category == "chair" and r.subtype is None)
    # generic chairs: to desks that still lack one, then to dining tables
    pool = list(office)
    if len(pool) < len(desks) and not dining_tables:
        pool += generic
        generic = []
    elif len(pool) < len(desks):
        need = len(desks) - len(pool)
        pool += generic[:need]
        generic = generic[need:]
    for d in desks:  # pair by shared id tokens (desk_a <-> chair_a), then order
        if not pool:
            break
        c = max(pool, key=lambda c, d=d: (len(_id_tokens(c) & _id_tokens(d)), -pool.index(c)))
        pool.remove(c)
        R.desk_pairs.append((d, c))
    if dining_tables:
        chairs = dining_chairs + generic
        generic = []
        for k, t in enumerate(dining_tables):
            R.dining.append((t, chairs[k :: len(dining_tables)]))
    beds = of(lambda r: r.category == "bed")
    stands_n = of(lambda r: r.subtype == "nightstand")
    for k, b in enumerate(beds):
        R.beds.append((b, stands_n[k :: len(beds)] if beds else []))
    R.wall_storage = of(lambda r: r.subtype in WALL_STORAGE)
    # zone invitations: benches first (piano benches before plain ones), then leftover
    # generic chairs, then stools. Benches and stools never become desk or dining chairs.
    taken = {c for _, c in R.desk_pairs} | {c for _, cs in R.dining for c in cs}
    taken |= set(R.secondary_seating) | {R.primary_seating}
    piano_benches = of(lambda r: r.subtype == "piano_bench")
    free = piano_benches + of(lambda r: r.subtype == "bench") + generic
    free = [i for i in free + of(lambda r: r.subtype == "stool") if i not in taken]
    for owner in assets:
        if owner.id in desks or owner.id in dining_tables:
            continue  # desk chairs and dining chairs are handled above
        for z in owner.clearance_zones:
            for cat in z.allowed_categories:
                cand = [i for i in free if roles[i].category == cat and i != owner.id]
                if cand:
                    free.remove(cand[0])
                    R.invited.append((owner.id, z.name, cand[0]))
                    break
    # a piano bench without an inviting zone goes to a piano-labelled asset (piano is not
    # a taxonomy category, so this is matched on words)
    pianos = [
        a.id
        for a in assets
        if roles[a.id].category != "chair"
        and "piano" in _id_tokens(f"{a.category} {a.label} {a.id}")
    ]
    lone = [b for b in piano_benches if b in free]
    for k, b in enumerate(lone):
        if pianos:
            R.invited.append((pianos[k % len(pianos)], None, b))
    # ottomans are footrests for the main seating (else the first secondary seat)
    seat_for_ottoman = R.primary_seating or (R.secondary_seating or [None])[0]
    if seat_for_ottoman:
        R.footrests = [(o, seat_for_ottoman) for o in of(lambda r: r.subtype == "ottoman")]
    return R


# ============================================================================ precedence


class _Precedence:
    """What existing constraints already say (ADR-0006: user/VLM win over rules)."""

    def __init__(self, problem: LayoutProblem):
        cs = problem.constraints
        user = [c for c in cs if c.origin != ORIGIN]
        self.ids = {c.id for c in cs}
        self.faces_targets: dict[str, set[str]] = {}
        for c in user:
            if isinstance(c, Faces):
                self.faces_targets.setdefault(c.a, set()).add(c.b)
        # implied faces(a, b) that may coexist with user faces(a, t) for every t in the set:
        # pairs the library itself always emits together (see implied_constraints)
        self.compatible_faces: dict[tuple[str, str], set[str]] = {}
        self.distance_pairs = {
            frozenset((c.a, c.b)) for c in user if isinstance(c, Near | DistanceAtLeast)
        }
        self.parallel_pairs = {frozenset((c.a, c.b)) for c in user if isinstance(c, Parallel)}
        self.on_wall = {c.a for c in cs if isinstance(c, AgainstWall)}
        self.has_walkway = any(isinstance(c, WalkwayWidth) for c in cs)
        self.cleared = {c.opening_id for c in cs if isinstance(c, KeepClear)}
        fixed = {c.a for c in cs if isinstance(c, Fixed)}
        self.pinned = {a.id for a in problem.assets if not a.movable} | {
            c.a for c in cs if isinstance(c, Fixed) and c.hard
        }
        self.no_wall_rule = (
            fixed | self.pinned | {a.id for a in problem.assets if a.mount == Mount.wall}
        )

    def allows(self, c: Constraint) -> bool:
        if c.id in self.ids:
            return False  # idempotence: already present (e.g. from an earlier run)
        refs = [v for v in (getattr(c, "a", None), getattr(c, "b", None)) if v is not None]
        if len(refs) == 2 and refs[0] == refs[1]:
            return False
        if len(refs) == 2 and all(r in self.pinned for r in refs):
            return False  # nothing can move: the relation is a constant
        if isinstance(c, Faces):
            targets = self.faces_targets.get(c.a)
            if not targets:
                return True
            if c.b in targets:
                return False
            return targets <= self.compatible_faces.get((c.a, c.b), set())
        if isinstance(c, Near | DistanceAtLeast):
            return frozenset((c.a, c.b)) not in self.distance_pairs
        if isinstance(c, Parallel):
            return frozenset((c.a, c.b)) not in self.parallel_pairs
        if isinstance(c, AgainstWall):
            return c.a not in self.on_wall and c.a not in self.no_wall_rule
        if isinstance(c, WalkwayWidth):
            return not self.has_walkway
        if isinstance(c, KeepClear):
            return c.opening_id not in self.cleared
        return True


# ============================================================================ helpers


def _front_zone_depth(asset: Asset, blocked_category: str) -> float:
    """How far the asset's clearance zones that ``blocked_category`` may not enter reach in
    front of its footprint (0 if none): the legroom a coffee table must respect."""
    pts = asset.effective_footprint().points
    ymax = max(p[1] for p in pts)
    xmin, xmax = min(p[0] for p in pts), max(p[0] for p in pts)
    depth = 0.0
    for z in asset.clearance_zones:
        if blocked_category in z.allowed_categories:
            continue
        zx = [p[0] for p in z.polygon.points]
        zy = [p[1] for p in z.polygon.points]
        if min(zy) >= ymax - 1e-6 and min(zx) < xmax and max(zx) > xmin:
            depth = max(depth, max(zy) - ymax)
    return depth


def _zone_in_front(asset: Asset, zone_name: str) -> bool:
    ymax = max(p[1] for p in asset.effective_footprint().points)
    for z in asset.clearance_zones:
        if z.name == zone_name:
            return min(p[1] for p in z.polygon.points) >= ymax - 1e-6
    return False


def _diagonal(width: float) -> float:
    return width / math.cos(math.atan2(9, 16))  # 16:9 screen


# ============================================================================ rules


def implied_constraints(problem: LayoutProblem) -> list[Constraint]:
    """Constraints the rule library adds to ``problem`` (never the existing ones)."""
    prec = _Precedence(problem)
    roles = assign_roles(problem)
    by = {a.id: a for a in problem.assets}
    out: list[Constraint] = []
    seen: set[str] = set()

    def add(c: Constraint) -> None:
        if c.id not in seen and prec.allows(c):
            seen.add(c.id)
            out.append(c)

    def faces(a, b, w, why, angle=20.0):
        w *= WEIGHT_SCALE
        add(
            Faces(
                id=f"rule:faces:{a}:{b}",
                a=a,
                b=b,
                max_angle_deg=angle,
                weight=w,
                origin=ORIGIN,
                rationale=why,
            )
        )

    def near(a, b, d, w, why):
        w *= WEIGHT_SCALE
        add(
            Near(
                id=f"rule:near:{a}:{b}",
                a=a,
                b=b,
                max_distance=round(d, 3),
                weight=w,
                origin=ORIGIN,
                rationale=why,
            )
        )

    def apart(a, b, d, w, why):
        w *= WEIGHT_SCALE
        add(
            DistanceAtLeast(
                id=f"rule:distance_at_least:{a}:{b}",
                a=a,
                b=b,
                min_distance=round(d, 3),
                weight=w,
                origin=ORIGIN,
                rationale=why,
            )
        )

    def parallel(a, b, w, why):
        w *= WEIGHT_SCALE
        add(Parallel(id=f"rule:parallel:{a}:{b}", a=a, b=b, weight=w, origin=ORIGIN, rationale=why))

    def wall(a, w, why):
        w *= WEIGHT_SCALE
        add(AgainstWall(id=f"rule:against_wall:{a}", a=a, weight=w, origin=ORIGIN, rationale=why))

    # ---------------------------------------------------------------- living
    seat, tv, coffee = roles.primary_seating, roles.tv, roles.coffee_table
    if seat and tv and coffee:
        # "the sofa faces the TV" does not cancel "the coffee table is in front of the sofa":
        # the library emits both itself, and they agree when the table is between them
        prec.compatible_faces[(seat, coffee)] = {tv}
    if seat and tv:
        faces(seat, tv, W_SEAT_TV, "the main seating faces the TV", SEAT_TV_ANGLE)
        faces(tv, seat, W_SEAT_TV, "the screen is turned towards the main seating", TV_SEAT_ANGLE)
        diag = _diagonal(by[tv].dimensions.width)
        far = min(4.0, max(2.4, 2.5 * diag - 0.3))
        close = max(1.2, 1.0 * diag - 0.3)
        near(
            seat,
            tv,
            far,
            W_VIEW,
            f"comfortable viewing distance for a ~{diag / 0.0254:.0f} in screen "
            f"(at most ~2.5x the diagonal)",
        )
        apart(seat, tv, close, W_VIEW, "not too close to the screen (~1x the diagonal)")
    if tv and roles.tv_stand and roles.tv_stand != tv:
        near(roles.tv_stand, tv, 0.05, W_TV_STAND, "the TV stands on / hangs above its stand")
    if seat and coffee:
        legroom = _front_zone_depth(by[seat], by[coffee].category)
        faces(seat, coffee, W_COFFEE_FACES, "the coffee table sits in front of the seating")
        near(
            coffee,
            seat,
            max(COFFEE_NEAR, legroom + 0.15),
            W_COFFEE_NEAR,
            "within reach from the seat",
        )
        apart(
            coffee,
            seat,
            max(COFFEE_LEGROOM, legroom),
            W_COFFEE_DIST,
            "leaves legroom in front of the seat"
            + (f" (the seat's legroom zone is {legroom:.2f} m deep)" if legroom else ""),
        )
        parallel(coffee, seat, W_COFFEE_PARALLEL, "coffee table squared to the seating")
    for s in roles.secondary_seating:
        if coffee:
            faces(s, coffee, W_SECONDARY, "secondary seating joins the coffee-table group", 30)
            near(s, coffee, SECONDARY_NEAR, 0.5, "close enough to the coffee table to use it")
        elif tv:
            faces(s, tv, W_SECONDARY * 0.5, "secondary seating can see the TV", 45)
        elif seat:
            faces(s, seat, W_SECONDARY * 0.5, "conversation: faces the main seating", 45)
    seats = roles.secondary_seating + ([seat] if seat else [])
    for k, t in enumerate(roles.side_tables):
        if seats:
            s = seats[k % len(seats)]
            near(t, s, 0.1, W_SIDE_TABLE, "a side table sits beside a seat")
    if tv and roles.by_id[tv].category == "television" and by[tv].mount != Mount.wall:
        wall(tv, W_STORAGE_WALL, "a free-standing TV has its back to a wall")

    # ---------------------------------------------------------------- work
    for d, ch in roles.desk_pairs:
        near(ch, d, DESK_CHAIR_NEAR, W_DESK_CHAIR, "the desk chair is pulled up to the desk")
        faces(ch, d, W_DESK_CHAIR, "the desk chair faces the desk")
        faces(d, ch, W_DESK_CHAIR, "the chair is at the desk's front, not its end")
    for d in roles.desks:
        wall(d, W_DESK_WALL, "desks usually stand against a wall (weak preference)")
    for owner, zone, member in roles.invited:
        if zone is None:
            why = f"the {roles.by_id[member].subtype} sits at the {owner}'s front"
        else:
            why = f"{member} belongs in {owner}'s '{zone}' zone, which admits it"
        near(member, owner, ZONE_INVITE_NEAR, W_INVITED, why)
        faces(member, owner, W_INVITED, why)
        if zone is None or _zone_in_front(by[owner], zone):
            faces(owner, member, W_INVITED, f"{why} (in front of {owner})")
    for o, seat in roles.footrests:
        near(o, seat, OTTOMAN_NEAR, W_OTTOMAN, f"an ottoman is a footrest/extra seat by {seat}")

    # ---------------------------------------------------------------- dining
    for t, chairs in roles.dining:
        for ch in chairs:
            near(ch, t, DINING_NEAR, W_DINING, "chairs of a table set are at the table")
            faces(ch, t, W_DINING, "chairs of a table set face the table")

    # ---------------------------------------------------------------- bedroom
    for b, stands in roles.beds:
        wall(b, W_BED_WALL, "the headboard is against a wall")
        for ns in stands:
            near(ns, b, 0.1, W_NIGHTSTAND, "a nightstand stands right next to the bed")
            parallel(ns, b, W_NIGHTSTAND_PARALLEL, "nightstands are squared to the bed")
            wall(ns, W_NIGHTSTAND_WALL, "nightstands stand against the headboard wall")
        if len(stands) == 2:
            width = by[b].dimensions.width
            apart(
                stands[0],
                stands[1],
                0.8 * width,
                W_NIGHTSTAND_SIDES,
                "one nightstand on each side of the bed (they are at least ~a bed width apart)",
            )

    # ---------------------------------------------------------------- storage
    for s in roles.wall_storage:
        if roles.by_id[s].subtype == "nightstand":
            continue
        wall(s, W_STORAGE_WALL, f"a {roles.by_id[s].subtype} stands with its back to a wall")

    # ---------------------------------------------------------------- safety
    if not prec.has_walkway:  # both or neither: they are one rule
        add(
            WalkwayWidth(
                id="rule:walkway_width",
                min_width=WALKWAY_MIN,
                hard=True,
                origin=ORIGIN,
                rationale="every door and every piece of furniture stays reachable "
                f"along a path at least {WALKWAY_MIN} m wide",
            )
        )
        add(
            WalkwayWidth(
                id="rule:walkway_width:preferred",
                min_width=WALKWAY_WIDTH,
                weight=W_WALKWAY,
                origin=ORIGIN,
                rationale=f"preferably {WALKWAY_WIDTH} m wide, as in most circulation guidance",
            )
        )
    for o in problem.room.openings:
        if o.type in (OpeningType.door, OpeningType.opening):
            add(
                KeepClear(
                    id=f"rule:keep_clear:{o.id}",
                    opening_id=o.id,
                    distance=DOOR_CLEAR,
                    hard=True,
                    origin=ORIGIN,
                    rationale=f"nothing stands in the first {DOOR_CLEAR} m inside a {o.type.value}",
                )
            )
        else:
            add(
                KeepClear(
                    id=f"rule:keep_clear:{o.id}",
                    opening_id=o.id,
                    distance=WINDOW_CLEAR,
                    weight=WINDOW_WEIGHT,
                    origin=ORIGIN,
                    rationale="tall furniture does not cover the window",
                )
            )
    return out


__all__ = ["Role", "Roles", "asset_role", "assign_roles", "implied_constraints"]
