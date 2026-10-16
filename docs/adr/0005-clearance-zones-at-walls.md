# ADR-0005: Clearance zones may pass through a wall only behind a flush back

- Status: accepted
- Date: 2026-10-16

## Context
Category-default clearance zones (WS-B) are swept around footprints and can poke
through a wall when an asset stands against it. WS-B proposed clipping every zone
to the floor polygon. WS-A found that literal clipping lets the solver make
unusable layouts: a wardrobe facing a wall 10 cm away, a bed with one access side
in a corner, a dining table pushed into a wall over its seating zone.

## Decision
A clearance zone must lie inside the room, with one exemption: the part of a zone
that is behind the asset's back edge is ignored when that back is flush against a
wall. Everything in front of or beside the asset must stay in the room.
(`ZONE_WALL_POLICY` in `roomlayout.solver.geometry`; `"clip"` restores the literal
reading if we ever need it.)

## Consequences
- Category defaults can be generous without blocking against-wall placements.
- Usability-critical zones (seating, doors, drawers) can't be silently discarded.
