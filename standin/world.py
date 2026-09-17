"""world - the move and pattern grammar every gridworld shares, and the contract.

Wired: WIRED (imported by `pacworld`, `programs`, and any world after them).
STANDALONE of any particular world: nothing here knows about ghosts, pellets or
mazes, and nothing here imports a world.

THIS IS THE SDK PIECE. Owner: *"it should be sdk-like so we can use the same
code in other worlds."* `verse.CubbyMan` was already the world-agnostic
explorer - its own docstring says subclasses re-skin it - but the grammar it
re-skins over was buried in `pacman.py`, so a second world would have had to
import its move vocabulary from a file named after the first one.

What is here is what does NOT change between worlds:

  MOVES / OPP        the six directions of a 3D grid and their opposites
  _assignments       every injective slot->direction assignment with no two
                     slots on opposite directions, because a back-and-forth is
                     not a move. Pure combinatorics over MOVES; it never knew
                     anything about pac-man
  pattern_name/_dna  how a combo is named and compared
  cross              how two combos breed
  pattern_cost       what a combo costs to run
  JUMP_COST /        the DEFAULT move economy. A world that prices movement
  MOVE_COST          differently overrides these; they live here so that
                     `programs` and `pacworld` can both read them without
                     either importing the other

A WORLD, for the purposes of this SDK, is anything that answers `cell`,
`coords`, `exits`, `observe`, `try_move` and `all_facts` - the surface
`verse.ToyVerse` and `pacworld.PacVerse` both already present. It is written
here as a Protocol rather than a base class on purpose: `GhostVerse` inherits
`PacVerse` for real code reuse, while a world from somewhere else only has to
match the shape.

EVENTS are the other half of the contract, and the half that was implicit. The
affect machinery does not care what happened, only what KIND of thing it was:
something hurt him, something rewarded him, he finished, he ran out. Those four
are named here so a second world raises the same ones and `coach`, `afferent`
and `neurochem` work in it unchanged.
"""
from __future__ import annotations

from typing import Protocol

__wiring__ = "WIRED"


# the six moves, exactly as pacman_3d.MOVES (pacman_live imports these too)
MOVES = {"right": (1, 0, 0), "left": (-1, 0, 0), "up": (0, 1, 0),
         "down": (0, -1, 0), "forward": (0, 0, 1), "back": (0, 0, -1)}


OPP = {"right": "left", "left": "right", "up": "down", "down": "up",
       "forward": "back", "back": "forward"}


JUMP_COST = 20                                           # our energy price for a hop (theirs is unrecorded here)


# and energy only comes back by RESTING in a safe place — or a little from pellets)
MOVE_COST = 1                                            # a base step



def pattern_cost(pattern: str | None) -> int:
    """A combo costs twice its length (AAB -> 6, AAAAA -> 10); a jump 20."""
    return JUMP_COST if pattern is None else 2 * len(pattern)


# ── generative superpowers: patterns over move SLOTS ────────────────────────
# A pattern is a word over slots A/B/C ("AAA" = one direction three times,
# "AB" = two perpendicular directions, "ABC" = three axes). At use time the
# slots are instantiated with any assignment of distinct, non-opposite
# directions, so one program covers a whole family of moves. The live game's
# DASH / BLINK / COMBO are three points in this space; the rest is his to
# find — combos that did not exist until he composed them.
_FLAVOR = {"AAA": "DASH", "AAAAA": "BLINK", "AB": "COMBO", "AAB": "KNIGHT", "ABC": "WARP",
           "AAAA": "SPRINT", "AABB": "ZIGZAG", "ABAB": "STAIRS", "ABA": "HOOK", "ABB": "ELBOW"}



def pattern_name(pattern: str) -> str:
    return _FLAVOR.get(pattern, f"COMBO-{pattern}")



# ── a pattern IS a shape DNA ────────────────────────────────────────────────
# Nick, 2026-09-15: *"combining combos + the DNA concept (square vs circle)
# could be the right thing there."* And it already is one — exp_r34's shape DNA
# is a set of role->filler pairs, and a pattern is exactly that: the slot
# positions are the roles, the letters are the fillers. "AAB" is a shape with
# three roles, two of which agree.
#
# So the properties of a combo are readable off its DNA without running it,
# which is the whole point of exp_r34's containment result (a small circle fits
# in a bigger square, from the DNA alone, having never seen a square):
#
#   LENGTH  how many steps it takes
#   AXES    how many distinct directions it commits to
#   RUN     the longest straight stretch -- what makes a move cover ground
#   TURNS   how many times it changes direction -- what makes it manoeuvre
#
# CROSSING two of them is the cirsquare: take the shared prefix (what the
# parents AGREE on -- bundling blends agreeing properties, exp_r34) and finish
# with the other parent's tail (where they differ -- bundling arbitrates). The
# child is a real shape neither parent was, and its properties are predictable
# from the parents' before the VM ever sees it.
def pattern_dna(pattern: str) -> dict:
    """The readable properties of a pattern, from the pattern alone."""
    runs, longest, turns = 1, 1, 0
    for a, b in zip(pattern, pattern[1:]):
        if a == b:
            runs += 1
            longest = max(longest, runs)
        else:
            runs, turns = 1, turns + 1
    return {"length": len(pattern), "axes": len(set(pattern)),
            "run": longest if pattern else 0, "turns": turns}



def cross(a: str, b: str) -> str | None:
    """A child of two patterns: what they agree on, then where they differ.

    Deterministic, and a child that is not a new legal shape is None — a
    cross that reproduces a parent, or that breaks the rules a pattern has to
    obey (2..5 slots, starts at A, at most three), is not a discovery."""
    if not a or not b:
        return None
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    child = a[:shared] + b[shared:]
    if child in (a, b) or not (2 <= len(child) <= 5) or not child.startswith("A"):
        return None
    if len(set(child)) > 3 or set(child) - set("ABC"):
        return None
    # the slots must be introduced in order (A before B before C) or the
    # pattern names a shape `_assignments` cannot instantiate
    seen: list[str] = []
    for ch in child:
        if ch not in seen:
            if ch != "ABC"[len(seen)]:
                return None
            seen.append(ch)
    return child



def _assignments(pattern: str):
    """Every injective slot -> direction assignment with no two slots on
    opposite directions (a back-and-forth is not a move)."""
    slots = []
    for ch in pattern:
        if ch not in slots:
            slots.append(ch)
    dirs = list(MOVES)

    def rec(i, chosen):
        if i == len(slots):
            yield dict(zip(slots, chosen))
            return
        for d in dirs:
            if d in chosen or OPP[d] in chosen:
                continue
            yield from rec(i + 1, chosen + [d])
    yield from rec(0, [])


# ── the contract, written down ──────────────────────────────────────────────

class World(Protocol):
    """What the explorer needs from a world, and nothing more.

    A Protocol rather than a base class, deliberately. `GhostVerse` inherits
    `PacVerse` because they share real code; a world written somewhere else
    shares nothing but this shape, and should not have to import a class from
    here to be usable. `verse.ToyVerse` and `pacworld.PacVerse` both already
    satisfy it — this is a description of what is, not a new requirement.

    The rule the whole architecture rests on: the world is the TERRITORY and
    the explorer's fact store is his MAP. `observe` is the only way across,
    everything he believes came through it, and the gap between the two is
    what the project is about. A world that lets him read its ground truth
    directly has deleted the experiment.
    """

    def cell(self, *coords) -> str:
        """Coordinates -> the place name he will use for them."""

    def coords(self, place: str) -> tuple:
        """A place name -> its coordinates."""

    def exits(self, place: str) -> dict:
        """place -> {move name: where it leads}. What is PHYSICALLY possible
        here, before he knows any of it."""

    def observe(self, place: str) -> list:
        """What is perceptible from here, in his fact language. The ONLY
        channel from territory to map."""

    def try_move(self, place: str, move: str, offered: dict) -> tuple:
        """Attempt a move -> (where he ends up, what he learned). A refused
        move is not an error: being stopped by a wall is how he finds out
        there is one."""

    def all_facts(self) -> list:
        """Ground truth, for SCORING a run from outside. Never for him."""


# The four kinds of thing that can happen to a body, named once so a second
# world raises the same ones and `coach`, `afferent` and `neurochem` work in it
# without being told where they are.
#
# The affect machinery never cares WHAT happened — a ghost, a fall, a deadline
# — only which of these it was. `pacworld` calls them by way of
# `CubbyGhost._on_caught`, `next_level` and the out-of-time branch; a different
# world calls them from wherever its own stakes live.
HURT = "hurt"        # something was done to him: it lands, and it lingers
REWARD = "reward"    # a real hit, not mild pleasure — the only tolerance input
CLEAR = "clear"      # he finished the thing
FAIL = "fail"        # he did not, and nothing hurt him; deflation, not pain

EVENTS = (HURT, REWARD, CLEAR, FAIL)
