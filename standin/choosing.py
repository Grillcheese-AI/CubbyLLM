"""choosing — the filter. Whatever the agent is picking between, the state bends it.

Wired: WIRED (any world's selection step). STANDALONE of every world: nothing
here knows what a ghost, a sentence or a file is.

Owner: *"we need to think globally not in terms of a specific world. It needs to
modulate whatever passes through that filter."*

THE GAP THIS CLOSES, and it was hiding in plain sight. `Neurochemistry.
modulation()` already returns seven world-agnostic knobs — creativity, caution,
warmth, energy, stability, focus, urgency — computed from the hormones and
carrying no feeling word and no world. It is consumed in exactly two places,
both in `pac_speech`, both to shape SENTENCES. The filter was built and wired
only to the mouth.

So affect changed what he SAID and never what he DID. `wariness` (a4229f7) was
me noticing that and fixing it the wrong way: one hormone hand-wired to one
world's quantity, noradrenaline to a maze berth. That is a patch per world, and
there is no such thing as a general mechanism you have to rebuild for every
world.

WHAT IS ACTUALLY UNIVERSAL is that every world offers CANDIDATES and something
picks among them. `CubbyMan._pick(exits)`, `candidate_moves`, the VM's ASK with
its offered replies, `CubbyChat.candidates` — all the same shape. That is the
chokepoint every decision passes through, so that is where a filter belongs.

A world describes each option in five terms it already knows, none of which
name anything about that world:

    value     how good it looks          a pellet, a correct answer, a fix
    risk      how badly it can go wrong  a ghost, a false claim, rm -rf
    novelty   how unfamiliar it is       an unseen cell, an untried approach
    cost      what it takes out of you   energy, tokens, time
    social    how much it involves someone else

The state supplies the gains. A maze scores a corridor; a chat scores a reply;
this module does not care which, and the same hormones produce the same BENDS
in both. That is what makes it a skill he carries rather than a trick he has in
one place.

TWO RULES THE FILTER MUST NOT BREAK, both learned here the hard way:

  IT MAY NOT INVERT THE WORLD'S JUDGEMENT. Risk always subtracts and value
  always adds; the state sets HOW MUCH, never the sign. No mood may make a
  lethal option attractive, because a body that can talk itself into the fire
  is not modulated, it is broken.

  NEUTRAL IS WHERE THE BODY ACTUALLY SITS, not 0.5. `modulation()` at rest does
  not return 0.5 across the board — it returns whatever the resting hormones
  imply — so gains are expressed against the QUIESCENT modulation, measured
  once from a body doing nothing. This is the same correction `quiescent()`
  made for the cube: measuring from a declared midpoint instead of the real one
  meant every reading started displaced, and a resting agent was permanently
  "feeling" something. A resting agent here bends nothing.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from neurochem import Neurochemistry  # noqa: E402

__wiring__ = "WIRED"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Option:
    """One thing the agent could do, in terms every world can supply.

    All five are 0..1 and all are the WORLD's judgement, not the body's. A
    world that cannot score a dimension leaves it at 0 and the filter simply
    has nothing to bend there — an agent with no notion of risk is not made
    reckless by this module, it is made indifferent, which is the honest
    reading of not knowing.
    """

    key: str
    value: float = 0.0
    risk: float = 0.0
    novelty: float = 0.0
    cost: float = 0.0
    social: float = 0.0
    meta: dict = field(default_factory=dict)      # whatever the world wants back


# How far a knob can bend its term, as a multiplier on the world's own number.
# A fully alarmed body triples the weight it puts on risk; it never reverses it.
SPAN = {"risk": 2.0, "novelty": 1.6, "cost": 1.6, "social": 1.6}
FLOOR = 0.2          # no term may be turned off, only turned down
BASE = {"value": 1.0, "risk": 1.0, "novelty": 0.35, "cost": 0.6, "social": 0.3}

# which knob drives which term
DRIVEN_BY = {"risk": "caution", "novelty": "creativity",
             "cost": "energy", "social": "warmth"}

_NEUTRAL: dict | None = None


def neutral() -> dict:
    """`modulation()` for a body at rest — the zero of this whole module.

    Computed once from a quiescent instance rather than assumed to be 0.5.
    The cube already taught this lesson: a declared midpoint that is not the
    real one means every reading starts displaced and a resting agent is
    permanently feeling something. Here it would mean a resting agent
    permanently biased — timid or reckless for no reason — which is worse,
    because it would look like a personality."""
    global _NEUTRAL
    if _NEUTRAL is None:
        probe = Neurochemistry()
        for _ in range(200):
            probe.update()
        _NEUTRAL = probe.modulation()
    return _NEUTRAL


def gains(mod: dict) -> dict:
    """The seven knobs -> the weight on each of the five option terms.

    A knob at its resting value gives exactly the base weight, so a body doing
    nothing weighs the world's numbers as the world gave them. `energy` is
    inverted on purpose: cost matters MORE when there is less left, which is
    why a tired person takes the short way round.
    """
    n = neutral()
    out = dict(BASE)
    for term, knob in DRIVEN_BY.items():
        dev = mod.get(knob, n.get(knob, 0.5)) - n.get(knob, 0.5)
        if term == "cost":
            dev = -dev                          # low energy -> cost weighs more
        # A FLOOR, so no term can be switched off. `energy` saturates at 1.0
        # on any real alarm while its resting value is 0.23, so the inverted
        # deviation drove the cost gain straight through zero and a stressed
        # agent became BLIND to effort rather than merely careless about it.
        # That is a saturation artefact wearing the costume of a finding. It
        # is the same rule as risk always subtracting: the state sets how
        # loudly one of the world's numbers speaks, never whether it speaks.
        out[term] = _clip(BASE[term] * (1.0 + SPAN[term] * dev),
                          BASE[term] * FLOOR, 4.0)
    return out


def weigh(options, mod: dict, gains_: dict | None = None) -> list:
    """-> [(Option, score)], best first. The world ranks; the body re-ranks.

    Value adds and risk subtracts, always. Nothing in here can make a lethal
    option look good — the state only decides how loudly the world's own
    numbers speak.
    """
    g = gains_ or gains(mod)
    out = []
    for o in options:
        score = (g["value"] * o.value
                 + g["novelty"] * o.novelty
                 + g["social"] * o.social
                 - g["risk"] * o.risk
                 - g["cost"] * o.cost)
        out.append((o, score))
    out.sort(key=lambda t: -t[1])
    return out


def breadth(n_options: int, mod: dict) -> int:
    """How many options get considered at all. Urgency narrows the field.

    The one effect here that is about the SHAPE of thinking rather than the
    weight of a term, and the one most clearly true of people: under pressure
    you do not weigh everything a little differently, you stop looking at most
    of it. Never below one — a body in a panic still does something."""
    n = neutral()
    over = _clip(mod.get("urgency", 0.0) - n.get("urgency", 0.0), 0.0, 1.0)
    keep = math.ceil(n_options * (1.0 - 0.6 * over))
    return max(1, min(n_options, keep))


def choose(options, mod: dict, previous: str | None = None):
    """The whole filter: narrow, re-weigh, and prefer to stay the course.

    `stability` is serotonin's contribution and it is about PERSISTENCE, not
    quality: a settled body sticks with what it was already doing, an unsettled
    one is quicker to switch. It is applied as a bonus to the option the agent
    chose last rather than as a weight on a term, because staying is not a
    property of an option, it is a property of a history.
    """
    if not options:
        return None, []
    g = gains(mod)
    ranked = weigh(options, mod, g)[:breadth(len(options), mod)]
    if previous is not None:
        n = neutral()
        stick = _clip(mod.get("stability", 0.0) - n.get("stability", 0.0), -1.0, 1.0) * 0.5
        ranked = sorted(((o, s + (stick if o.key == previous else 0.0))
                         for o, s in ranked), key=lambda t: -t[1])
    return ranked[0][0], ranked


def explain(mod: dict) -> dict:
    """What the body is currently doing to the choice, in numbers a caller can
    log or show. No feeling word, by the same rule as `manner()`."""
    n, g = neutral(), gains(mod)
    return {"gains": {k: round(v, 3) for k, v in g.items()},
            "narrowed_to": None,
            "deviation": {k: round(mod.get(k, 0.0) - n.get(k, 0.0), 3)
                          for k in ("caution", "creativity", "energy",
                                    "warmth", "stability", "urgency", "focus")}}
