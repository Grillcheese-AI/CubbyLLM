"""goemotions_cube — where each GoEmotions label sits in OUR cube.

Wired: WIRED (imported by the feel-family builder; validated by
`standin/tests/test_goemotions_cube.py` and by running this file).

DERIVED, NOT ASSERTED. The coordinates here are computed from two tables the
repo already relies on and that a human already checked:

  * `build_chat_sft.GOEMOTIONS_PETAL` — all 28 GoEmotions labels onto the eight
    Plutchik petals. In the tree since the emotion-recognition family.
  * `pacman._PETAL` — our Lövheim corners onto those same petals, with three
    intensity tiers each. What the compass already draws.

so the only new thing is the arithmetic between them. That matters because the
alternative — a hand-written `label -> [S, DA, NE]` table — is a third opinion
about the same space, and this project has now been bitten twice by two
opinions about one space disagreeing in the band between them. A coordinate
nobody can derive is a coordinate nobody can check.

The check is the point: `validate()` runs every derived coordinate back through
the SAME nearest-corner classifier the agent uses and requires it to land on
the corner its petal implies. A mapping that cannot survive its own classifier
is caught here, before it reaches 58k rows of corpus where the damage is
invisible.

Run it:  python standin/data/goemotions_cube.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE, os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_chat_sft import GOEMOTIONS, GOEMOTIONS_PETAL  # noqa: E402
from neurochem import Neurochemistry  # noqa: E402
from pacman import _PETAL  # noqa: E402

__wiring__ = "WIRED"

CENTRE = (0.5, 0.5, 0.5)

# WHAT THIS MAPPING CANNOT DO, stated up front because a limitation nobody
# wrote down becomes a bug somebody else finds. A label's position is its
# petal times its tier, which is 8 x 3 = 24 distinguishable points for 28
# labels — so by construction it CANNOT separate them all, and it does not
# pretend to. `disappointment`, `embarrassment`, `remorse` and `sadness` share
# a point; so do `excitement`, `love` and `pride`. That is exact for what this
# is used for — bucketing GoEmotions rows by CORNER, to fill the feel family's
# style bank — and it is wrong for anyone wanting a lookup that tells
# remorse from sadness. Several of the collisions are real Plutchik dyads
# (love = joy+trust, remorse = sadness+disgust) and could be separated by
# pulling toward the second petal's corner; that is a further mapping with its
# own validation to pass, not a line to slip in here.

# petal -> the corner that owns it, against the CORRECTED `_CORNERS`
# (2026-09-17, after only two of the eight turned out to sit where Lövheim puts
# them). Inverting `_PETAL` is still not a function — `spent` and `distress`
# both carry the sadness petal, because the cube separates depletion from
# anguish and Plutchik does not — so it is resolved explicitly rather than by
# dict-iteration order. "Whichever came last" is how the duplicate-key bug in
# the proposed recipe made ANGER unreachable.
PETAL_CORNER = {
    "joy": "joy",
    "fear": "fear",              # was `anxious`; that coordinate is Lövheim's fear
    "surprise": "surprise",
    "sadness": "distress",       # the aroused end; `spent` is the flat end
    "disgust": "disgust",        # was `contempt`, and was excluded as social
    "anger": "anger",
    "anticipation": "interest",  # (1,1,1) interest/excitement — the explorer's vertex
}

# ANTICIPATION NOW HAS A CORNER, and TRUST is the one that does not.
#
# This was backwards. The note here used to say anticipation had no vertex and
# `_classify_emotion` returned `curious` as a novelty OVERRIDE — true at the
# time, but only because JOY had been placed on Lövheim's interest/excitement
# corner. With (1,1,1) named `interest`, anticipation is reachable through the
# geometry and the override is deleted.
#
# Trust takes its place as the homeless petal, and for a principled reason
# rather than a bookkeeping one: trust is oxytocin, and oxytocin is not an axis
# of a cube built from serotonin, dopamine and noradrenaline. No amount of
# relabelling produces it. The GoEmotions labels that map to trust stay
# unplaced, and they come back when a world supplies a second agent.
NO_CORNER = "trust"

# how intense each label is, as the tier index its petal uses (0 mild, 1 the
# petal's own name, 2 the outer ring). Taken from where the word actually sits
# in `_PETAL`'s tier triple where it appears there, and otherwise judged once,
# here, in the open.
TIER = {
    "admiration": 2, "amusement": 1, "anger": 1, "annoyance": 0, "approval": 0,
    "caring": 1, "confusion": 0, "curiosity": 0, "desire": 1, "disappointment": 1,
    "disapproval": 1, "disgust": 1, "embarrassment": 1, "excitement": 2, "fear": 1,
    "gratitude": 1, "grief": 2, "joy": 1, "love": 2, "nervousness": 0,
    "optimism": 1, "pride": 2, "realization": 0, "relief": 1, "remorse": 1,
    "sadness": 1, "surprise": 1, "neutral": 0,
}

# how far out of the cube each tier sits. The centre is neutral and the vertex
# is the pure corner, so a tier is a fraction of that radius — the same reading
# of intensity `felt_names` uses when it picks a tier from a distance.
TIER_REACH = (0.45, 0.72, 0.95)


def label_coords(label: str) -> tuple[float, float, float] | None:
    """(5-HT, DA, NE) for a GoEmotions label, or None when it has no corner.

    Interpolates from the centre toward the corner its petal owns, by the
    label's intensity tier. `neutral` is the centre itself."""
    if label == "neutral":
        return CENTRE
    petal = GOEMOTIONS_PETAL.get(label)
    if petal in (None, NO_CORNER, "calm"):
        return None
    corner = PETAL_CORNER.get(petal)
    if corner is None:
        return None
    vertex = Neurochemistry._CORNERS[corner]
    reach = TIER_REACH[TIER.get(label, 1)]
    return tuple(round(c + (v - c) * reach, 3) for c, v in zip(CENTRE, vertex))


LABEL_CUBE = {lab: label_coords(lab) for lab in GOEMOTIONS}
PLACED = {k: v for k, v in LABEL_CUBE.items() if v is not None}
UNPLACED = sorted(k for k, v in LABEL_CUBE.items() if v is None)


def nearest_corner(vec) -> str:
    """The agent's own classifier, so validation tests the real thing."""
    return min(Neurochemistry._CORNERS.items(),
               key=lambda kv: sum((p - q) ** 2 for p, q in zip(vec, kv[1])) ** 0.5)[0]


def validate() -> list[str]:
    """Every placed label must land back on the corner its petal implies.

    Returns the problems, empty when the table is self-consistent. This is the
    check the proposed hand-written table could not have passed: its `anger`
    coordinate resolved to the anticipation corner, `neutral` — the largest
    class in GoEmotions — fell through to sadness, and twelve of twenty-eight
    labels disagreed with true nearest-corner."""
    problems = []
    for label, vec in sorted(PLACED.items()):
        if label == "neutral":
            continue                                     # the centre belongs to no corner
        want = PETAL_CORNER[GOEMOTIONS_PETAL[label]]
        got = nearest_corner(vec)
        if got != want:
            problems.append(f"{label}: {vec} lands on {got!r}, petal implies {want!r}")
    seen = {}
    for label, vec in PLACED.items():
        if label != "neutral":
            seen.setdefault(vec, []).append(label)
    for vec, labels in seen.items():
        if len(labels) > 1:
            # not an error: two labels of the same petal and tier ARE the same
            # point. Reported so it is a decision rather than a surprise.
            problems.append(f"note: {', '.join(sorted(labels))} share {vec}")
    return problems


def style_bank_key(label: str) -> str | None:
    """The CORNER a GoEmotions row should be filed under for the feel family's
    style bank. None for labels with no corner, which are simply not used."""
    vec = label_coords(label)
    return None if vec is None or label == "neutral" else nearest_corner(vec)


if __name__ == "__main__":
    print(f"{'label':<16}{'petal':<14}{'corner':<10}{'tier':>5}  coords")
    for lab in GOEMOTIONS:
        v = LABEL_CUBE[lab]
        petal = GOEMOTIONS_PETAL.get(lab, "-")
        corner = "(centre)" if lab == "neutral" else (nearest_corner(v) if v else "-")
        print(f"{lab:<16}{petal:<14}{corner:<10}{TIER.get(lab, 1):>5}  "
              f"{v if v else 'UNPLACED - no corner for this petal'}")
    print()
    print(f"placed {len(PLACED)} / {len(GOEMOTIONS)}   unplaced: {', '.join(UNPLACED) or 'none'}")
    problems = validate()
    print(f"\nvalidation: {'CLEAN' if not problems else f'{len(problems)} to look at'}")
    for p in problems:
        print("   ", p)
