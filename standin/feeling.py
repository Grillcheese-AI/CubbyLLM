"""feeling - Plutchik names for a point in the cube. Layer 3, and portable.

Wired: WIRED (the pac agent's affect readout). STANDALONE of any world.

Owner, on the affect stack: *"do not focus on maze explorer as cubby will
always have its emotions following it everywhere. So its a global thing /
portable skill."* This is the part that was still living in the maze file.

`neurochem` decides WHERE he is (the corner, the intensity, the runner-up and
the margin). This decides what that place can honestly be CALLED: the petal for
the corner, the tier for the intensity, and a dyad when the margin is small
enough that he is genuinely between two. No world appears anywhere in it.
"""
from __future__ import annotations

import json

from pacpaths import PLUTCHIK_JSON  # noqa: E402
__wiring__ = "WIRED"



# our Lövheim readout -> the live game's Plutchik compass (petal, angle, petal
# color, the three intensity tiers whose color/message come from plutchik.json)
# Keyed on the corrected `neurochem._CORNERS` (2026-09-17), whose names are
# monoamine syndromes rather than petals. The mapping is MANY-TO-ONE on purpose:
# the cube distinguishes depletion from anguish and Plutchik does not, so
# `spent` and `distress` share the sadness petal rather than one of them being
# forced somewhere it does not belong.
#
# TRUST HAS NO ROW. It is oxytocin, not one of the three axes, so no corner can
# produce it — which is the honest answer to why `warm` used to sit on the trust
# petal: that coordinate is Lövheim's enjoyment/joy, and calling it trust was
# the mismatch in miniature. Trust returns when a world supplies a second agent.
#
# `interest` -> anticipation is the fix that retires the novelty override: the
# explorer's vertex now has the explorer's petal.
_PETAL = {"joy": ("joy", 0, "#ffca05", ("serenity", "joy", "ecstasy")),
          "fear": ("fear", 90, "#00a551", ("apprehension", "fear", "terror")),
          "surprise": ("surprise", 135, "#0099cd", ("distraction", "surprise", "amazement")),
          "distress": ("sadness", 180, "#2983c5", ("pensiveness", "sadness", "grief")),
          "spent": ("sadness", 180, "#2983c5", ("pensiveness", "sadness", "grief")),
          "disgust": ("disgust", 225, "#8973b3", ("boredom", "disgust", "loathing")),
          "anger": ("anger", 270, "#f05b61", ("annoyance", "anger", "rage")),
          "interest": ("anticipation", 315, "#f6923d", ("interest", "anticipation", "vigilance"))}


# EMPTY, and that is the finding. This set used to hold `contempt` and `shame`,
# muted because a maze has nobody to feel them about. The reasoning was right
# and the coordinates were wrong: those vertices are Lövheim's FEAR and DISGUST,
# which a solitary agent among ghosts has every reason to reach — so the mute was
# silencing the fear corner, and that is a large part of Nick's *"he never feels
# anxiety when he fails a level"*. Nick's other note still holds and is now served
# properly: *"sick of it should be neutral"* two steps into a fresh level is the
# `intensity < 0.15` centre test, not a corner exclusion.
#
# No corner of this cube is social. The social petal is trust, it has no corner,
# and it comes back as a WORLD CAPABILITY when a world declares other agents —
# not as a hard-coded exclusion here.
_SOCIAL_CORNERS: frozenset[str] = frozenset()


# How a person would actually SAY the feeling, first person and plain, instead
# of the taxonomy noun. Nick, 2026-09-15: *"instead of 'I feel acceptance' can
# we tell it to say 'it feels right', which is what someone would really
# say."* Handing the model a clinical label gets a clinical sentence back —
# "I feel acceptance" is not a thing anyone says — so the percept record
# carries the felt phrasing and the model has language to work with.
#
# This is a lexicon for internal states, not a script for events: one phrase
# per compass tier, no variants, no per-event copy. Where cubbyverse's
# plutchik.json already has a usable `sensations` value it is the fallback
# (see `_felt`), so this only names the tiers whose data reads oddly in the
# first person.
_FELT = {"acceptance": "it feels right", "trust": "this feels safe",
         "admiration": "I'm glad of this", "serenity": "I'm settled",
         "joy": "this is going well", "ecstasy": "this is better than I hoped",
         "interest": "something here is worth a look", "anticipation": "something is coming",
         "vigilance": "I'm watching closely", "distraction": "I don't know what matters here",
         "surprise": "that was not what I expected", "amazement": "I did not see that coming",
         "apprehension": "I can't settle", "fear": "something I care about is at risk",
         "terror": "I need to get away", "pensiveness": "I'm low", "sadness": "something is lost",
         "grief": "I've lost something that mattered", "boredom": "there's nothing here for me",
         "disgust": "something is wrong here", "loathing": "this is badly wrong",
         "annoyance": "something is in my way", "anger": "something is blocking me",
         "rage": "I'm blocked from something I need"}



def felt(tier: str) -> str:
    """The first-person phrasing for a compass tier: `_FELT` first, then
    plutchik.json's own `sensations` value, then the tier's name."""
    if tier in _FELT:
        return _FELT[tier]
    sens = (_plutchik().get(tier) or {}).get("sensations")
    return f"it feels {sens.lower()}" if sens else tier


_PLUTCHIK: dict | None = None



def _plutchik() -> dict:
    """cubbyverse's Plutchik metadata (tier color + felt message), loaded by
    file path — data, not a package import. {} when the checkout is absent."""
    global _PLUTCHIK
    if _PLUTCHIK is None:
        try:
            _PLUTCHIK = json.loads(PLUTCHIK_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _PLUTCHIK = {}
    return _PLUTCHIK



# ── Layer 3: what that place in the cube can honestly be CALLED ─────────────
#
# Layer 1 (somatic) and Layer 2 (geometry) live in neurochem.py and carry no
# names. This is the naming, and it is the game's because the vocabulary is —
# Plutchik's petals are what the compass already draws.
#
# Two things the corner alone does not say, and both are already in the repo:
#
#   INTENSITY.  A corner is an octant, not a point. `_PETAL` has carried three
#               tiers per petal all along (annoyance / anger / rage), and how
#               far out toward the corner he actually is picks which one. The
#               same octant is annoyance near the middle and rage at the edge.
#   THE EDGE.   A point sitting between two corners should not be described as
#               squarely in either. Plutchik's name for an adjacent pair is a
#               DYAD, and naming the pair is the honest form of "he is leaning
#               toward the next one" — unlike returning the neighbour's
#               primary, which would put him in a corner the classifier did not
#               pick and reopen the very contradiction this split closed.
#
# `similar-words` comes from plutchik.json: human-authored alternatives for the
# tier, which is a better fork than any list invented here.
#
# Four of Plutchik's eight primary dyads are SOCIAL — love, submission,
# contempt and remorse all need somebody else to be aimed at — so they are
# omitted rather than mapped, the same call `_SOCIAL_CORNERS` makes about the
# corners. Left in, they fire on their geometry alone: eating a ghost put joy
# next to trust and offered him *"love"*, which is a correct dyad and a
# meaningless thing to feel about a ghost.
_DYADS = {frozenset(("fear", "surprise")): "awe",
          frozenset(("surprise", "sadness")): "disapproval",
          frozenset(("anger", "anticipation")): "aggressiveness",
          frozenset(("anticipation", "joy")): "optimism"}


DYAD_MARGIN = 0.28          # nearer than this to the runner-up and the pair is the honest name


NAMEABLE = 0.18             # below this he is at the centre of the cube: nothing to name



def felt_names(chem, limit: int = 3) -> list[str]:
    """The readings that fit this state — offered to the model, never asserted.

    Returns [] when he is at the centre of the cube, which is the correct
    answer for a body doing nothing: a person at rest does not report a
    feeling, and handing the model a name there is what made every idle step
    sound like an announcement."""
    somatic = chem.somatic()
    if somatic:                              # Layer 1 outranks the cube
        return somatic[:limit]
    g = chem.corner_position()
    if g["corner"] in _SOCIAL_CORNERS or g["intensity"] < NAMEABLE:
        return []
    petal, _angle, _color, tiers = _PETAL[g["corner"]]
    tier = tiers[0 if g["intensity"] < 0.45 else 1 if g["intensity"] < 0.75 else 2]
    names = [felt(tier)]
    for w in (_plutchik().get(tier, {}).get("similar-words") or "").split(","):
        w = w.strip().lower()
        if w and w not in names:
            names.append(w)
    if g["margin"] < DYAD_MARGIN and g["runner_up"] not in _SOCIAL_CORNERS:
        dyad = _DYADS.get(frozenset((petal, _PETAL[g["runner_up"]][0])))
        if dyad and dyad not in names:
            names.insert(1, dyad)            # second: the corner still leads
    return names[:limit]
