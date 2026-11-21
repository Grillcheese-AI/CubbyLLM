"""mood -- Cubby's mood on Plutchik's eight petals, each an integer from -10 to +10, and its GoEmotions shade
(order 0079).

Wired: STANDALONE (the 0079 data builder and its checks; the host reads it behind a flag once measured).

Nick, 2026-11-20: "mood we should use the full plutchik wheel"; "each emotion should have many levels from -10 to +10
0 being neutral"; "all 8 with their own -10 to +10"; "its emotion should be regulated by the hormonal system"; "we can
also map goemotion for more nuance".

NOTHING HERE HOLDS STATE. Every number is read from a `Neurochemistry` (the ODE), so an emotion moves only as the
hormones do: paced by their rates, bounded by their bands and receptors, back to 0 as they settle.
- **joy, fear, surprise, sadness, disgust, anger, anticipation:** the state's place in the cube (`NeuroBody.cube`,
  scaled so the quiescent state is the centre) against each petal's Lövheim corner (`feeling._PETAL`), as closeness
  1 - d / sqrt(3): 0.5 at the centre, 1 at the corner, 0 at the opposite corner. The scale is (closeness - 0.5) * 20,
  so the quiescent state reads 0, the petal's own corner +10 and the farthest point -10. Sadness owns two corners
  (distress, spent): the nearer one counts.
- **trust:** oxytocin, which is not a cube axis (`feeling.py`: "trust is oxytocin"), against its quiescent level,
  scaled by the room its band (`_BAND["OT"]`) leaves on each side.
- **The names** read the positive side: +1 to +3, +4 to +7, +8 to +10 are the petal's three tiers; a dyad is two
  neighbouring petals of the wheel both at +4 or more; all eight within one of 0 is calm.
- **The shade:** GoEmotions' labels of the petals at +4 or more, at the tier the scale is in
  (`goemotions_cube.TIER`). Labels of one petal and tier share a point in the cube (that module says so), so the
  hormones cannot tell them apart; the tie is broken by the second petal a label is a dyad of (love leans on trust,
  remorse on disgust, gratitude on joy), and otherwise by the petal's own word. It is a reading, never a fact.
"""
from __future__ import annotations

import math

__wiring__ = "STANDALONE"

ORDER = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
TIERS = {"joy": ("serenity", "joy", "ecstasy"), "trust": ("acceptance", "trust", "admiration"),
         "fear": ("apprehension", "fear", "terror"), "surprise": ("distraction", "surprise", "amazement"),
         "sadness": ("pensiveness", "sadness", "grief"), "disgust": ("boredom", "disgust", "loathing"),
         "anger": ("annoyance", "anger", "rage"), "anticipation": ("interest", "anticipation", "vigilance")}
# the wheel's neighbours and their primary dyads, in the wheel's order
DYADS = {("joy", "trust"): "love", ("trust", "fear"): "submission", ("fear", "surprise"): "awe",
         ("surprise", "sadness"): "disapproval", ("sadness", "disgust"): "remorse", ("disgust", "anger"): "contempt",
         ("anger", "anticipation"): "aggressiveness", ("anticipation", "joy"): "optimism"}
OPPOSITE = {"joy": "sadness", "trust": "disgust", "fear": "anger", "surprise": "anticipation"}
OPPOSITE.update({v: k for k, v in list(OPPOSITE.items())})
# GoEmotions labels that Plutchik writes as a dyad of their own petal and a second one; the second petal breaks a
# tie. Only his own dyads: love = joy + trust, remorse = sadness + disgust, optimism = anticipation + joy (primary);
# disappointment = surprise + sadness, anxiety (nervousness) = anticipation + fear (secondary); pride = anger + joy
# (tertiary). No lean is invented for a label he gives no dyad.
LEANS = {"love": "trust", "remorse": "disgust", "optimism": "joy", "disappointment": "surprise",
         "nervousness": "anticipation", "pride": "anger"}
DIAG = math.sqrt(3)
# GoEmotions' labels on the petals and their tiers, copied from the builders' checked tables
# (`validation/sft/chat_sft_v6.GOEMOTIONS_PETAL`, `validation/builders/goemotions_cube.TIER`) so the host never imports
# a builder; `tests/affect/test_mood.py` pins them equal.
LABEL_PETAL = {"anger": "anger", "annoyance": "anger", "curiosity": "anticipation", "desire": "anticipation",
               "optimism": "anticipation", "disapproval": "disgust", "disgust": "disgust", "fear": "fear",
               "nervousness": "fear", "amusement": "joy", "excitement": "joy", "joy": "joy", "love": "joy",
               "pride": "joy", "relief": "joy", "disappointment": "sadness", "embarrassment": "sadness",
               "grief": "sadness", "remorse": "sadness", "sadness": "sadness", "confusion": "surprise",
               "realization": "surprise", "surprise": "surprise", "admiration": "trust", "approval": "trust",
               "caring": "trust", "gratitude": "trust", "neutral": "calm"}
LABEL_TIER = {"admiration": 2, "amusement": 1, "anger": 1, "annoyance": 0, "approval": 0, "caring": 1, "confusion": 0,
              "curiosity": 0, "desire": 1, "disappointment": 1, "disapproval": 1, "disgust": 1, "embarrassment": 1,
              "excitement": 2, "fear": 1, "gratitude": 1, "grief": 2, "joy": 1, "love": 2, "nervousness": 0,
              "optimism": 1, "pride": 2, "realization": 0, "relief": 1, "remorse": 1, "sadness": 1, "surprise": 1,
              "neutral": 0}


def _corner_petal() -> dict[str, str]:
    from .feeling import _PETAL
    return {corner: row[0] for corner, row in _PETAL.items()}


def tier_of(v: int) -> int | None:
    """0, 1 or 2 for a scale at +1..+3, +4..+7, +8..+10; None at or below 0."""
    return None if v <= 0 else 0 if v <= 3 else 1 if v <= 7 else 2


def scales_at(pos: tuple, ot: float, quiet_ot: float, band_ot: tuple) -> dict[str, int]:
    """The eight scales for a cube position (5-HT, DA, NE, quiescent at 0.5) and an oxytocin level."""
    from .neurochem import Neurochemistry
    near: dict[str, float] = {}
    for corner, petal in _corner_petal().items():
        c = Neurochemistry._CORNERS[corner]
        close = 1.0 - math.dist(pos, c) / DIAG
        near[petal] = max(near.get(petal, 0.0), close)
    out = {p: max(-10, min(10, round((near[p] - 0.5) * 20))) for p in near}
    lo, hi = band_ot
    room = (hi - quiet_ot) if ot >= quiet_ot else (quiet_ot - lo)
    out["trust"] = max(-10, min(10, round(10 * (ot - quiet_ot) / max(room, 1e-6))))
    return {p: out[p] for p in ORDER}


def scales(chem) -> dict[str, int]:
    """The eight scales of a `Neurochemistry`, read now."""
    q = chem.quiescent()
    return scales_at(chem.cube(), chem.oxytocin, q["OT"], chem._BAND["OT"])


def line(sc: dict[str, int]) -> str:
    """The talk seat's line: `Mood: joy +6, trust +3, fear -1, ...`, all eight in a fixed order."""
    return "Mood: " + ", ".join(f"{p} {sc[p]:+d}" if sc[p] else f"{p} 0" for p in ORDER)


def named(sc: dict[str, int]) -> list[str]:
    """The wheel's names that read the scales, strongest first, dyads ahead of their petals; ["calm"] at rest."""
    if all(abs(v) <= 1 for v in sc.values()):
        return ["calm"]
    out = [d for (a, b), d in DYADS.items() if sc[a] >= 4 and sc[b] >= 4]
    for p in sorted((p for p in ORDER if sc[p] >= 1), key=lambda p: -sc[p]):
        out.append(TIERS[p][tier_of(sc[p])])
    return out


def shade(sc: dict[str, int], limit: int = 2) -> list[str]:
    """GoEmotions labels for the petals at +4 or more, at their tier; [] when none is."""
    TIER, GOEMOTIONS_PETAL = LABEL_TIER, LABEL_PETAL
    out = []
    for p in sorted((p for p in ORDER if sc[p] >= 4), key=lambda p: -sc[p]):
        t = tier_of(sc[p])
        labels = sorted(lab for lab, pet in GOEMOTIONS_PETAL.items() if pet == p and TIER.get(lab) == t)
        if not labels:                                   # no label at this tier: the nearest tier's
            labels = sorted((lab for lab, pet in GOEMOTIONS_PETAL.items() if pet == p),
                            key=lambda lab: (abs(TIER.get(lab, 1) - t), lab))[:1]
        leaning = [lab for lab in labels if LEANS.get(lab) and sc[LEANS[lab]] >= 4]
        pick = leaning[0] if leaning else (p if p in labels else labels[0])
        if pick not in out:
            out.append(pick)
        if len(out) >= limit:
            break
    return out


def validate() -> list[str]:
    """The checks 0079 sets: the quiescent state reads 0 on all eight; each corner reads +8 or more on its own petal;
    the petal opposite it on the wheel reads below 0 there (trust and disgust excepted: trust has no corner)."""
    from .neurochem import Neurochemistry
    q, band = Neurochemistry.quiescent(), Neurochemistry._BAND["OT"]
    problems = []
    rest = scales_at((0.5, 0.5, 0.5), q["OT"], q["OT"], band)
    if any(rest.values()):
        problems.append(f"quiescent reads {rest}")
    for corner, petal in _corner_petal().items():
        sc = scales_at(Neurochemistry._CORNERS[corner], q["OT"], q["OT"], band)
        if sc[petal] < 8:
            problems.append(f"corner {corner}: {petal} reads {sc[petal]}")
        opp = OPPOSITE[petal]
        if opp != "trust" and sc[opp] >= 0:
            problems.append(f"corner {corner}: the opposite petal {opp} reads {sc[opp]}")
    return problems


__all__ = ["ORDER", "DYADS", "TIERS", "line", "named", "scales", "scales_at", "shade", "tier_of", "validate"]
