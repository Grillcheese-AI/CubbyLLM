"""cubbyllm/affect/afferent_drives.py -- the afferent signals' parts: what a user states about themself, how they
behave (bursts, gaps, the late hour), and the drives these give the chemistry. Split out of afferent.py (order 0031),
verbatim; afferent.py re-exports every name.

Wired: WIRED (the afferent).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── the stated channel: the user naming their own state ─────────────────────
#
# (valence, arousal), both in [-1, 1] / [0, 1]. EN, FR, and Québécois — the
# last of these is not decoration: `tanné`, `à boutte` and `brûlé` are the
# three most common self-reports in the harvested work corpus and none of them
# appears in any English lexicon or survives translation with its arousal
# intact. `écoeuré` is in here with a NEGATIVE valence and a note, because in
# Québec usage it inverts often enough ("écoeurant" = excellent) that the
# surrounding sign has to be checked rather than assumed.
#
# Arousal is what earns this table its keep. A general word lexicon cannot
# tell `tanné` (depleted, low arousal) from `en calvaire` (high arousal) — the
# VAD check says word-level arousal is nearly absent — but a person naming
# their own state names the activation too, and that is the one place the
# third axis is legible in language.
_STATED = {
    # spent / low arousal, negative
    r"\b(exhausted|drained|burnt? ?out|wiped|knackered|dead tired|so tired|running on empty)\b": (-0.5, 0.15),
    r"\b(tann[ée]|[àa] boutte|br[ûu]l[ée]|vid[ée]|cr[ée]v[ée]|[ée]puis[ée]|plus de jus)\b": (-0.5, 0.15),
    r"\b(tired|fatigu[ée]e?|sleepy|worn out)\b": (-0.25, 0.20),
    # high arousal, negative
    r"\b(furious|livid|pissed|fed up|so angry|raging)\b": (-0.7, 0.85),
    r"\b(en calvaire|en tabarnak|en crisse|choqu[ée]|enrag[ée]|[ée]coeur[ée])\b": (-0.7, 0.85),
    r"\b(frustrated|frustr[ée]e?|annoyed|irritated|tanned)\b": (-0.5, 0.60),
    r"\b(stressed|anxious|panicking|overwhelmed|stress[ée]e?|anxieu|d[ée]bord[ée])\b": (-0.5, 0.75),
    r"\b(scared|afraid|terrified|peur|effray[ée])\b": (-0.6, 0.80),
    # negative, mid
    r"\b(sad|down|low|miserable|gutted|triste|d[ée]prim[ée]|[àa] terre)\b": (-0.6, 0.25),
    r"\b(disappointed|d[ée][çc]u)\b": (-0.45, 0.30),
    r"\b(lost|stuck|confused|perdu|bloqu[ée]|m[êe]l[ée])\b": (-0.3, 0.45),
    # positive
    r"\b(excited|pumped|stoked|can'?t wait|hyp[ée]|excit[ée])\b": (0.7, 0.85),
    r"\b(happy|glad|great|relieved|content|heureu|soulag[ée]|ben content)\b": (0.6, 0.35),
    r"\b(calm|fine|ok|alright|good|[çc]a va|correct|tranquille)\b": (0.25, 0.20),
    r"\b(proud|fier|fi[èe]re)\b": (0.6, 0.50),
}


# "I am X" / "je suis X" / "chu X" — the frame that makes a word a SELF-report
# rather than a word about the world. "chu" and "j'su" are the spoken
# contractions; without them the Québécois half of the table never fires.
_SELF = re.compile(
    r"\b(i'?m|i am|i feel|im|feeling|me sens|je suis|j'?su[iy]s?|chu|chui|j'?me sens|"
    r"on est|je me sens)\b", re.I)


def stated(text: str) -> dict | None:
    """The user naming their own state -> {"valence", "arousal", "cue"}.

    Requires a self-report frame. "I'm frustrated" is the user's state; "the
    client is frustrated" is a fact about a third party and must not move
    Cubby's oxytocin toward the client.
    """
    if not text or not _SELF.search(text):
        return None
    best = None
    for rx, (v, a) in _STATED.items():
        m = re.search(rx, text, re.I)
        if m and (best is None or abs(v) > abs(best[1])):
            best = (m.group(0).lower(), v, a)
    if best is None:
        return None
    return {"cue": best[0], "valence": best[1], "arousal": best[2]}


# ── the behavioural channel: arousal, off the clock and the cadence ─────────
#
# Every one of these is already measured somewhere in validation/ against the
# harvested corpus. They are here as a reader rather than a finding: the
# thresholds are the stand-in's, and the honest thing to say about them is
# that the DIRECTIONS are evidenced and the NUMBERS are not yet.
BURST_AROUSAL = 0.18        # per message beyond the first in a run, capped


GAP_FAST = 20.0             # seconds; a reply this quick is an engaged one


LATE_HOUR = 22              # local; past here, activation is spent rather than high


def behavioural(gap_s: float | None = None, burst: int = 0, hour: int | None = None,
                words: int | None = None) -> dict | None:
    """Timing and cadence -> arousal only. Never valence.

    The channels a writer does not choose. `docs/cube_vs_human_vad.md` is the
    reason this exists as a separate channel at all: the arousal axis is not
    readable from emotion words (best of 126 NE labellings, rho +0.282) and
    has to come from behaviour or not at all.

    Returns None when nothing was supplied, which is the honest answer for a
    caller that has no timing — not a zero, which would read as "calm".
    """
    parts, why = [], []
    if burst:
        parts.append(min(0.55, BURST_AROUSAL * burst))
        why.append(f"burst x{burst}")
    if gap_s is not None and gap_s < GAP_FAST:
        parts.append(0.30 * (1 - gap_s / GAP_FAST))
        why.append("fast reply")
    if hour is not None and (hour >= LATE_HOUR or hour < 5):
        parts.append(-0.25)                     # still up, and running down rather than hot
        why.append("late")
    if words is not None and words <= 3:
        parts.append(0.10)                      # clipped, which cuts both ways; small on purpose
        why.append("clipped")
    if not parts:
        return None
    return {"arousal": _clip(0.35 + sum(parts), 0, 1), "cues": why}


# ── the transfer: what another person's state may do to this body ──────────
#
# Eight numbers, written down, arguable. This is the whole asymmetry.
#
# Read the columns as claims. AROUSAL_CONTAGION is the one automatic transfer
# and it is sign-free. CONCERN is what negative valence turns INTO rather than
# what it transfers as. SHARED_JOY is deliberately smaller than the joy Cubby
# gets from his own world, because it should be.
AROUSAL_CONTAGION = 0.55     # their EXCESS activation -> his `surge`. No sign. Automatic.


NEUTRAL_AROUSAL = 0.45       # below this a person is not activating anybody


#
# Contagion is about excess, not level, and the first version got this wrong in
# a way a test caught: `surge = 0.55 * arousal` gave a quiet, sad user (arousal
# 0.25) a small positive surge, which ran through `NE_suppresses_5HT` and
# dropped Cubby's serotonin 0.04 below a control that received nothing. He was
# catching it after all, by the back door — not through the valence term this
# module carefully blocks, but through an arousal term that should never have
# fired. Someone subdued does not key you up. There is no negative surge
# either: a calm person does not sedate you, they just fail to activate you.
CONCERN_SOCIAL = 0.70        # their trouble -> his oxytocin. This is the caring.


CONCERN_FOCUS = 0.45         # their trouble -> his attention. This is the listening.


SHARED_JOY = 0.40            # their good news -> his valence, attenuated


SHARED_JOY_SOCIAL = 0.50     # ... and the affiliation that comes with it


WARMTH_SOCIAL = 0.60         # being thanked, greeted, treated well


NEVER = ("pain", "reward", "threat")   # efferent only: his body, his world


def drives(read: dict | None) -> dict:
    """A read of the user -> the ODE's inputs. The asymmetry lives here.

    Negative valence produces NO negative valence. It produces `social` and
    `focus`: he attends and he warms, which is what concern is made of. The
    sign never crosses. Positive valence does cross, attenuated, because
    shared joy is real in a way shared misery is not.
    """
    out = {"novelty": 0.0, "threat": 0.0, "focus": 0.0, "valence": 0.0,
           "social": 0.0, "surge": 0.0}
    if not read:
        return out
    v = float(read.get("valence") or 0.0)
    a = float(read.get("arousal") or 0.0)
    warmth = float(read.get("warmth") or 0.0)

    # arousal, sign-free, automatic, and only the part above neutral
    excess = (a - NEUTRAL_AROUSAL) / (1.0 - NEUTRAL_AROUSAL)
    out["surge"] = _clip(AROUSAL_CONTAGION * max(0.0, excess), 0, 1)

    if v < 0:
        out["social"] = _clip(CONCERN_SOCIAL * -v, 0, 1)
        out["focus"] = _clip(CONCERN_FOCUS * -v, 0, 1)
        # valence stays at 0.0. This line is the entire point of the module.
    elif v > 0:
        out["valence"] = _clip(SHARED_JOY * v, 0, 1)
        out["social"] = _clip(SHARED_JOY_SOCIAL * v, 0, 1)

    out["social"] = _clip(max(out["social"], WARMTH_SOCIAL * warmth), 0, 1)
    return out
