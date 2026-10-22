"""afferent — the user's state reaching Cubby's body, without becoming it.

Wired: WIRED (stand-in serve path: `CubbyChat.nudge`). Nothing in cubbyllm/
imports this.

THE EFFERENT CHANNEL ALREADY EXISTED and is the rest of this system: Cubby's
own hormones, driven by what happens to Cubby, coming out as manner. This is
the other direction — what the person on the far side of the wire is feeling,
and what that is allowed to do to him.

WHAT WAS THERE BEFORE, AND WHY IT WAS WRONG. `perception.PETAL_DRIVES` mapped
the user's read emotion straight onto Cubby's drives:

    "sadness": {"valence": -0.7, "social": 0.2}
    "anger":   {"threat":   0.6, "valence": -0.7}
    "fear":    {"threat":   0.8, "valence": -0.5}

Read those as behaviour rather than as a table. A sad user drops Cubby's
serotonin and dopamine: he gets sad too. An angry user registers as a THREAT:
Cubby becomes afraid of the person he is talking to. That is a mirror and a
flinch, and the owner's brief is neither — *"his main mission is to become
your best friend, helper and counselor ... and help you better your life."*
A friend whose mood collapses into yours is not counsel, and a friend who
reads your anger as danger to himself is the flinch that makes people stop
talking.

THE PRINCIPLE, which is also how bodies actually work: another person's state
is a FACT ABOUT THE WORLD, not a force on your chemistry — with two exceptions
that are real and physiological.

  AROUSAL IS CONTAGIOUS. Someone keyed up keys you up; this is fast, automatic,
  and carries NO SIGN. It lands on `surge` (adrenaline, the one NE input with
  no negative attached) and nowhere else.
  AFFILIATION IS CONTAGIOUS. Warmth begets warmth. It lands on `social`, which
  is the only oxytocin drive there is.

Valence does NOT transfer. Someone else's bad day makes you ATTENTIVE AND WARM
— which is what concern is made of — not sad. So distress maps to `social` and
`focus`, and the sign stays at zero. Shared joy is the one asymmetry in the
other direction: it transfers, attenuated, because it genuinely does.

PAIN IS NEVER AFFERENT. A user writing "this hurts" does not hurt Cubby. The
somatic layer intercepts the naming above a threshold (`Neurochemistry.
somatic`), and letting a sentence reach it would let anyone counterfeit a body
state and take the compass with it. `pain`, `reward` and `threat` stay
efferent: Cubby's own body, Cubby's own world.

WHICH CHANNEL MAY SET WHAT, and this is the part the evidence decided rather
than taste. `docs/cube_vs_human_vad.md`, against two human lexicons:

    valence ~ (5HT+DA)/2   +0.538 to +0.800 across three vocabularies
    arousal ~ NE           -0.101, and the BEST of 126 possible NE labellings
                           reaches only +0.282

Emotion words carry valence and barely carry arousal. So:

  LEXICAL      the words themselves -> VALENCE ONLY. Its arousal claim is
               dropped on the floor, because it was measured and it is not
               there.
  BEHAVIOURAL  burst depth, gap before the message, hour, length -> AROUSAL
               ONLY. Never valence. These are the channels the writer does not
               choose, which is precisely why they carry activation, and this
               repo already measures every one of them.
  STATED       the user naming their own state -> BOTH, and it outranks the
               other two. A person saying "chu tanné" is the authority on
               their own valence AND their own arousal, and the distinction
               a general word lexicon cannot make — `tanné` is low-arousal
               negative, `en calvaire` is high-arousal negative — is exactly
               what self-report gives for free.

CONSENT, which the owner set the shape of: *"it should always be asked to the
user"*. The three channels do not need the same permission and pretending they
do is how a privacy control becomes theatre.

  lexical      always on. Reading the message the user chose to send is not
               surveillance; it is the conversation.
  stated       on when volunteered. Saying "I'm frustrated" IS the consent for
               that turn.
  behavioural  OFF until explicitly granted. Timing a person's replies and
               reading their tiredness off the clock is inference from
               metadata they did not offer, and it is the one that needs
               asking. `grant()` / `decline()`, three states, never assumed.

WHAT THIS BRINGS TO CUBBYLLM. The user can be read without being copied, so
Cubby can be steady while you are not — the thing a counselor has to be. It
wakes the oxytocin axis, which the Lövheim cube has no room for and which
therefore had almost nothing driving it: concern and warmth now have an input.
And it is the shaping mechanism — `state()` is a small, inspectable,
user-owned record of how you have been, which the emitter reads at generation
time.

THE ADVANTAGE OVER A CONVENTIONAL MODEL. A conventional assistant has one
lever for the user's mood: the text in its context window, which it responds
to by matching register — the sycophancy failure, where sadness is met with
sadness and anger with apology, because nothing in the architecture
distinguishes YOUR state from ITS state. Here they are separate objects with
separate dynamics, the transfer between them is eight numbers you can read and
argue with, and the arousal half rides channels no wording can fake. Nothing
is retrained to get it: this is host state on the same frozen trunk.
"""
from __future__ import annotations

import re

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .afferent_drives import (AROUSAL_CONTAGION, BURST_AROUSAL, CONCERN_FOCUS, CONCERN_SOCIAL, GAP_FAST,  # noqa: F401
                              LATE_HOUR, NEUTRAL_AROUSAL, NEVER, SHARED_JOY, SHARED_JOY_SOCIAL, WARMTH_SOCIAL, _SELF,
                              _STATED, _clip, behavioural, drives, stated)

__wiring__ = "WIRED"


class Afferent:
    """The user's state, as Cubby is allowed to receive it.

    Holds the consent gate and the short memory that makes a stated state last
    a few turns instead of one. Not a profile and not a log: `state()` is a
    handful of numbers the user can be shown in full.
    """

    NOT_ASKED, GRANTED, DECLINED = "not-asked", "granted", "declined"
    DECAY = 0.65                 # per turn; a stated mood fades over ~4 turns
    FLOOR = 0.08                 # below this it is gone, not faintly remembered

    def __init__(self, timing_consent: str = NOT_ASKED) -> None:
        if timing_consent not in (self.NOT_ASKED, self.GRANTED, self.DECLINED):
            raise ValueError(f"unknown consent state {timing_consent!r}")
        self.timing_consent = timing_consent
        self.valence = 0.0
        self.arousal = 0.0
        self.source: str | None = None       # what set the current reading
        self.turns = 0

    # ── the gate ────────────────────────────────────────────────────────────
    def grant(self) -> None:
        """The user allowed Cubby to read timing and cadence. Their words, not
        an inference from their silence."""
        self.timing_consent = self.GRANTED

    def decline(self) -> None:
        self.timing_consent = self.DECLINED
        self.source = None

    @property
    def may_read_timing(self) -> bool:
        return self.timing_consent == self.GRANTED

    # ── the read ────────────────────────────────────────────────────────────
    def read(self, text: str, *, lexical_valence: float = 0.0, warmth: float = 0.0,
             gap_s: float | None = None, burst: int = 0, hour: int | None = None,
             words: int | None = None) -> dict:
        """One user turn -> the current reading, after decay.

        Precedence is credibility, not recency: a stated state outranks a
        guess off the wording, and nothing outranks a stated state. Each
        channel may set only what the evidence says it can read.
        """
        self.turns += 1
        self.valence *= self.DECAY
        self.arousal *= self.DECAY
        if abs(self.valence) < self.FLOOR:
            self.valence = 0.0
        if self.arousal < self.FLOOR:
            self.arousal = 0.0

        src = []
        said = stated(text)
        if said:                                  # both dimensions; the user is the authority
            self.valence, self.arousal = said["valence"], said["arousal"]
            src.append(f"stated:{said['cue']}")
        else:
            if lexical_valence:                   # VALENCE ONLY — see the header
                self.valence = _clip(lexical_valence, -1, 1)
                src.append("lexical")
            if self.may_read_timing:              # AROUSAL ONLY, and only with consent
                beh = behavioural(gap_s=gap_s, burst=burst, hour=hour, words=words)
                if beh:
                    self.arousal = beh["arousal"]
                    src.append("timing:" + "+".join(beh["cues"]))

        self.source = ", ".join(src) or None
        self.warmth = _clip(warmth, 0, 1)
        return self.state()

    def state(self) -> dict:
        """What Cubby holds about the user. Small enough to show them."""
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "warmth": round(getattr(self, "warmth", 0.0), 3),
                "source": self.source, "timing_consent": self.timing_consent,
                "turns": self.turns}

    def drives(self) -> dict:
        """The ODE inputs this reading is allowed to produce."""
        return drives(self.state())

    def forget(self) -> None:
        """Drop everything held about the user. The user asked; it goes."""
        self.valence = self.arousal = 0.0
        self.warmth = 0.0
        self.source = None
