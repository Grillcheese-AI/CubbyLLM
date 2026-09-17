"""coach — talking to Cubby-Man while he plays, and what your words have earned.

Wired: WIRED (stand-in game path: `CubbyPac.hear`). Nothing in cubbyllm/
imports this.

Owner: *"if we can send encouraging messages to cubby man as he plays it could
help maybe ... per example: you can do it!! or be careful of the ghost!"* and
then, on the first version of this file: *"well I should be able to chat with
it live not only 2 messages."*

Quite right, and the first version deserved it. It matched two phrase lists
and dropped everything else on the floor, which is a pair of buttons with a
regular expression painted on them. THIS IS A CHAT CHANNEL. Anything you say
reaches him. What the module does is notice the one kind of message that is
different in kind, not filter for the two it knows.

  ALMOST EVERYTHING     is about HIM, or about nothing in particular — "you
                        can do it", "that maze looks horrible", "how's it
                        going". It is affect arriving from a person and it
                        goes through the ordinary afferent transfer, which
                        already knows that another person's feelings are not
                        his feelings.
  A WARNING             is about THE WORLD. "Careful of the ghost" is not a
                        feeling, it is a claim that something is out there,
                        and it is the one kind of message allowed to raise
                        `threat` — which `afferent.py` otherwise forbids on
                        principle.

The exception is narrow and consistent with that principle. The rule there is
that the user's FEAR is not Cubby's danger. Someone POINTING AT A GHOST is not
reporting a feeling; they are reporting a fact, and Cubby lives in a world
where facts can be checked.

WHICH IS THE WHOLE POINT. A warning is FALSIFIABLE. The game knows whether a
ghost actually closed in over the next few steps, so every warning settles
into right or wrong and the speaker's standing moves. That gives TRUST A
REFERENT — the thing the oxytocin axis never had, and most of why that corner
sat dead. Cubby does not trust you because a table says so. He trusts you
because you have been right about ghosts, and the number is one he worked out
himself and can lose.

Standing is scored AGAINST THE BASE RATE rather than against chance. In a
crowded level where a ghost is near half the time, being right half the time
is worth nothing, and without that correction the way to farm trust would be
to shout "ghost!" every step.

WHY THE CHEERING CANNOT BE FARMED EITHER. Encouragement that always works is
a dial, and a dial the player can hold down is a cheat code — the "stoned all
the time high on oxytocin and dopamine" failure this project already hit once,
from an ambient `+0.05` nobody noticed. Three things stop it:

  IT IS NEVER `reward`. That input builds tolerance and belongs to the world:
  eating the thing, clearing the level. A sentence may not deliver it. Kind
  words reach `social` and a little `valence` and stop there.
  IT HABITUATES. The tenth "you got this" inside ten steps is noise and is
  treated as noise. Receptors downregulate; so does this.
  IT LANDS IN PROPORTION TO NEED. Encouragement to someone having a fine time
  is worth little. After the fourth death on level three it is worth a lot.
  That is not a nerf — it is what makes it encouragement instead of a slider.

ONE READING IS DELIBERATELY DIFFERENT HERE THAN IN CHAT. `appraise` treats
shouting and `!!` as threat, which is right for a stranger typing in a text
box and wrong for someone yelling GO GO GO at a maze. On this channel the
exclamation is AROUSAL, not danger: it goes to the contagion term and the
threat term never sees it. Only an actual warning raises threat.

WHAT IT BRINGS TO CUBBYLLM. It is the behavioural readout the VAD check said
this project needs. `docs/cube_vs_human_vad.md` concluded that the arousal
axis cannot be validated from text and has to be measured against behaviour —
and a maze run IS behaviour, with deaths, clears and an occupancy histogram.
Coach-on against coach-off on the same seed is a real experiment about an
affective input, which no chat corpus can supply.

It is also the shaping mechanism made concrete. The player who talked him
through level three has a Cubby who trusts them specifically, by a number that
was earned.

THE ADVANTAGE OVER A CONVENTIONAL MODEL. Tell an assistant "be careful of the
ghost" and it emits text agreeing to be careful; nothing about it has changed
and it is exactly as careless on the next step. There is no state for the
warning to land in, and being wrong about a ghost costs nothing, so trust can
only ever be asserted. Here the warning moves a body, the body changes what he
does, the claim is scored against what happened, and standing is earned
against a base rate. No retraining: this is host state on a frozen trunk.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from afferent import Afferent  # noqa: E402
from lexicon import AffectLexicon, Eligibility  # noqa: E402
from neurochem import appraise  # noqa: E402
from world import CLEAR, FAIL, HURT, REWARD  # noqa: E402

__wiring__ = "WIRED"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Shouting at a maze is excitement, not menace. `appraise` scores `!!` and
# CAPS as threat, which is right for a stranger in a text box and wrong for
# someone yelling GO GO GO at a screen.
_SHOUT = re.compile(r"!!|[A-Z]{3,}")

# How much learned threat a message needs before it counts as a claim about
# the world rather than a feeling about him. One warning word is enough —
# `AffectLexicon.read` takes the MAX of the threat column for exactly that
# reason — so this is a threshold on the strongest word, not on an average.
WARN_AT = 0.55


class Coach:
    """The player at the side of the maze, and what they have earned.

    One per player per agent. It owns an `Afferent`, so the no-mirroring rule
    and the consent gate are the same ones the chat path uses rather than a
    second copy of them; `credibility` is the part that only exists here,
    because only here can a claim be checked.
    """

    CHEER_FLOOR = 0.35           # what a kind word is worth to someone doing fine
    HABITUATION = 8              # steps; how fast repetition stops counting
    WARN_THREAT = 0.75           # at full standing. Scaled down by it.
    WARN_FOCUS = 0.50            # he looks around, whoever said it
    WARN_WINDOW = 6              # steps a warning may come true in
    STRANGER = 0.50              # what an unknown voice's warning is worth
    RATE = 0.30                  # how fast standing moves, before the base rate scales it

    def __init__(self, credibility: float | None = None) -> None:
        self.credibility = self.STRANGER if credibility is None else _clip(credibility, 0, 1)
        self.afferent = Afferent()
        self.lex = AffectLexicon()               # this player's words, learned from outcomes
        self.traces = Eligibility()              # and how recently each thing was said
        self.pending: list[dict] = []
        self.said: list[dict] = []                  # the transcript, for the report
        self.settled = {"right": 0, "wrong": 0}
        self._seen: set[str] = set()
        self._last_cheer_step: int | None = None
        self._cheer_run = 0
        # The base rate: how often a ghost is near ANYWAY. A warning that only
        # matches the background is worth nothing, and without this the way to
        # farm standing would be to shout "ghost!" on every step.
        self._steps = 0
        self._near_steps = 0

    # A prior on how often a ghost is near, worth 4 observations. The raw
    # fraction is unusable early and silently breaks the ledger: the first
    # warning settles on step 1 with `_steps == 1`, so a true one measures the
    # base rate at 1.0 and earns `RATE * (1 - 1.0)` — exactly nothing. Being
    # right must never be worth zero because it was the first thing observed.
    PRIOR_NEAR, PRIOR_N = 1.0, 4.0        # mean 0.25

    @property
    def base_rate(self) -> float:
        """P(a ghost is near) with nobody saying anything, smoothed."""
        return (self._near_steps + self.PRIOR_NEAR) / (self._steps + self.PRIOR_N)

    # ── hearing: anything at all ────────────────────────────────────────────
    def hear(self, text: str, *, step: int = 0, need: float = 0.0) -> dict:
        """One message from the player -> the drives it produces.

        `need` is how badly he is doing, 0..1 — pain, craving, a run of
        deaths. It scales affect and nothing else: a warning is about the
        world and is worth the same whether he is enjoying himself or not.

        Every message returns drives. A message that is only a greeting
        returns small ones, which is right; it should not return nothing,
        because somebody spoke to him.
        """
        text = (text or "").strip()
        if not text:
            return {"kind": None, "drives": {}, "why": "empty"}

        shouting = bool(_SHOUT.search(text))

        # TWO READERS, and they answer different questions. `appraise` is the
        # fixed general EN/FR lexicon: it knows "merci" and "horrible" and will
        # never know anything else. `self.lex` is what THIS player has taught
        # him, seeded but not fixed, and it is the only one that can report a
        # threat, because a claim about the world is the one reading that gets
        # checked against the world and therefore the only one it is safe to
        # let a stranger move.
        general = appraise(text, self._seen)
        learned = self.lex.read(text)
        self._seen.update(re.findall(r"[\w']+", text.lower()))
        warning = learned["threat"] >= WARN_AT
        # the larger-magnitude reading wins; a learned word beats a silent one
        valence = (learned["valence"] if abs(learned["valence"]) > abs(general["valence"])
                   else general["valence"])

        # Habituation: consecutive warm messages with no gap wear off.
        weight = 1.0
        if valence > 0:
            if self._last_cheer_step is not None and step - self._last_cheer_step <= self.HABITUATION:
                self._cheer_run += 1
            else:
                self._cheer_run = 0
            self._last_cheer_step = step
            weight = (self.CHEER_FLOOR + (1 - self.CHEER_FLOOR) * _clip(need, 0, 1)) \
                / (1.0 + self._cheer_run)

        # The channel's own reading of arousal. Shouting at a maze is
        # excitement, not menace — `appraise` scores it as threat, which is
        # right for a stranger in a text box and wrong here.
        self.afferent.valence = _clip(valence * weight, -1, 1)
        self.afferent.arousal = max(0.80 if shouting else 0.0, learned["arousal"])
        # `weight` rides the warmth too, not only the valence. It did not, and
        # `afferent.drives` takes `max(concern, WARMTH_SOCIAL * warmth)` — so
        # the warmth floor dominated and came out at a constant whatever the
        # habituation or the need said. Both halves of a kind word fade together
        # or the fading is decorative.
        self.afferent.warmth = _clip(max(general["social"], max(0.0, valence) * 0.8) * weight, 0, 1)
        self.afferent.source = "coach"
        d = self.afferent.drives()

        out = {"kind": "warn" if warning else "talk", "drives": d, "at": step,
               "weight": round(weight, 3), "text": text[:120],
               "known_words": learned["known"]}
        if warning:
            # the one message type that carries a world-claim, worth what the
            # claimant is worth
            self.pending.append({"at": step, "settled": False, "text": text})
            d["threat"] = round(self.WARN_THREAT * self.credibility, 3)
            d["focus"] = max(d["focus"], round(self.WARN_FOCUS * self.credibility, 3))
            out["credibility"] = round(self.credibility, 3)
            out["why"] = f"a claim about the world; standing {self.credibility:.2f}"
        else:
            out["why"] = (f"valence {valence:+.2f} x {weight:.2f}, "
                          f"{learned['known']} known words"
                          f"{', shouted' if shouting else ''}")
        self.said.append(out)
        # PRE, the first of the three factors: these words were used. Whether
        # they are ever learned depends on what the world does next.
        self.traces.mark(text, step)
        return out

    # ── settling: was there actually a ghost? ───────────────────────────────
    def tick(self, step: int, ghost_near: bool) -> list[dict]:
        """One game step. Updates the base rate, settles anything due.

        A warning is RIGHT when a ghost came near inside the window and WRONG
        when the window closed without one. Standing moves against the BASE
        RATE, not against chance: being right where a ghost is always near
        earns almost nothing, and being right where they rarely are is worth
        a great deal.
        """
        self._steps += 1
        self.traces.decay_to(step)               # POST: eligibility fades with the clock
        if ghost_near:
            self._near_steps += 1
        base, out = self.base_rate, []
        for p in self.pending:
            if ghost_near:
                p["settled"] = True
                self.settled["right"] += 1
                self.credibility = _clip(self.credibility + self.RATE * (1 - base), 0, 1)
                # THE WORLD JUST LABELLED THIS MESSAGE. GrillCheese's amygdala
                # had to be handed `target_emotion` by a caller; here a ghost
                # turned up, so the words that announced it were right and move
                # toward threat on their own.
                self.lex.learn(p["text"], valence=-0.2, arousal=0.75, threat=1.0)
                out.append({"verdict": "right", "at": p["at"], "credibility": self.credibility})
            elif step - p["at"] >= self.WARN_WINDOW:
                p["settled"] = True
                self.settled["wrong"] += 1
                self.credibility = _clip(self.credibility - self.RATE * max(base, 0.15), 0, 1)
                # and a false alarm moves them back, which is how a word that
                # only LOOKED like a warning stops being one
                self.lex.learn(p["text"], valence=0.0, arousal=0.3, threat=0.0, weight=0.6)
                out.append({"verdict": "wrong", "at": p["at"], "credibility": self.credibility})
        self.pending = [p for p in self.pending if not p["settled"]]
        return out

    def state(self) -> dict:
        """The ledger, in full. Small enough to show the player, which is the
        only honest way to keep a number about somebody."""
        return {"credibility": round(self.credibility, 3),
                "base_rate": round(self.base_rate, 3),
                "warnings": dict(self.settled), "messages": len(self.said),
                "pending": len(self.pending), "vocabulary": len(self.lex),
                "eligible": len(self.traces)}

    # ── the third factor: the world moves a neuromodulator ─────────────────
    #
    # The event kinds are `world.EVENTS`, so a different world raises the same
    # four and this works there untouched. The pac names are accepted as
    # aliases because that is what its call sites already say.
    NEUROMOD = {
        HURT:      (-0.6, 0.85, 0.35),   # it landed, it was loud, something was out there
        REWARD:    (0.8, 0.7, 0.0),
        CLEAR:     (0.7, 0.6, 0.0),
        FAIL:      (-0.4, 0.4, 0.0),     # deflation, not pain: nothing hurt him
        "caught":  (-0.6, 0.85, 0.35),
        "cleared": (0.7, 0.6, 0.0),
        "failed":  (-0.4, 0.4, 0.0),
    }
    MIN_CREDIT = 0.08          # below this a word is not being taught, it is being nudged by noise

    def outcome(self, kind: str, *, step: int = 0, magnitude: float = 1.0) -> int:
        """Something happened to him; whatever is still eligible gets the credit.

        THIS IS THE THIRD FACTOR ARRIVING. `hear` supplied PRE (the words were
        used), `Eligibility` carries POST (they are still in play), and this is
        MOD — a neuromodulator actually moved, and by how much. The product of
        the three is the learning rate for each message, which is what
        reward-modulated plasticity is.

        `magnitude` is meant to be a real measurement rather than a constant:
        the caller passes how far the chemistry moved, so a catch that took
        dopamine down 0.30 teaches harder than one that grazed him. Defaulting
        it to 1.0 keeps old call sites working and is the only part of this
        that is still a stand-in.

        WHAT THIS REPLACED was a fixed 12-step window at a flat 0.35: a
        sentence one step before a death and one twelve steps before got
        identical credit, one thirteen steps before got none, and the size of
        what happened did not enter at all. A cliff and a constant. This is
        graded, has no edge, and scales with the event.
        """
        target = self.NEUROMOD.get(kind)
        if target is None:
            return 0
        v, a, t = target
        self.traces.decay_to(step)
        mod = _clip(abs(float(magnitude)), 0.0, 1.0)
        if mod <= 0.0:
            return 0
        taught = 0
        for text, trace in self.traces.active():
            weight = trace * mod
            if weight < self.MIN_CREDIT:
                continue                          # too faint to be evidence about anything
            taught += self.lex.learn(text, valence=v, arousal=a, threat=t, weight=weight)
        return taught
