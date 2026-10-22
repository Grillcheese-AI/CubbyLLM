"""neurochem — the real 5-signal neurochemical ODE, ported from cubemind.

Wired: WIRED (stand-in serve path: CubbyChat's state source — replaces the
nudge() placeholder; nothing in cubbyllm/ imports this).

Ported clean from `cubemind/cubemind/brain/neurochemistry.py` (the model the
cubbyverse demo runs) per the port-don't-link rule: the registry decorator is
stripped, the math is kept verbatim — dH/dt = alpha*drive*sensitivity -
beta*(H - resting), receptor-saturated couplings, cortisol as a slow EMA of
arousal (the HPA cascade), refractory receptor sensitivity, and the Lövheim-
cube emotion readout. All values in [0, 1]; the clip bands are the ones
`identity.HORMONE_RANGE` already pins.

`appraise(text, seen)` is the HOST's transparent, lexical (EN/FR) reading of
one user message into the ODE's five drive inputs (novelty, threat, focus,
valence, social). It is deliberately simple and inspectable — the honest
stand-in for a perception stack. `Neurochemistry.step_message()` runs a
couple of perception frames per message so bursts register within a turn.
"""
from __future__ import annotations

import math
import re

# split out in order 0031 (tools/split_module.py); re-exported so importers do not change
from .neurochem_appraise import _FOCUS, _NEG, _POS, _SOCIAL, _THREAT, _clip, _hits, _sigmoid, appraise  # noqa: F401
from .neurochem_update import NeuroUpdate
from .neurochem_signals import NeuroSignals
from .neurochem_body import NeuroBody

__wiring__ = "WIRED"


class Neurochemistry(NeuroBody, NeuroSignals, NeuroUpdate):
    """5-hormone neurochemical ODE with receptor dynamics (cubemind port).

    Dopamine/noradrenaline fast, serotonin/oxytocin medium, cortisol a very
    slow integrator. Update once per perception frame; `dt` sets smoothness.
    """

    def __init__(self, dt: float = 0.8) -> None:
        self._resting = {"DA": 0.30, "5HT": 0.45, "NE": 0.15, "OT": 0.20, "C": 0.15}
        # Start where this ODE actually settles, not at the declared resting
        # levels. They are not the same point — the couplings are additive
        # pushes that do not vanish at rest — so starting at `_resting` meant
        # every fresh agent began ALREADY DISPLACED and spent its first stretch
        # drifting, with everything that read the hormones reading a transient.
        # (Caught by the compass test: six frames of pure threat came out
        # `surprise` instead of fear, because serotonin had not yet fallen to
        # where it lives.)
        #
        # AND THE BASELINE IS COMPUTED HERE IF NOBODY ELSE HAS DONE IT, which
        # this used to leave to luck. It read `self._QUIESCENT or self._resting`
        # and relied on the first instance in a process being the probe — true
        # only when something calls `quiescent()` before the first real agent
        # is built. `CubbyChat.__init__` does not, so the first Cubby in a
        # fresh process started at the DECLARED levels and every later one
        # started settled: caution 0.255 against 0.130, a body nearly twice as
        # guarded as its own resting state, for the length of the first
        # conversation. Harmless while nothing read the knobs; not harmless now
        # that `choosing` sets a speaking threshold from them, which is how it
        # was found. `_SETTLING` keeps the probe's own construction out of the
        # recursion.
        if Neurochemistry._QUIESCENT is None and not Neurochemistry._SETTLING:
            Neurochemistry._SETTLING = True
            try:
                Neurochemistry.quiescent()
            finally:
                Neurochemistry._SETTLING = False
        start = self._QUIESCENT or self._resting
        self.dopamine = start["DA"]
        self.serotonin = start["5HT"]
        self.noradrenaline = start["NE"]
        self.oxytocin = start["OT"]
        self.cortisol = start["C"]
        self._da_sensitivity = 1.0
        self._ne_sensitivity = 1.0
        self._alpha = {"DA": 0.35, "5HT": 0.20, "NE": 0.40, "OT": 0.22, "C": 0.02}
        self._beta = {"DA": 0.25, "5HT": 0.10, "NE": 0.30, "OT": 0.18, "C": 0.01}
        self._couplings = {
            "C_suppresses_DA": -0.10, "C_suppresses_5HT": -0.06, "C_suppresses_OT": -0.04,
            "5HT_boosts_OT": 0.05, "OT_boosts_DA": 0.06, "DA_modulates_OT": 0.03,
            "NE_boosts_DA": 0.05, "NE_suppresses_5HT": -0.03, "5HT_dampens_NE": -0.04,
        }
        self._dt = dt
        self.valence = 0.0
        self.affect_arousal = 0.0
        self.stress = 0.0
        self.dominant_emotion = "neutral"
        # ── pain, and the slow half of reward (2026-09-15) ──────────────────
        # `_da_sensitivity` above is the FAST refractory homeostat: it tracks
        # the current dopamine level and drifts back to 1.0 within a few
        # frames, so nothing can accumulate in it across a run. Tolerance is
        # the slow one, and it is a separate term on purpose — two timescales
        # that would otherwise fight, and only one of them is what "he needs
        # more of it now" means.
        self._da_tolerance = 1.0                         # 1.0 = a naive receptor; falls with repeated big rewards
        self._since_reward = 999                         # frames since the last one, for abstinence
        self.pain = 0.0                                  # the ache that outlasts the event
        self.rewards_taken = 0                           # how many big hits, ever

    DA_MIN_TOLERANCE = 0.45      # receptors downregulate; they do not vanish
    TOLERANCE_RATE = 0.055       # per unit of reward actually delivered
    RECOVERY_RATE = 0.004        # per frame of abstinence, once it has lasted
    ABSTINENCE_AFTER = 40        # frames before the receptors start coming back
    PAIN_DECAY = 0.12            # per frame; a bad catch is felt for a while


    # ── interoception: the state from the inside, with no name on it ────────
    #
    # Owner, 2026-09-15: *"after failing many times it could feel despair, more
    # courage, or something else based on the hormones level and how it
    # interprets it. We need to not tell it how to interpret the hormonal
    # changes. However we can name emotions that suit with hormone A or B or C
    # mixed with D to guide the model in its speech."*
    #
    # So this layer stops short of the verdict. `body()` reports SENSATIONS —
    # what a mixture feels like, in the words a body has, none of which is an
    # emotion word. `could_be()` offers NAMES that fit the region, as
    # candidates, and deliberately offers opposed ones wherever the chemistry
    # genuinely underdetermines the meaning: low dopamine under high cortisol
    # is despair or it is grim stubbornness, and nothing in the hormones
    # decides which. That choice is the speaker's, which is the whole point —
    # a label computed here and handed over is not a feeling, it is a readout
    # with a feeling word printed on it.
    #
    # `_BODY` and `_NAMES` are ordered by how far the reading is from resting,
    # so the strongest thing is said first and the list stays short.

    # the biological clamps `update` enforces, so a deviation can be measured
    # against the room the hormone actually has in each direction
    _BAND = {"DA": (0.15, 0.85), "5HT": (0.10, 0.90), "NE": (0.05, 0.90),
             "OT": (0.05, 0.85), "C": (0.05, 0.80)}

    _QUIESCENT: dict | None = None
    _SETTLING = False        # the probe is building the baseline; do not re-enter

    @classmethod
    def quiescent(cls) -> dict:
        """Where this ODE actually sits when nothing is happening.

        NOT `_resting`. `_resting` is where the port SAYS each hormone rests,
        but the receptor couplings are additive per-frame pushes that do not
        vanish at rest: `5HT_dampens_NE` alone drags noradrenaline from a
        declared 0.15 down to about 0.07, and `5HT_boosts_OT` plus
        `DA_modulates_OT` park oxytocin well above 0.20. Measuring deviation
        from `_resting` therefore measured the port's own bias and not the
        agent's state — which is why a body doing nothing read as *"slow,
        heavy-limbed"* and an agent alone in a maze read as *"warm toward
        things"*, and why patching the thresholds one region at a time kept
        moving the problem around instead of fixing it.

        So: run a fresh instance with no input until it settles, once, and
        measure everything against THAT. At true rest every deviation is zero
        and he says nothing about his body, which is correct — a person at rest
        does not report their noradrenaline."""
        if cls._QUIESCENT is None:
            probe = cls.__new__(cls)                     # a bare instance: no subclass __init__ side effects
            Neurochemistry.__init__(probe)
            # Seed it with the declared levels BEFORE the probe runs. `update`
            # ends by classifying the emotion, the classifier asks for
            # quiescent, and without this the first call recurses into itself
            # forever. The probe's own labels are thrown away, so a provisional
            # baseline for those 400 frames costs nothing.
            cls._QUIESCENT = dict(probe._resting)
            for _ in range(400):
                probe.update()
            cls._QUIESCENT = {"DA": probe.dopamine, "5HT": probe.serotonin,
                              "NE": probe.noradrenaline, "OT": probe.oxytocin,
                              "C": probe.cortisol}
        return cls._QUIESCENT


    def to_dict(self) -> dict:
        return {"cortisol": self.cortisol, "dopamine": self.dopamine,
                "serotonin": self.serotonin, "oxytocin": self.oxytocin,
                "noradrenaline": self.noradrenaline, "valence": self.valence,
                "arousal": self.arousal, "stress": self.stress, "weight": self.weight,
                "da_sensitivity": self._da_sensitivity, "emotion": self.dominant_emotion,
                "pain": self.pain, "tolerance": self.tolerance, "craving": self.craving,
                "rewards_taken": self.rewards_taken}


