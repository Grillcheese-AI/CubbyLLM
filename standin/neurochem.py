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

__wiring__ = "WIRED"


def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
    """Receptor saturation curve. Prevents linear blow-up at extremes."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Neurochemistry:
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
        # where it lives.) The very first instance is the probe that computes
        # quiescent, and it starts at `_resting` and converges regardless.
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

    # ── core ODE update (one perception frame) ──────────────────────────────
    def update(self, novelty: float = 0.0, threat: float = 0.0, focus: float = 0.0,
               valence: float = 0.0, social: float = 0.0,
               pain: float = 0.0, surge: float = 0.0, reward: float = 0.0) -> None:
        """One perception frame.

        The three newer signals are deliberately not shades of the old ones:

          PAIN    is the thing happening, where `threat` is the thing being
                  about to happen. It spikes NE, suppresses DA and 5-HT,
                  drives cortisol, and — the part that matters — it LINGERS,
                  decaying over frames rather than ending with the event.
          SURGE   is adrenaline: a hard NE spike with no negative sign on it.
                  Threat and delight both surge; only one of them is bad.
          REWARD  is a big hit, not the mild pleasure `valence` carries. It is
                  the only thing that builds tolerance, because a life of small
                  pleasures does not."""
        # the ache outlasts the blow: whichever is worse, this hit or what is
        # left of the last one
        self.pain = _clip(max(_clip(pain, 0, 1), self.pain * (1 - self.PAIN_DECAY)), 0, 1)
        joy = max(0.0, valence)
        neg = max(0.0, -valence) + self.pain             # hurting IS feeling bad, whatever else is going on

        # the somatic skew, from the named influence vectors rather than
        # coefficients buried in three different expressions
        s_5ht, s_da, s_ne = (p * self.pain + c * self.craving
                             for p, c in zip(self.PAIN_INFLUENCE, self.CRAVING_INFLUENCE))
        drive_NE = ((novelty * 0.5 + threat * 0.4 + focus * 0.4
                     + surge * 0.7 + s_ne) * self._ne_sensitivity)
        # Both receptor terms ride the whole drive: the fast refractory one and
        # the slow tolerance. That is the mechanism — the world hands him the
        # same ghost every time and he feels less of it each time.
        #
        # TOLERANCE BLUNTS EVERYTHING, not just the drug. A downregulated
        # receptor is downregulated for ordinary pleasures too, which is what
        # makes the gap between hits FEEL like a gap — put it on the reward
        # term alone and baseline dopamine never drops, so there is no
        # withdrawal, no shortfall, and `craving` cannot climb past the
        # constant part of its own formula. That was the second reason
        # exp_r39 came out null.
        #
        # THE CONSTANTS ARE GONE, and that is a correction rather than a tweak.
        # `delta = alpha*drive - beta*(value - resting)` already pulls each
        # hormone toward its resting level, so a constant term in the drive
        # does not represent tonic activity — it MOVES THE FIXED POINT. The
        # inherited `0.15` here and `0.35` in serotonin put the real
        # equilibrium at DA 0.62 and 5-HT 0.90 (its ceiling) against declared
        # rests of 0.30 and 0.45. The owner, watching a live run: *"it never
        # feels anxiety when it fails a level or frustration ... still seems
        # always happy."* It could not: Lövheim's nearest corner to a
        # permanently saturated (0.90, 0.62, 0.05) is "warm", every time,
        # whatever happened to him.
        #
        # THREAT SUPPRESSES DOPAMINE, and that is the input this port was
        # missing rather than a new idea. `_on_caught` in pacman.py already
        # subtracted 0.30 from dopamine by hand, with a comment saying so:
        # *"the aversive outcome: a dopamine DIP (reward-prediction error),
        # which the port has no input for -- without it the NE->DA coupling
        # lifts dopamine and Lövheim's low-5HT/high-DA/high-NE corner reads as
        # RAGE instead of fear."* The fix belongs here, once, not at whichever
        # call sites happened to notice: without it every frightening thing
        # that is not a catch reads as anger, because NE_boosts_DA carries
        # dopamine UP on pure threat.
        drive_DA = ((novelty * 0.5 + joy * 0.4 + reward * 0.9)
                    * self._da_sensitivity * self._da_tolerance
                    + s_da - threat * 0.45)
        drive_5HT = 0.25 * joy - 0.2 * threat - 0.1 * neg + s_5ht
        # No ambient affiliation term. The `+0.05` this port inherited kept
        # oxytocin sitting well above resting with no social input at all,
        # which is defensible for a chat agent and plainly wrong for an agent
        # alone in a maze: it made "warm toward things" his DEFAULT state, and
        # it outranked despair after five failed levels in a row. Oxytocin now
        # rises when something social or good actually happens, and otherwise
        # decays to rest.
        drive_OT = social * 0.5 + joy * 0.2

        # DA and 5-HT may go NEGATIVE. They are the two drives with subtractive
        # terms — threat and pain suppress both — and clipping the net input at
        # zero threw that suppression away entirely: the most a frightening
        # thing could do was stop *raising* dopamine, while `NE_boosts_DA` went
        # on lifting it, so six frames of pure threat came out as rage. An
        # inhibitory input is an ordinary thing for a drive to carry; NE and OT
        # have no subtractive terms and stay non-negative.
        drives = {"DA": _clip(drive_DA, -1, 1), "5HT": _clip(drive_5HT, -1, 1),
                  "NE": _clip(drive_NE, 0, 1), "OT": _clip(drive_OT, 0, 1)}
        values = {"DA": self.dopamine, "5HT": self.serotonin,
                  "NE": self.noradrenaline, "OT": self.oxytocin}

        new = {}
        for h in ("DA", "NE", "5HT", "OT"):
            delta = self._alpha[h] * drives[h] - self._beta[h] * (values[h] - self._resting[h])
            new[h] = values[h] + self._dt * delta

        # cortisol: slow EMA of arousal (the HPA cascade, minutes timescale)
        instantaneous_stress = _clip(new["NE"] * 0.5 + threat * 0.3 + neg * 0.2
                                     + self.pain * 0.4, 0, 1)
        new["C"] = self.cortisol + 0.015 * (instantaneous_stress - self.cortisol)

        # receptor-saturated couplings
        c_sat = _sigmoid(new["C"], 0.4, 8)
        da_sat = _sigmoid(new["DA"], 0.4, 8)
        sht_sat = _sigmoid(new["5HT"], 0.5, 8)
        ot_sat = _sigmoid(new["OT"], 0.4, 8)
        ne_sat = _sigmoid(new["NE"], 0.3, 8)
        new["DA"] += self._couplings["C_suppresses_DA"] * c_sat
        new["5HT"] += self._couplings["C_suppresses_5HT"] * c_sat
        new["OT"] += self._couplings["C_suppresses_OT"] * c_sat
        new["OT"] += self._couplings["5HT_boosts_OT"] * sht_sat
        new["DA"] += self._couplings["OT_boosts_DA"] * ot_sat
        new["OT"] += self._couplings["DA_modulates_OT"] * da_sat
        new["DA"] += self._couplings["NE_boosts_DA"] * ne_sat
        new["5HT"] += self._couplings["NE_suppresses_5HT"] * ne_sat
        new["NE"] += self._couplings["5HT_dampens_NE"] * sht_sat

        # biological clamps (identity.HORMONE_RANGE pins the same bands)
        self.dopamine = _clip(new["DA"], 0.15, 0.85)
        self.serotonin = _clip(new["5HT"], 0.10, 0.90)
        self.noradrenaline = _clip(new["NE"], 0.05, 0.90)
        ot_decay = 0.05 * (new["OT"] - self._resting["OT"])
        self.oxytocin = _clip(new["OT"] - ot_decay, 0.05, 0.85)
        self.cortisol = _clip(new["C"], 0.05, 0.80)

        # refractory receptor sensitivity
        if self.dopamine < 0.2:
            self._da_sensitivity = min(1.5, self._da_sensitivity + 0.02)
        elif self.dopamine > 0.6:
            self._da_sensitivity = max(0.5, self._da_sensitivity - 0.01)
        else:
            self._da_sensitivity += 0.005 * (1.0 - self._da_sensitivity)
        if self.noradrenaline < 0.15:
            self._ne_sensitivity = min(1.5, self._ne_sensitivity + 0.02)
        elif self.noradrenaline > 0.6:
            self._ne_sensitivity = max(0.5, self._ne_sensitivity - 0.01)
        else:
            self._ne_sensitivity += 0.005 * (1.0 - self._ne_sensitivity)

        # ── the slow loop: tolerance, and what abstinence gives back ────────
        # Separate from the refractory block above and on a different clock. A
        # big reward costs receptor sensitivity that the next one then has to
        # push through, so the SAME event delivers less each time; staying away
        # from it long enough brings them back, which is why a level with no
        # stars in it pulls him out of the hole rather than deepening it.
        if reward > 0:
            self._da_tolerance = max(self.DA_MIN_TOLERANCE,
                                     self._da_tolerance - self.TOLERANCE_RATE * reward)
            self._since_reward = 0
            self.rewards_taken += 1
        else:
            self._since_reward += 1
            if self._since_reward > self.ABSTINENCE_AFTER:
                self._da_tolerance = min(1.0, self._da_tolerance + self.RECOVERY_RATE)

        # pain is felt, not merely recorded: it drags the reported valence down
        # for as long as it lasts, so what he SAYS during a bad stretch reads
        # like someone having a bad stretch
        self.valence = _clip(valence - self.pain, -1, 1)
        self.affect_arousal = _clip(novelty + threat + surge + self.pain
                                    + self.noradrenaline * 0.3, 0, 1)
        self.stress = self.cortisol
        self.dominant_emotion = self._classify_emotion(novelty)

    def _classify_emotion(self, novelty: float) -> str:
        """Lövheim cube: nearest (5-HT, DA, NE) corner; curious/neutral overrides.

        The coordinates are each hormone's position BETWEEN ITS OWN QUIESCENT
        LEVEL AND ITS BAND EDGE, not its raw concentration. The corners are at
        0 and 1, and raw concentrations never go near either — noradrenaline
        lives around 0.15 in a band topping out at 0.90, so the raw point sat
        permanently in the low-NE half of the cube and no amount of threat
        could carry it to an `anxious` or `angry` corner. Scaled, quiescent is
        the middle of the cube, every corner is reachable, and a body doing
        nothing sits at the centre and is called neutral — which is the reading
        the `intensity < 0.15` test was always meant to produce and could not,
        because the raw point is never near the centre either."""
        corners = {"joy": (1, 1, 1), "warm": (1, 1, 0), "surprise": (1, 0, 1),
                   "shame": (1, 0, 0), "angry": (0, 1, 1), "contempt": (0, 1, 0),
                   "anxious": (0, 0, 1), "sad": (0, 0, 0)}
        base = self.quiescent()
        pos = []
        for key, now in (("5HT", self.serotonin), ("DA", self.dopamine), ("NE", self.noradrenaline)):
            rest, (lo, hi) = base[key], self._BAND[key]
            span = (hi - rest) if now >= rest else (rest - lo)
            pos.append(_clip(0.5 + 0.5 * (now - rest) / max(span, 1e-6), 0, 1))
        pos = tuple(pos)
        best, best_d = "neutral", float("inf")
        for emotion, corner in corners.items():
            d = sum((p - c) ** 2 for p, c in zip(pos, corner)) ** 0.5
            if d < best_d:
                best, best_d = emotion, d
        intensity = sum((p - 0.5) ** 2 for p in pos) ** 0.5
        if novelty > 0.5 and pos[2] > 0.6:
            return "curious"
        if intensity < 0.15:
            return "neutral"
        return best

    # ── message-level convenience ───────────────────────────────────────────
    def step_message(self, signals: dict, frames: int = 2) -> dict:
        """Run `frames` perception frames on one message's appraisal so a
        burst registers within a single chat turn, then return to_dict()."""
        for _ in range(frames):
            self.update(**signals)
        return self.to_dict()

    # ── modulation (same readout the SNN/routing layers use in cubemind) ────
    def modulate_threshold(self, base_threshold: float) -> float:
        """DA lowers the threshold (excited), cortisol raises it (defensive),
        NE lowers it (alert). In the serve brain this scales the ROUTING
        threshold: a stressed Cubby demands better evidence before engaging
        the reasoning cortex — caution changes, facts never do."""
        return base_threshold * (1.0 - 0.25 * (self.dopamine - 0.4)
                                 - 0.15 * (self.noradrenaline - 0.25)
                                 + 0.15 * self.cortisol)

    def modulate_tau(self, base_tau: float) -> float:
        """5-HT stabilizes (higher tau), cortisol and NE speed up."""
        return base_tau * (1.0 + 0.25 * (self.serotonin - 0.5)
                           - 0.3 * (self.cortisol - 0.2)
                           - 0.15 * (self.noradrenaline - 0.25))

    @property
    def arousal(self) -> float:
        return _clip(0.3 * self.noradrenaline + 0.25 * self.cortisol
                     + 0.25 * self.dopamine + 0.2 * (1 - self.serotonin), 0, 1)

    @property
    def weight(self) -> float:
        """Hartmann valence-as-weight: DA/(DA+C)."""
        return self.dopamine / (self.dopamine + self.cortisol + 1e-8)

    @property
    def tolerance(self) -> float:
        """0 = a naive receptor, 1 = as downregulated as it gets. How much of
        the reward the same event no longer delivers."""
        span = 1.0 - self.DA_MIN_TOLERANCE
        return _clip((1.0 - self._da_tolerance) / span, 0, 1)

    @property
    def craving(self) -> float:
        """Wanting, which is not the same as liking, and is the part that
        changes behaviour.

        It needs BOTH terms. Tolerance alone is a receptor fact he could carry
        around indefinitely without it meaning anything; the shortfall alone is
        just a quiet afternoon. Wanting is the gap a downregulated receptor
        opens, *felt in the moment he is not getting it* — which is why it
        climbs during the stretch after the last one and collapses the instant
        he gets another.

        The shortfall is measured against where dopamine ACTUALLY sits when
        nothing is happening, not against `_resting`. Owner, 2026-09-15, on the
        null result in exp_r39: *"it was having no addiction because it was
        stoned all the time high on oxytocin and dopamin."* Exactly that — the
        declared resting level is 0.30 and the ODE's real fixed point is 0.62,
        so `_resting["DA"] - dopamine` was negative on every step of every run
        and clipped to zero. The felt half of wanting was pinned at zero by
        arithmetic, and no amount of ghost-eating could have moved it."""
        base = self.quiescent()["DA"]
        short = _clip((base - self.dopamine) / max(base - self._BAND["DA"][0], 1e-6), 0, 1)
        return _clip(self.tolerance * (0.35 + 0.65 * short), 0, 1)

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

    def _dev(self) -> dict:
        """How far each hormone sits from quiescent, as a fraction of the room
        it has IN THAT DIRECTION.

        Asymmetric on purpose. Dopamine settles near 0.30 in a band of
        0.15–0.85, so it has ~0.15 of room below and ~0.55 above; dividing both
        by one span made "low dopamine" mean *below 0.195* — a hair off the
        floor and effectively unreachable, which is why five failed levels in a
        row never once registered as low."""
        base, out = self.quiescent(), {}
        for key, now in (("DA", self.dopamine), ("5HT", self.serotonin), ("NE", self.noradrenaline),
                         ("OT", self.oxytocin), ("C", self.cortisol)):
            rest = base[key]
            lo, hi = self._BAND[key]
            span = (hi - rest) if now >= rest else (rest - lo)
            out[key] = _clip((now - rest) / max(span, 1e-6), -1, 1)
        return out

    MIN_FELT = 0.035              # a hormone has to actually move to be felt

    def body(self, limit: int = 3) -> list[str]:
        """What it is like in here, strongest first. No emotion words.

        A sensation needs a real move behind it, in RAW units as well as
        normalized ones. Noradrenaline rests at 0.15 against a floor of 0.05,
        so with asymmetric scaling a two-hundredths dip reads as fully low and
        a body doing nothing at all came out *"slow, heavy-limbed"*."""
        d, base = self._dev(), self.quiescent()
        raw = {"DA": self.dopamine, "5HT": self.serotonin, "NE": self.noradrenaline,
               "OT": self.oxytocin, "C": self.cortisol}
        for key in list(d):
            if abs(raw[key] - base[key]) < self.MIN_FELT:
                d[key] = 0.0
        felt = [
            (self.pain, "still hurting" if self.pain > 0.5 else "still sore"),
            (d["NE"], "wired, everything loud"),
            (-d["NE"], "slow, heavy-limbed"),
            (-d["DA"], "nothing much feels worth the walk"),
            (d["DA"], "everything looks worth a go"),
            (d["C"], "wound tight, and it will not let go"),
            (-d["5HT"], "thin-skinned"),
            (d["5HT"], "settled"),
            (d["OT"], "warm toward things"),
            (self.craving, "wanting something I am not getting"),
        ]
        return [w for v, w in sorted(felt, key=lambda t: -t[0]) if v > 0.22][:limit]

    # A region of hormone space -> names that FIT it. Opposed readings live in
    # the same list on purpose: the chemistry says how the body is, never what
    # to call it. Each list is a real fork — despair and stubbornness are the
    # same body read two ways, and so are elation and recklessness.
    #
    # THE ORDER IS LOAD-BEARING: `could_be` takes the first hot region, so this
    # is a priority list, not a lookup table. Pain first, because it is an
    # explicit signal rather than an inference and a body in pain is not also
    # having a nice time. Then the TWO-HORMONE conjunctions, then the
    # single-signal readings — a conjunction is more diagnostic than one
    # hormone moving, so it should win when both fit. Ordering by pleasantness
    # instead put "rawness or irritation" ahead of elation on a serotonin dip
    # that a noradrenaline surge had caused, and eating a power star came out
    # as irritation.
    _NAMES: list[tuple[str, list[str]]] = [
        ("hurt",        ["shock", "fear", "anger"]),
        ("low_da_hi_c", ["despair", "stubbornness", "grim determination", "being fed up"]),
        ("hi_ne_lo_da", ["dread", "being rattled", "nerve"]),
        # High dopamine with high noradrenaline is TWO different states and
        # serotonin is what separates them — Lövheim puts anger at (low 5HT,
        # high DA, high NE) and joy at (high 5HT, high DA, high NE). Collapsing
        # them into one region let the compass say *furious* while the names on
        # offer said *elation*, and the model duly reported having a great time
        # in a rage. Two classifiers over one hormone space have to agree about
        # the space.
        ("hi_da_hi_ne_lo_5ht", ["anger", "fury", "being pushed too far"]),
        ("hi_da_hi_ne", ["elation", "recklessness", "being on a roll"]),
        ("hi_da_lo_ne", ["contentment", "ease", "quiet satisfaction"]),
        ("craving",     ["wanting", "restlessness", "an itch"]),
        ("lo_5ht",      ["rawness", "irritation"]),
        ("hi_ot",       ["warmth", "trust"]),
    ]

    # The Lövheim corner this state is nearest -> the honest readings OF THAT
    # CORNER. Keyed off `dominant_emotion` rather than off a second set of
    # thresholds, because two classifiers over one space disagree in the band
    # between them however carefully each is tuned: the corner classifier is
    # nearest-corner and the thresholds were cutoffs, so the compass said
    # `annoyance` while the names said `elation` on the same step. One
    # classifier decides WHERE he is; this table only decides what that place
    # can honestly be called, and the fork inside each list is the part worth
    # handing over.
    #
    # The two social corners are empty on purpose: contempt and shame need
    # somebody else, and there is nobody else in a maze. Same call
    # `pacman._SOCIAL_CORNERS` makes.
    _BY_CORNER: dict[str, list[str]] = {
        "joy":      ["elation", "being on a roll", "recklessness"],
        "warm":     ["warmth", "ease", "quiet satisfaction"],
        "surprise": ["surprise", "being caught out", "confusion"],
        "angry":    ["anger", "fury", "being pushed too far"],
        "anxious":  ["dread", "being rattled", "nerve"],
        "sad":      ["despair", "stubbornness", "being fed up"],
        "curious":  ["curiosity", "interest"],
        "shame":    [],
        "contempt": [],
        "neutral":  [],
    }

    # ── Layer 1: the somatic drives the cube has no axis for ───────────────
    #
    # Pain and craving are not cognitive states sitting somewhere in monoamine
    # space; they are homeostatic imperatives on their own circuits, and the
    # Lövheim cube reads 5-HT, DA and NE only. Forced onto those three axes
    # they get swallowed — a hurting body lands wherever high NE and low 5-HT
    # put it, which is the anger corner. The corner is doing its job on the
    # evidence it has, and the evidence does not contain the thing that matters.
    #
    # So they do two things, not one. They SKEW the monoamines (below, through
    # the drives, so the skew integrates and decays like everything else rather
    # than being pasted onto the vector at read time — that is what makes pain
    # linger instead of ending with the blow). And above a threshold they
    # INTERCEPT the naming, because past a point "he is in pain" is the true
    # thing to say about him whatever corner the three amines imply.
    #
    # The influence vectors are named rather than scattered through the drive
    # expressions so they can be read, argued with and tested as a unit.
    #                       5-HT    DA     NE
    PAIN_INFLUENCE =      (-0.25, -0.30, +0.60)   # vulnerability, anti-reward, hyper-arousal
    CRAVING_INFLUENCE =   (-0.10,  0.00, +0.30)   # frustration, searching alertness
    #
    # CRAVING'S DOPAMINE TERM IS DELIBERATELY ZERO, against the obvious reading
    # of incentive salience. Wanting IS mesolimbic dopamine — but the phasic
    # burst to a cue, not the tonic level, and `craving` here is DERIVED from
    # the tonic shortfall (`tolerance` x how far dopamine sits below quiescent).
    # A positive term would therefore close the very gap that produced it: DA
    # rises, the shortfall shrinks, craving collapses, DA falls, craving
    # returns — an oscillator that erases the withdrawal this models. The
    # phasic half is real and belongs on a CUE (a frightened ghost actually in
    # sight), not on the baseline; see `pacman.CubbyGhost.chasing`, which is
    # where a cue exists. Wiring it here would be a feedback loop wearing the
    # costume of neuroscience.
    PAIN_INTERCEPT, AGONY = 0.35, 0.70
    CRAVING_INTERCEPT, FRANTIC = 0.30, 0.65
    SOMATIC_NAMES = {
        "pain":    ["it hurts", "shock", "fear"],
        "agony":   ["agony", "panic", "rage"],
        "craving": ["wanting", "restlessness", "an itch"],
        "frantic": ["need", "a craving I can't sit with", "desperation"],
    }

    def somatic(self) -> list[str]:
        """Layer 1. The readings that outrank the cube, or [] when neither fires.

        Graded, because a scraped knee and a broken leg are not the same word:
        past the second threshold the name changes rather than just getting
        louder. Pain wins ties — a body that is both hurting and wanting has a
        more urgent problem than the wanting."""
        if self.pain >= self.PAIN_INTERCEPT and self.pain >= self.craving:
            return list(self.SOMATIC_NAMES["agony" if self.pain > self.AGONY else "pain"])
        if self.craving >= self.CRAVING_INTERCEPT:
            return list(self.SOMATIC_NAMES["frantic" if self.craving > self.FRANTIC else "craving"])
        return []

    def cube(self) -> tuple:
        """(5-HT, DA, NE) scaled so QUIESCENT IS THE CENTRE of the cube.

        Raw concentrations never approach 0 or 1 — noradrenaline lives near
        0.15 in a band topping out at 0.90 — so the raw point sits permanently
        in one half of the cube and the corners on the other side are
        unreachable by construction. That is why he could not be anxious."""
        base, out = self.quiescent(), []
        for key, now in (("5HT", self.serotonin), ("DA", self.dopamine), ("NE", self.noradrenaline)):
            rest, (lo, hi) = base[key], self._BAND[key]
            span = (hi - rest) if now >= rest else (rest - lo)
            out.append(_clip(0.5 + 0.5 * (now - rest) / max(span, 1e-6), 0, 1))
        return tuple(out)

    _CORNERS = {"joy": (1, 1, 1), "warm": (1, 1, 0), "surprise": (1, 0, 1),
                "shame": (1, 0, 0), "angry": (0, 1, 1), "contempt": (0, 1, 0),
                "anxious": (0, 0, 1), "sad": (0, 0, 0)}

    # ── Layer 2: where in the cube, how far out, and what he is near ────────
    def corner_position(self) -> dict:
        """The geometry, with no names attached — naming is Layer 3's job.

        `intensity` is how far out toward the corner he actually is, which is
        what separates annoyance from rage inside one octant: the corner says
        WHICH family, the distance says how much of it. `runner_up` and
        `margin` are the boundary case — a point sitting between two corners
        should not be described as squarely in either, and Plutchik already has
        a name for the pair.

        Nearest-corner, so the eight cells meet at the centre and every corner
        is reachable. A threshold classifier over the same space carves unequal
        boxes and disagrees with this one in the band between them, which is
        exactly the bug that put `annoyance` on the compass and `elation` in
        the names on the same step."""
        pos = self.cube()
        def dist(c):
            return sum((p - q) ** 2 for p, q in zip(pos, c)) ** 0.5
        ranked = sorted(self._CORNERS.items(), key=lambda kv: dist(kv[1]))
        best, second = ranked[0][0], ranked[1][0]
        reach = sum((p - 0.5) ** 2 for p in pos) ** 0.5 / (0.75 ** 0.5)
        return {"corner": best, "intensity": _clip(reach, 0, 1), "runner_up": second,
                "margin": dist(self._CORNERS[second]) - dist(self._CORNERS[best]),
                "pos": pos}

    def could_be(self, limit: int = 3) -> list[str]:
        """Layer 1, then a default Layer 3 for callers with no naming table.

        The game overrides the naming with the Plutchik one (`pacman.felt_names`),
        which reads the intensity tier and the boundary dyad. This keeps a
        sensible answer for ToyVerse and the tests, and it is the same corner
        either way — only the vocabulary differs."""
        somatic = self.somatic()
        if somatic:
            return somatic[:limit]
        return list(self._BY_CORNER.get(self.dominant_emotion, [])[:limit])

    # ── the dials: hormones -> HOW he speaks, with no feeling word anywhere ──
    #
    # Ported from GrillCheese's `EndocrineSystem.get_modulation_factors`
    # (backend/brain/endocrine.py), which is the piece this port was missing.
    # That system never hands the generator a label either; it hands it
    # continuous knobs — creativity, caution, warmth, energy, stability, focus
    # — and lets the voice fall out of them. Two additions here, for the two
    # signals GrillCheese's version has no input for: pain and craving.
    #
    # This is the answer to *"it still sounds a bit too artificial"*. A model
    # told "you feel anxious" says the word anxious. A model whose sentence is
    # generated short, clipped and wary SOUNDS anxious without the word
    # appearing anywhere — which is what feeling something looks like from
    # outside, and it cannot be faked by copying a label out of a record.
    def modulation(self) -> dict:
        """Continuous 0..1 knobs on the voice. No emotion words, by design."""
        da, ne = self.dopamine, self.noradrenaline
        sht, ot, c = self.serotonin, self.oxytocin, self.cortisol
        hurt, want = self.pain, self.craving
        return {
            "creativity": _clip(da - c * 0.5 + 0.3 - hurt * 0.3, 0, 1),
            "caution": _clip(c * 1.5 + hurt * 0.4, 0, 1),
            "warmth": _clip(ot * 1.2 + sht * 0.3 - hurt * 0.5, 0, 1),
            "energy": _clip(ne * 1.3 + da * 0.2, 0, 1),
            "stability": _clip(sht * 1.2 - c * 0.3 - hurt * 0.4, 0, 1),
            "focus": _clip(ne * 0.8 + c * 0.2 + want * 0.3, 0, 1),
            "urgency": _clip(ne * 0.6 + hurt * 0.6 + want * 0.4, 0, 1),
        }

    # each dial, at its extremes, in terms of MANNER — how a person talks, not
    # how they feel. The speaker is never told what is happening to it.
    _MANNER = {
        "caution":   (0.62, "measured, and not committing to much", 0.18, None),
        "energy":    (0.55, "quick, a bit clipped", 0.20, "slow, not much left in it"),
        "warmth":    (0.55, "open", 0.15, "flat, keeping its distance"),
        "stability": (0.18, None, 0.62, None),
        "urgency":   (0.60, "short — no time to round it off", 0.15, None),
    }

    def manner(self, limit: int = 2) -> str:
        """The dials as a short instruction about DELIVERY, for the prompt.

        Only the extremes speak: a dial sitting mid-range says nothing, because
        a voice note for every knob on every turn is a style sheet, and a style
        sheet read aloud is exactly the artificial thing this replaces."""
        m, out = self.modulation(), []
        for key, (hi, hi_word, lo, lo_word) in self._MANNER.items():
            v = m[key]
            if hi_word and v >= hi:
                out.append((v - hi, hi_word))
            elif lo_word and v <= lo:
                out.append((lo - v, lo_word))
        out.sort(key=lambda t: -t[0])
        return ", ".join(w for _, w in out[:limit])

    def to_dict(self) -> dict:
        return {"cortisol": self.cortisol, "dopamine": self.dopamine,
                "serotonin": self.serotonin, "oxytocin": self.oxytocin,
                "noradrenaline": self.noradrenaline, "valence": self.valence,
                "arousal": self.arousal, "stress": self.stress, "weight": self.weight,
                "da_sensitivity": self._da_sensitivity, "emotion": self.dominant_emotion,
                "pain": self.pain, "tolerance": self.tolerance, "craving": self.craving,
                "rewards_taken": self.rewards_taken}


# ── the host's lexical appraisal (EN/FR) ────────────────────────────────────
_THREAT = re.compile(r"\b(urgent|emergency|danger|help|asap|hurry|now|broken|crash|"
                     r"urgence|danger|aide|vite|casse|panne|au secours)\b", re.I)
_SOCIAL = re.compile(r"\b(thanks|thank you|please|hi|hello|hey|welcome|friend|"
                     r"merci|s'il te pla[iî]t|bonjour|salut|coucou|bienvenue|ami)\b", re.I)
_POS = re.compile(r"\b(great|good|love|cool|awesome|nice|perfect|fun|happy|thanks?|thank you|"
                  r"super|g[ée]nial|bravo|parfait|chouette|content|heureux|merci)\b|:\)|<3", re.I)
_NEG = re.compile(r"\b(bad|wrong|hate|angry|terrible|awful|stupid|sad|"
                  r"nul|mauvais|faux|d[ée]teste|f[âa]ch[ée]|triste|horrible)\b|:\(", re.I)
_FOCUS = re.compile(r"\?|\b(how|what|why|when|where|which|explain|calculate|solve|"
                    r"comment|quoi|pourquoi|quand|o[ùu]|quel|explique|calcule|r[ée]sous)\b", re.I)


def _hits(rx: re.Pattern, text: str, per: float, cap: float) -> float:
    return min(cap, per * len(rx.findall(text)))


def appraise(text: str, seen: set[str]) -> dict:
    """One message -> the ODE's five drive inputs. Lexical and inspectable:
    novelty is the unseen-token fraction against the session vocabulary
    (halved, so first contact reads as new without drowning the rest);
    shouting (caps, !!) reads as threat/urgency alongside the lexicons."""
    tokens = re.findall(r"[\w']+", text.lower())
    unseen = sum(1 for t in tokens if t not in seen) / max(1, len(tokens))
    shouting = 0.4 if (text.isupper() and len(text) > 3) else 0.0
    bangs = min(0.3, 0.15 * text.count("!!"))
    valence = _hits(_POS, text, 0.4, 1.0) - _hits(_NEG, text, 0.4, 1.0)
    return {"novelty": round(0.5 * unseen, 3),
            "threat": round(min(1.0, _hits(_THREAT, text, 0.35, 0.8) + shouting + bangs), 3),
            "focus": round(_hits(_FOCUS, text, 0.3, 0.8), 3),
            "valence": round(_clip(valence, -1, 1), 3),
            "social": round(_hits(_SOCIAL, text, 0.4, 1.0), 3)}
