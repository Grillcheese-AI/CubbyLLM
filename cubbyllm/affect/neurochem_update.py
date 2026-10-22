"""cubbyllm/affect/neurochem_update.py -- a step of the chemistry: the ODE update from an appraisal, the emotion it
reads, the step's message, and the threshold and tau it modulates. Split out of neurochem.py (order 0031) as a mixin
of Neurochemistry, verbatim.

Wired: WIRED (Neurochemistry).
"""
from __future__ import annotations

from .neurochem_appraise import _clip, _sigmoid

__wiring__ = "WIRED"


class NeuroUpdate:
    """Neurochemistry's step (a mixin; order 0031)."""

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
        # The corner table lived here TWICE - once as `self._CORNERS` and once
        # hard-coded in this function - and both copies were wrong in the same
        # way, which is exactly how a duplicated constant fails: nothing
        # disagrees, so nothing is detected. One table now, and `cube()` does
        # the scaling both callers were doing separately.
        pos = self.cube()
        # Sorted, so an exact tie resolves the same way every run. It used to
        # resolve by dict insertion order, which is not a decision anybody made:
        # a discovery frame lands at (0.50, 1.00, 0.79), exactly equidistant from
        # anger (0,1,1) and interest (1,1,1), and whichever key happened to be
        # written first won. A tie means the state is genuinely BETWEEN two
        # corners; `corner_position()` already returns `margin` for that, and
        # Layer 3 reads a small margin as a Plutchik dyad. This function has to
        # return one name, so it returns a stable one rather than an accidental
        # one - the honest handling lives where the margin is visible.
        best, best_d = "neutral", float("inf")
        for emotion in sorted(self._CORNERS):
            corner = self._CORNERS[emotion]
            d = sum((p - c) ** 2 for p, c in zip(pos, corner)) ** 0.5
            if d < best_d:
                best, best_d = emotion, d
        intensity = sum((p - 0.5) ** 2 for p in pos) ** 0.5
        # The novelty override is GONE. `if novelty > 0.5 and pos[2] > 0.6:
        # return "curious"` was a second classifier over the same space - the
        # thing this architecture was rebuilt to eliminate - and it existed only
        # because the interest/excitement vertex was mislabelled `joy`, leaving
        # curiosity with nowhere to land. With (1,1,1) named `interest`, high
        # novelty and high NE reaches it through the geometry, as it always
        # should have. `novelty` stays in the signature: it drives the ODE
        # upstream, which is the only place it belongs.
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
