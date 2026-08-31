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
        self.dopamine = self._resting["DA"]
        self.serotonin = self._resting["5HT"]
        self.noradrenaline = self._resting["NE"]
        self.oxytocin = self._resting["OT"]
        self.cortisol = self._resting["C"]
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

    # ── core ODE update (one perception frame) ──────────────────────────────
    def update(self, novelty: float = 0.0, threat: float = 0.0, focus: float = 0.0,
               valence: float = 0.0, social: float = 0.0) -> None:
        joy = max(0.0, valence)
        neg = max(0.0, -valence)

        drive_NE = (novelty * 0.5 + threat * 0.4 + focus * 0.4) * self._ne_sensitivity
        drive_DA = (novelty * 0.5 + joy * 0.4 + 0.15) * self._da_sensitivity
        drive_5HT = 0.35 + 0.25 * joy - 0.2 * threat - 0.1 * neg
        drive_OT = social * 0.5 + joy * 0.2 + 0.05

        drives = {"DA": _clip(drive_DA, 0, 1), "5HT": _clip(drive_5HT, 0, 1),
                  "NE": _clip(drive_NE, 0, 1), "OT": _clip(drive_OT, 0, 1)}
        values = {"DA": self.dopamine, "5HT": self.serotonin,
                  "NE": self.noradrenaline, "OT": self.oxytocin}

        new = {}
        for h in ("DA", "NE", "5HT", "OT"):
            delta = self._alpha[h] * drives[h] - self._beta[h] * (values[h] - self._resting[h])
            new[h] = values[h] + self._dt * delta

        # cortisol: slow EMA of arousal (the HPA cascade, minutes timescale)
        instantaneous_stress = _clip(new["NE"] * 0.5 + threat * 0.3 + neg * 0.2, 0, 1)
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

        self.valence = _clip(valence, -1, 1)
        self.affect_arousal = _clip(novelty + threat + self.noradrenaline * 0.3, 0, 1)
        self.stress = self.cortisol
        self.dominant_emotion = self._classify_emotion(novelty)

    def _classify_emotion(self, novelty: float) -> str:
        """Lövheim cube: nearest (5-HT, DA, NE) corner; curious/neutral overrides."""
        corners = {"joy": (1, 1, 1), "warm": (1, 1, 0), "surprise": (1, 0, 1),
                   "shame": (1, 0, 0), "angry": (0, 1, 1), "contempt": (0, 1, 0),
                   "anxious": (0, 0, 1), "sad": (0, 0, 0)}
        pos = (self.serotonin, self.dopamine, self.noradrenaline)
        best, best_d = "neutral", float("inf")
        for emotion, corner in corners.items():
            d = sum((p - c) ** 2 for p, c in zip(pos, corner)) ** 0.5
            if d < best_d:
                best, best_d = emotion, d
        intensity = sum((p - 0.5) ** 2 for p in pos) ** 0.5
        if novelty > 0.5 and self.noradrenaline > 0.3:
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

    def to_dict(self) -> dict:
        return {"cortisol": self.cortisol, "dopamine": self.dopamine,
                "serotonin": self.serotonin, "oxytocin": self.oxytocin,
                "noradrenaline": self.noradrenaline, "valence": self.valence,
                "arousal": self.arousal, "stress": self.stress, "weight": self.weight,
                "da_sensitivity": self._da_sensitivity, "emotion": self.dominant_emotion}


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
