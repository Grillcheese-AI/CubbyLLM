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

        drive_NE = ((novelty * 0.5 + threat * 0.4 + focus * 0.4
                     + surge * 0.7 + self.pain * 0.6) * self._ne_sensitivity)
        # the reward term carries BOTH receptor terms: the fast refractory one
        # and the slow tolerance. Which is the whole mechanism — the world hands
        # him the same ghost every time and he feels less of it each time.
        drive_DA = ((novelty * 0.5 + joy * 0.4 + 0.15) * self._da_sensitivity
                    + reward * 0.9 * self._da_sensitivity * self._da_tolerance
                    - self.pain * 0.3)
        drive_5HT = 0.35 + 0.25 * joy - 0.2 * threat - 0.1 * neg - 0.25 * self.pain
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
        he gets another."""
        short = _clip((self._resting["DA"] - self.dopamine) / self._resting["DA"], 0, 1)
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

    def _dev(self) -> dict:
        """How far each hormone sits from ITS OWN resting level, in units of
        the room it has to move. A raw 0.42 of dopamine means nothing; 0.42
        where this body rests at 0.30 is something."""
        out = {}
        for key, now in (("DA", self.dopamine), ("5HT", self.serotonin), ("NE", self.noradrenaline),
                         ("OT", self.oxytocin), ("C", self.cortisol)):
            rest = self._resting[key]
            span = max(rest, 1.0 - rest)
            out[key] = _clip((now - rest) / span, -1, 1)
        return out

    def body(self, limit: int = 3) -> list[str]:
        """What it is like in here, strongest first. No emotion words."""
        d = self._dev()
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

    # a region of hormone space -> names that FIT it. Opposed readings live in
    # the same list on purpose: the chemistry says how the body is, never what
    # to call it.
    _NAMES: list[tuple[str, list[str]]] = [
        ("hurt",        ["shock", "fear", "anger"]),
        ("low_da_hi_c", ["despair", "stubbornness", "grim determination", "being fed up"]),
        ("hi_ne_lo_da", ["dread", "being rattled", "nerve"]),
        ("hi_da_hi_ne", ["elation", "recklessness", "being on a roll"]),
        ("craving",     ["wanting", "restlessness", "an itch"]),
        ("hi_da_lo_ne", ["contentment", "ease", "quiet satisfaction"]),
        ("lo_5ht",      ["rawness", "irritation"]),
        ("hi_ot",       ["warmth", "trust"]),
    ]

    def could_be(self, limit: int = 3) -> list[str]:
        """Names that suit this mixture — offered, never asserted.

        Where two readings of the same chemistry are both honest they are both
        here, so the sentence that comes out is an interpretation and not a
        lookup. One name per region, so the list reads as a real choice rather
        than a thesaurus."""
        d = self._dev()
        hot = {
            "hurt": self.pain > 0.35,
            "low_da_hi_c": d["DA"] < -0.15 and d["C"] > 0.10,
            "hi_ne_lo_da": d["NE"] > 0.20 and d["DA"] < -0.05,
            "hi_da_hi_ne": d["DA"] > 0.15 and d["NE"] > 0.15,
            "craving": self.craving > 0.30,
            "hi_da_lo_ne": d["DA"] > 0.15 and d["NE"] < 0.05,
            "lo_5ht": d["5HT"] < -0.15,
            "hi_ot": d["OT"] > 0.20,
        }
        flat: list[str] = []
        regions = 0
        for region, names in self._NAMES:
            if not hot.get(region) or regions >= limit:
                continue
            regions += 1
            for n in names:                              # every candidate survives; the speaker chooses
                if n not in flat:
                    flat.append(n)
        return flat[:6]

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
