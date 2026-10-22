"""cubbyllm/affect/neurochem_signals.py -- what the chemistry signals: arousal, weight, tolerance and craving, a
level's deviation from its band, the body's state. Split out of neurochem.py (order 0031) as a mixin of
Neurochemistry, verbatim, with the constant it reads.

Wired: WIRED (Neurochemistry).
"""
from __future__ import annotations

from .neurochem_appraise import _clip

__wiring__ = "WIRED"


class NeuroSignals:
    """Neurochemistry's signals (a mixin; order 0031)."""

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
