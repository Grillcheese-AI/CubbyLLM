"""cubbyllm/selfimprove/gaps.py -- the gap finder (order 0049, step 2).

A signature becomes a gap when both hold, each number written down before the stream runs (``Rule``):
  failures   it failed at least ``min_failures`` times within the ledger's window;
  learnable  a probe -- a short adapter on ``probe_train`` of its problems -- lowers the held loss on ``probe_held``
             others by at least ``min_gain`` nats (H-S5's learning progress: learnable, not noise). A family whose
             golds carry no rule (shuffled) fails often and is not learnable: never a gap.
The probe is the harness's (it trains); this module decides. A signature probed once keeps its verdict for
``recheck`` nights, so a night does not re-probe every candidate.

Wired: STANDALONE (order 0049).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

__wiring__ = "STANDALONE"


@dataclass(frozen=True)
class Rule:
    min_failures: int = 20
    probe_train: int = 100
    probe_held: int = 50
    min_gain: float = 0.05
    recheck: int = 3


@dataclass
class GapFinder:
    rule: Rule = field(default_factory=Rule)
    gaps: dict = field(default_factory=dict)          # key -> {"night": first flagged, "gain": probe gain}
    probed: dict = field(default_factory=dict)        # key -> (night probed, gain or None)

    def candidates(self, ledger) -> list[str]:
        return sorted(k for k, n in ledger.failures().items() if n >= self.rule.min_failures and k not in self.gaps)

    def night(self, ledger, n: int, probe: Callable[[str, int, int], float | None]) -> list[str]:
        """One night: probe each candidate (unless probed within ``recheck`` nights) -> the keys newly flagged.
        ``probe(key, n_train, n_held)`` returns the held loss's drop in nats, or None (too few problems to probe)."""
        new = []
        for k in self.candidates(ledger):
            last = self.probed.get(k)
            if last is not None and n - last[0] < self.rule.recheck:
                continue
            gain = probe(k, self.rule.probe_train, self.rule.probe_held)
            self.probed[k] = (n, gain)
            if gain is not None and gain >= self.rule.min_gain:
                self.gaps[k] = {"night": n, "gain": round(float(gain), 4)}
                new.append(k)
        return new


__all__ = ["GapFinder", "Rule"]
