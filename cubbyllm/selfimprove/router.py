"""cubbyllm/selfimprove/router.py -- the gate, the router and the lifecycle (order 0049, steps 5-7).

  gate       an adapter is promoted only if its gap's held-out right rate rises over the base adapter's and the base
             battery holds: no battery line falls by more than ``battery_drop`` (1 point; H-A7's learning gate)
  router     a request goes to the promoted adapter whose gap signature it matches, else to the base; every decision
             is logged (the request, its key, the adapter)
  lifecycle  a promoted adapter is "loaded" (routed to) or "latent" (on disk, routed to no more: unused for
             ``latent_after`` requests of its gap); a request of a latent adapter's gap loads it again.
             "shadow" and the nightly merge come with T4 (the next order).

Wired: STANDALONE (order 0049).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__wiring__ = "STANDALONE"

BASE = "base"


@dataclass(frozen=True)
class Gate:
    battery_drop: float = 0.01

    def judge(self, gap_base: float, gap_new: float, battery_base: dict, battery_new: dict) -> tuple[bool, str]:
        """(promoted, why). Rates are shares in [0, 1]; a battery line missing from the new read refuses."""
        if gap_new <= gap_base:
            return False, f"the gap's held-out rate did not rise ({gap_base:.3f} -> {gap_new:.3f})"
        for name, b in battery_base.items():
            if name not in battery_new:
                return False, f"battery line {name} not read"
            if battery_new[name] < b - self.battery_drop:
                return False, f"battery line {name} fell {b:.3f} -> {battery_new[name]:.3f}"
        return True, f"the gap {gap_base:.3f} -> {gap_new:.3f}; the battery held"


@dataclass
class Router:
    latent_after: int = 5000
    adapters: dict = field(default_factory=dict)      # gap key -> {"name", "state", "last": request count}
    decisions: list = field(default_factory=list)
    n: int = 0

    def promote(self, key: str, name: str) -> None:
        self.adapters[key] = {"name": name, "state": "loaded", "last": self.n}

    def route(self, key: str, request: str = "") -> str:
        self.n += 1
        a = self.adapters.get(key)
        if a is None:
            name = BASE
        else:
            a["state"], a["last"], name = "loaded", self.n, a["name"]
        for k, other in self.adapters.items():
            if other["state"] == "loaded" and self.n - other["last"] > self.latent_after:
                other["state"] = "latent"
        self.decisions.append({"request": request, "key": key, "adapter": name})
        return name

    def loaded(self) -> list[str]:
        return sorted(a["name"] for a in self.adapters.values() if a["state"] == "loaded")


__all__ = ["BASE", "Gate", "Router"]
