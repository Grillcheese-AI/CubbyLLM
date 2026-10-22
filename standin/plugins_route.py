"""standin/plugins_route.py -- which plugin a question would wake (order 0028).

Each plugin's ``description`` and ``examples`` are one line each in a ``worlds.FactStore``, the store worlds are routed
with: the brain's encoder when one is given (unit cosine), its IDF-weighted word overlap otherwise. A question's score
for a plugin is its best line's; a plugin whose score is at least ``tau`` wakes. ``tau`` started at ``Worlds.TAU``
(0.35) and is tuned on the plugins' ``examples`` / ``not`` lines, written first (the routing test:
``validation/plugins_routing.py``): 0.4 with the serve stack's encoder, 0.475 with word overlap alone. A disabled plugin never wakes; a pinned one is always loaded and needs no score.

Wired: WIRED (``plugins_load.Loader.wake``).
"""
from __future__ import annotations

import sys
from pathlib import Path

__wiring__ = "WIRED"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worlds import FactStore, Worlds  # noqa: E402

TAU_START = Worlds.TAU               # where the tuning started (0.35)
# tuned on the four proof plugins' examples / not lines (validation/logs/plugins_routing_enc_2026-10-22.json and
# plugins_routing_2026-10-22.json): the lowest tau where no plugin's not-lines wake it more than once in ten
TAU_ENCODER, TAU_OVERLAP = 0.4, 0.475
TAU_PLUGIN = TAU_ENCODER


class Router:
    def __init__(self, index, enc=None, tau: float = TAU_PLUGIN, leave_out: str | None = None) -> None:
        """``leave_out``: a line not to index (the routing test's leave-one-out: an example scored against the others)."""
        self.tau = float(tau)
        self.owner: dict[str, str] = {}
        lines = []
        for p in index.plugins.values():
            for line in [p.description] + list(p.examples):
                key = " ".join(line.split())
                if key and key != leave_out and key not in self.owner:
                    self.owner[key] = p.name
                    lines.append(key)
        self.store = FactStore(lines, enc=enc, name="plugins")

    def scores(self, question: str, k: int = 32) -> dict[str, float]:
        """Each plugin's best line score for ``question`` (plugins with no matching line are left out)."""
        out: dict[str, float] = {}
        for s, line in self.store(question, k):
            name = self.owner.get(" ".join(line.split()))
            if name is not None and s > out.get(name, 0.0):
                out[name] = float(s)
        return out

    def wake(self, question: str, disabled=()) -> list[tuple[str, float]]:
        """The plugins this question wakes, best first."""
        got = [(n, s) for n, s in self.scores(question).items() if s >= self.tau and n not in disabled]
        return sorted(got, key=lambda x: (-x[1], x[0]))
