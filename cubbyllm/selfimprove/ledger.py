"""cubbyllm/selfimprove/ledger.py -- each request's signature and outcome (order 0049, step 1).

An outcome is the VM's verdict ("right": spoken and verified, "refused", "caught": a wrong program the VM caught) and,
where the record has a gold, whether the spoken answer matched it (``gold_right``). A failure, for the gap finder, is
anything but a verified answer that matches its gold (a refusal included: a skill the trunk lacks shows as refusals).
The ledger keeps the last ``window`` requests; ``failures`` counts them per signature.

Wired: STANDALONE (order 0049).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

__wiring__ = "STANDALONE"

VERDICTS = ("right", "refused", "caught")


@dataclass
class Entry:
    key: str
    verdict: str
    gold_right: bool | None = None
    request: str = ""
    sig_from: str = "reference"

    def failed(self) -> bool:
        return not (self.verdict == "right" and self.gold_right is not False)


@dataclass
class Ledger:
    window: int = 2000
    entries: collections.deque = field(default_factory=collections.deque)
    seen: int = 0

    def add(self, key: str, verdict: str, gold_right: bool | None = None, request: str = "",
            sig_from: str = "reference") -> Entry:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict {verdict!r}: one of {VERDICTS}")
        e = Entry(key, verdict, gold_right, request, sig_from)
        self.entries.append(e)
        self.seen += 1
        while len(self.entries) > self.window:
            self.entries.popleft()
        return e

    def failures(self) -> collections.Counter:
        return collections.Counter(e.key for e in self.entries if e.failed())

    def requests(self, key: str, failed_only: bool = False) -> list[str]:
        return [e.request for e in self.entries if e.key == key and (e.failed() or not failed_only)]

    def rate(self, key: str) -> float | None:
        es = [e for e in self.entries if e.key == key]
        return None if not es else sum(not e.failed() for e in es) / len(es)


__all__ = ["Entry", "Ledger", "VERDICTS"]
