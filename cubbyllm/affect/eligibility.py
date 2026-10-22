"""cubbyllm/affect/eligibility.py -- the eligibility trace that carries a word's credit to the outcome that comes
later. Split out of word_affect.py (order 0031), verbatim; word_affect.py re-exports it.

Wired: WIRED (the word affect).
"""
from __future__ import annotations

from .word_seed import _clip

__wiring__ = "WIRED"


class Eligibility:
    """What was said recently, and how much of it still counts.

    One trace per message rather than per token, because the message is the
    unit a person actually chose. `strength` starts at 1.0 and decays each
    step; below `FLOOR` it is dropped, which is what stops a lexicon filling up
    with superstition about something said a minute ago.
    """

    DECAY = 0.85                  # per step; ~0.05 after 18 steps
    FLOOR = 0.05
    MAX_HELD = 40                 # a bound, so a chatty player cannot grow it forever

    def __init__(self) -> None:
        self.held: list[dict] = []
        self._step = 0

    def mark(self, text: str, step: int, strength: float = 1.0) -> None:
        """Something was said. It is eligible from now until it decays out."""
        if not (text or "").strip():
            return
        self.decay_to(step)
        self.held.append({"text": text, "trace": _clip(strength, 0, 1), "at": step})
        if len(self.held) > self.MAX_HELD:
            self.held = sorted(self.held, key=lambda h: -h["trace"])[:self.MAX_HELD]

    def decay_to(self, step: int) -> None:
        """Advance the clock. Idempotent for a step already applied, so a
        caller may tick and mark in either order without double-decaying."""
        gap = step - self._step
        if gap <= 0:
            return
        self._step = step
        f = self.DECAY ** gap
        for h in self.held:
            h["trace"] *= f
        self.held = [h for h in self.held if h["trace"] >= self.FLOOR]

    def active(self) -> list[tuple]:
        """[(text, trace)], strongest first."""
        return [(h["text"], h["trace"]) for h in sorted(self.held, key=lambda h: -h["trace"])]

    def clear(self) -> None:
        self.held.clear()

    def __len__(self) -> int:
        return len(self.held)
