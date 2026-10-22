"""A Neurochemistry after a few appraised turns, for the board's Mood test."""
from __future__ import annotations

__wiring__ = "STANDALONE"


def chem():
    from neurochem import Neurochemistry, appraise
    c, seen = Neurochemistry(), set()
    for text in ("hi Cubby!", "URGENT the server crashed, help now!!", "thanks, that fixed it"):
        drives = appraise(text, seen)
        seen |= set(text.lower().split())
        c.step(**drives) if hasattr(c, "step") else c.update(**drives)
    return c
