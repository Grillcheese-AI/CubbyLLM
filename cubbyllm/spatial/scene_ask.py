"""cubbyllm/spatial/scene_ask.py -- three question shapes a mounted scene answers that no emitter grammar names (order
0035, part A): "where is X?", "what is n <direction> of X?" and "what is at x y z?".

The host reads them off the question itself, as it reads "who / what / where is X" for a profile, and restates each in
the scene's own grammar ("what is the R of S?"), so the walk, the plan check and the VM treat it as they treat the
four shapes that already reached the scene:

    Where is the forge?          -> What is the position of forge?      (position)
    What is 3 east of the well?  -> What is the 3 east of well?         ("<n> <direction>"; n = 1 is "east neighbour")
    What is at 10 9 0?           -> What is the occupant of 10 9 0?     (occupant)

A name the scene does not hold is no scene question (the loop goes on as before: "Where is Paris?" stays a profile
ask). The scene's facts (``Scene.facts``) state the offsets and the occupants from its exact reading, as they state the
neighbours; an empty cell states nothing, so the walk refuses.

Wired: WIRED (``reasoning.ask.AskLoop._ask``, when a scene is mounted).
"""
from __future__ import annotations

import re

from .scene import DIRECTIONS, _norm

__wiring__ = "WIRED"

_DIRS = "|".join(DIRECTIONS)
WHERE = re.compile(r"^\s*where(?:'s| is| are)\s+(?P<x>.+?)\s*\??\s*$", re.I)
OFFSET = re.compile(rf"^\s*what(?:'s| is)\s+(?P<n>\d+)\s+(?:(?:steps?|cells?|blocks?)\s+)?(?P<d>{_DIRS})\s+of\s+(?P<x>.+?)\s*\??\s*$",
                    re.I)
AT = re.compile(r"^\s*what(?:'s| is)\s+at\s+(?P<c>-?\d+[ ,]+-?\d+[ ,]+-?\d+)\s*\??\s*$", re.I)


def scene_ask(question: str, scene) -> tuple[str, str, str] | None:
    """(the question restated, the relation, the entity) for a shape the scene answers; None otherwise."""
    q = " ".join((question or "").split())
    m = AT.match(q)
    if m:
        cell = " ".join(re.split(r"[ ,]+", m["c"].strip()))
        return f"What is the occupant of {cell}?", "occupant", cell
    m = OFFSET.match(q)
    if m and scene.where(m["x"]) is not None:
        n, d, x = int(m["n"]), m["d"].lower(), _norm(m["x"])
        rel = f"{d} neighbour" if n == 1 else f"{n} {d}"
        return f"What is the {rel} of {x}?", rel, x
    m = WHERE.match(q)
    if m and scene.where(m["x"]) is not None:
        x = _norm(m["x"])
        return f"What is the position of {x}?", "position", x
    return None
