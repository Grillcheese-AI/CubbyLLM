"""s12_questions -- order 0034's questions about a scene, fixed before any run: the shapes, how many of each, and how
each is drawn and answered from ``Scene``'s exact reading (the spatial shard's shapes, ``make_spatial_shard.py``).

Per village, 30 questions (``PLAN``):

    position   5   What is the position of X?            -> "x y z"
    where      4   Where is X?  (X a villager when the village has one)  -> "x y z"
    neighbour  5   What is the <direction> neighbour of X?   -> a thing | nothing
    offset     5   What is <n> <direction> of X?  (n 2..4)   -> a thing | nothing
    nearest    4   What is the nearest <kind> of X?      -> a thing | nothing
    region     3   What is the region of X?              -> a region | nothing
    at         4   What is at x y z?                     -> a thing | nothing

The draws lean towards answers that are things, so a rate is not carried by empty cells. Half the neighbour and
offset questions point at a cell that holds something (when one does). Three in four "at" questions point at an
occupied cell, and when the village has villagers, half of those occupied picks are a villager's cell. Region
questions pick a thing inside a region 7 times in 10. Every answer has a class: ``name`` (a villager: open
vocabulary, the copy's), ``kind`` (any other thing: a closed set of words), ``region``, ``position``, ``nothing``.

``address`` is the cell a placed question is about (``PLACED``: neighbour, offset, at), the cell the trunk's
``here`` token carries, as in run 5's shard (``--ask placed``): the host computes it, the trunk reads it.

Wired: STANDALONE (``exp_s12_scene_files.py``).
"""
from __future__ import annotations

import random

from cubbyllm.spatial.scene import DIRECTIONS, Scene

__wiring__ = "STANDALONE"

PLAN = (("position", 5), ("where", 4), ("neighbour", 5), ("offset", 5), ("nearest", 4), ("region", 3), ("at", 4))
PLACED = ("neighbour", "offset", "at")                   # the shapes run 5 was trained on (spatial_here)
VILLAGER = "villager"
TRIES = 20                                               # draws before a repeat is let through


def _cls(sc: Scene, answer: str | None) -> str:
    if answer is None:
        return "nothing"
    return "name" if sc.kinds.get(answer) == VILLAGER else "kind"


def _pointing(sc: Scene, rng: random.Random, names: list, steps) -> tuple[str, str, int]:
    """(thing, direction, n): half the time one whose target cell holds something (a villager half of those, when
    one is reachable); otherwise any."""
    pairs = [(t, d, n) for t in names for d in DIRECTIONS for n in steps if sc.at(sc.offset(t, d, n)) is not None]
    if pairs and rng.random() < 0.5:
        people = [p for p in pairs if sc.kinds.get(sc.at(sc.offset(p[0], p[1], p[2]))) == VILLAGER]
        return rng.choice(people if people and rng.random() < 0.5 else pairs)
    return rng.choice(names), rng.choice(list(DIRECTIONS)), rng.choice(list(steps))


def _box(sc: Scene):
    cs = list(sc.cells)
    return [min(c[i] for c in cs) for i in range(3)], [max(c[i] for c in cs) for i in range(3)]


def _draw(sc: Scene, rng: random.Random, shape: str, names: list, people: list, kinds: list, lo, hi) -> dict:
    """One question of ``shape``."""
    q = {"shape": shape, "address": None}
    if shape in ("position", "where"):
        x = rng.choice(people) if shape == "where" and people else rng.choice(names)
        c = sc.where(x)
        text = f"Where is {x}?" if shape == "where" else f"What is the position of {x}?"
        q.update(question=text, answer=" ".join(map(str, c)), cls="position", subject=x)
    elif shape == "neighbour":
        x, d, _n = _pointing(sc, rng, names, (1,))
        a = sc.adjacent(x).get(d)
        q.update(question=f"What is the {d} neighbour of {x}?", answer=a or "nothing", cls=_cls(sc, a), subject=x,
                 address=list(sc.offset(x, d, 1)))
    elif shape == "offset":
        x, d, n = _pointing(sc, rng, names, (2, 3, 4))
        a = sc.at(sc.offset(x, d, n))
        q.update(question=f"What is {n} {d} of {x}?", answer=a or "nothing", cls=_cls(sc, a), subject=x,
                 address=list(sc.offset(x, d, n)))
    elif shape == "nearest":
        x, want = rng.choice(names), rng.choice(kinds)
        a = sc.nearest(x, want)
        q.update(question=f"What is the nearest {want} of {x}?", answer=a or "nothing", cls=_cls(sc, a), subject=x)
    elif shape == "region":
        inside = [n for n in names if sc.regions_of(n)]
        x = rng.choice(inside) if inside and rng.random() < 0.7 else rng.choice(names)
        rs = sc.regions_of(x)
        q.update(question=f"What is the region of {x}?", answer=rs[0] if rs else "nothing",
                 cls="region" if rs else "nothing", subject=x)
    else:
        if rng.random() < 0.75:
            pool = people if people and rng.random() < 0.5 else names
            c = sc.where(rng.choice(pool))
        else:
            c = tuple(rng.randint(lo[i], hi[i]) for i in range(3))
        a = sc.at(c)
        q.update(question="What is at {} {} {}?".format(*c), answer=a or "nothing", cls=_cls(sc, a), subject=None,
                 address=list(c))
    return q


def questions(sc: Scene, seed: int) -> list[dict]:
    """The village's 30 questions, drawn by ``seed`` (the same scene and seed give the same questions). A question
    already asked is drawn again, up to ``TRIES`` times (a small village may not have 30 different ones)."""
    rng = random.Random(seed)
    names = sorted(sc.things)
    people = [n for n in names if sc.kinds.get(n) == VILLAGER]
    kinds = sorted({k for k in sc.kinds.values() if k})
    lo, hi = _box(sc)
    out, seen = [], set()
    for shape, k in PLAN:
        for _ in range(k):
            for _try in range(TRIES):
                q = _draw(sc, rng, shape, names, people, kinds, lo, hi)
                if q["question"] not in seen:
                    break
            seen.add(q["question"])
            out.append(q)
    return out


def forms(sc: Scene) -> dict:
    """The host's grammar for this scene, for ``question_forms.Forms``: one template per relation the scene states."""
    rels = sc.relations("position") + sc.relations("region")
    rels += [f"{d} neighbour" for d in DIRECTIONS] + [f"nearest {k}" for k in sorted({k for k in sc.kinds.values() if k})]
    return {r: ("state", {"present": [f"What is the {r} of {{S}}?"]}) for r in rels}
