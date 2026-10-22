"""fact_names -- the wiki world's names as words, for ``make_fact_shard`` (order 0011, ``--names v2``):
relations, entities and event ids each rendered readably, and the question asked without a doubled word.
Without ``v2``, the rendering of every fact shard built before 0011, bit for bit."""
from __future__ import annotations

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubbyllm.model.fact_layout import CAMEL, DIGITS, words  # noqa: E402,F401 -- with the layout (order 0031)


def question(s: str, r: str, v2: bool = False) -> str:
    """``What is the works for of Keiko Matsuzaka?``; with ``v2`` a relation that ends in *of* asks with the
    subject first (``VARIANT_OF``: ``What is Spanish Dancer variant of?``, not ``the variant of of``): dropping
    the doubled *of* instead would ask the inverse ("the capital of Paris" for ``Paris CAPITAL_OF France``),
    and a *the* parts a name and a relation that share a word (``What is Camp Album the album of?``)."""
    rel, subj = words(r, True, v2), words(s, v2=v2)
    if not (v2 and rel.endswith(" of")):
        return f"What is the {rel} of {subj}?"
    return f"What is {subj} the {rel}?" if subj.casefold().split()[-1:] == rel.split()[:1] else f"What is {subj} {rel}?"


def doubled(text: str, name: str = "") -> list:
    """The adjacent repeated words of ``text`` (case folded) outside ``name``: ``of of`` and ``Album album of``
    are defects; ``Duran Duran`` and "the author of *Of* Mice And Men" are the name's own (inside it, or a
    title that begins with the template's word)."""
    if name:                                                   # the name's last word stays: a repeat after it counts
        text = text.replace(name, " @ " + (re.findall(r"[\w']+", name) or [""])[-1])
    w = re.findall(r"[\w']+|@", text.casefold())
    return [a for a, b in zip(w, w[1:]) if a == b and a != "@"]
