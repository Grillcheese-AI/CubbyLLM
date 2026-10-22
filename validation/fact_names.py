"""fact_names -- the wiki world's names as words, for ``make_fact_shard`` (order 0011, ``--names v2``):
relations, entities and event ids each rendered readably, and the question asked without a doubled word.
Without ``v2``, the rendering of every fact shard built before 0011, bit for bit."""
from __future__ import annotations

import re

CAMEL = r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
DIGITS = r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Z][a-z])"     # a digit run is a word: Began1933, 1982French


def words(name: str, relation: bool = False, v2: bool = False) -> str:
    """A name as words. ``v2`` (``--names v2``, order 0011): a ``relation`` ``WORKS_FOR`` -> ``works for``;
    an entity ``KeikoMatsuzaka`` -> ``Keiko Matsuzaka``, an all-caps one kept (``ABBA``, ``1 BCE``); an
    event id (any name with an underscore) part by part, split on CamelCase and digit runs, case kept:
    ``FortPeckDam_ConstructionBegan1933`` -> ``Fort Peck Dam Construction Began 1933``,
    ``Rocket_UseInChina13thCentury`` -> ``Rocket Use In China 13th Century`` (an id cannot tell a word from
    a proper noun, so nothing is lowercased: "china" would be wrong, "Began" is only stiff). Without
    ``v2``, every shard before 0011, bit for bit: a name with an underscore or in capitals lowercased with
    its underscores spaced (``fortpeckdam constructionbegan1933``), any other split on CamelCase."""
    if not v2:
        if "_" in name or name.isupper():
            return name.replace("_", " ").lower()
        return re.sub(CAMEL, " ", name)
    if relation:
        return name.replace("_", " ").lower()
    if "_" in name:
        return " ".join(p if p.isupper() else re.sub(DIGITS, " ", re.sub(CAMEL, " ", p)) for p in name.split("_") if p) or name
    return name if name.isupper() else re.sub(CAMEL, " ", name)


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
