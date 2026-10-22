"""cubbyllm/model/fact_layout.py -- where a far fact is written in the scene, and how its names are worded (orders
0004, 0011; the far-fact shards and the 450M's loci reader both use it, so order 0031 moved it here from
``validation/make_fact_shard.py`` and ``validation/fact_names.py``, verbatim).

``address(subject, relation index, piece)``: the entity's cell (a hash into the lattice), the relation's step on the
grid, piece i ``6 i`` cells up z. ``words(name)``: a name as the shards write it (``v2``: order 0011's readable names).

Wired: WIRED (the loci reader; the shard builders import it).
"""
from __future__ import annotations

import hashlib
import re

__wiring__ = "WIRED"

ORIGIN, SIDE, STEP, GRID = 4096, 1024, 6, 54      # the lattice of entity cells, the step grid of relations (x, y)


def cell(entity: str) -> tuple:
    """The entity's cell: a hash into the lattice, the same for every document and every run."""
    h = hashlib.blake2b(entity.encode("utf-8"), digest_size=6).digest()
    return tuple(ORIGIN + int.from_bytes(h[2 * i:2 * i + 2], "little") % SIDE for i in range(3))


def step(rel_index: int) -> tuple:
    return (STEP * (rel_index % GRID), STEP * (rel_index // GRID), 0)


def address(subject: str, rel_index: int, piece: int = 0) -> tuple:
    """The fact's address, and the i-th piece of its object's name ``6 i`` cells up z."""
    c, d = cell(subject), step(rel_index)
    return (c[0] + d[0], c[1] + d[1], c[2] + d[2] + STEP * piece)


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
