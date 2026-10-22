"""cubbyllm/reasoning/emitted.py -- the emitter's plan, read off the program it emits: a ``CotPlan`` (``SEED`` and
``HOPk`` bindings) or a ``CotChain`` (``Hk_<RELATION>`` roles) -> (relations inner to outer, seed). The plan is a
proposal; ``plan_verify`` disposes of it. Order 0031 moved it here, verbatim, from
``validation/exp_r9_matched_pairs.py`` (where it was first measured), since the ask loop reads it.

Wired: WIRED (the ask loop's emitter seat; the r-series experiments import it).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"

BIND = re.compile(r'bind\s+frame\s*,\s*(?P<role>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"(?P<obj>(?:[^"\\]|\\.)*)"\s*;')
HOP = re.compile(r"^HOP(?P<hop>\d+)$")
ROLE = re.compile(r"^H(?P<hop>\d+)_(?P<rel>.+)$")


def emitted_plan(text: str, normalize):
    """CotPlan (SEED + HOPk) or CotChain (H-roles) -> (rels inner->outer, seed) or None."""
    seed, hops, roles = None, {}, []
    for m in BIND.finditer(text or ""):
        r, o = m.group("role"), m.group("obj")
        if r == "SEED" and seed is None:
            seed = o
        elif (h := HOP.match(r)):
            hops.setdefault(int(h.group("hop")), o)
        elif (h := ROLE.match(r)) and r not in roles:
            roles.append(r)
    if hops:
        rels = [hops[k] for k in sorted(hops)]
    elif roles:
        rels = [ROLE.match(r).group("rel").lower().replace("_", " ") for r in sorted(roles, key=lambda r: int(ROLE.match(r).group("hop")))]
    else:
        return None
    return [r.strip() for r in rels], (normalize(seed) if seed else None)
