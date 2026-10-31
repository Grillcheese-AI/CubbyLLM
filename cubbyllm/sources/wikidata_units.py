"""cubbyllm/sources/wikidata_units.py -- a quantity said with its unit (order 0047, item 3; ``CB_HELDOUT_FIXES=1``).

0046's R0: "How tall is Big Ben?" was answered "96" -- the source dropped the statement's unit. Now:
- a statement with a unit is said with the unit's own symbol (P5061, English first), else its label: "96 m";
- a statement with no unit, of a property whose constraints allow units (P2302 = Q21514353), is not served: the
  number alone would mislead;
- a property without units (a count: population) is said as a bare number, as before.

Wired: WIRED (``wikidata.WikidataSource.facts``).
"""
from __future__ import annotations

__wiring__ = "WIRED"

SYMBOL, CONSTRAINT, ALLOWED_UNITS = "P5061", "P2302", "Q21514353"


def _qid(unit: str) -> str | None:
    tail = (unit or "").rsplit("/", 1)[-1]
    return tail if tail.startswith("Q") else None


def takes_unit(src, pid: str) -> bool:
    for st in src._claims(pid).get(CONSTRAINT, []):
        if ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id") == ALLOWED_UNITS:
            return True
    return False


def symbol(src, qid: str) -> str | None:
    syms = []
    for st in src._claims(qid).get(SYMBOL, []):
        v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
        if isinstance(v, dict) and v.get("text"):
            syms.append((v.get("language") not in ("en", "mul"), v["text"]))
    if syms:
        return sorted(syms)[0][1]
    return src._labels_for([qid]).get(qid)


def quantity_text(src, pid: str, value: dict) -> str | None:
    """The quantity as said, or None when it may not be said (a unit the property takes, missing)."""
    amount = (value.get("amount") or "").lstrip("+")
    q = _qid(value.get("unit") or "1")
    if q:
        sym = symbol(src, q)
        return f"{amount} {sym}" if sym else None
    return None if takes_unit(src, pid) else amount


__all__ = ["quantity_text", "symbol", "takes_unit"]
