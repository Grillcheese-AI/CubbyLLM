"""cubbyllm/worlds/wd_store_values.py -- how the hot Wikidata store's values are said (order 0055, step 3).

A store statement's value comes back exactly as the dump wrote it (`wdstore serve`'s `get`, with its value nodes).
This turns it into the words a fact holds, or None when it cannot be said exactly -- nothing is rounded or guessed:

- an item: its English label, else its `mul` label (none: not said);
- a time at day, month or year precision: "11 March 1952", "March 1952", "1952"; a year before 1 as "500 BC". A
  Julian-calendar date is said as its year only (its day and month are the Julian ones). Below year precision
  (decade, century, ...) or with no value node to give the precision: not said;
- a quantity: its amount as written (no float: the text, its sign dropped) and its unit's label ("1.96 metre");
  without a unit ("1"), the amount alone; a unit with no label: not said;
- monolingual text: English or `mul` only;
- a string: only for a property whose datatype is String (an external id is a string too, and is never a fact here).

Which statements: Wikidata's "truthy" ones -- best rank (preferred where any, else normal), never deprecated. On a
relation whose value changes (the time-sensitive ones: spouse, employer, position held, ...), not one with an end
time (P582): it has ended, and the loop says such a relation "as of" the dump. On any other relation an end time does
not make the fact false ("educated at" stays true after the degree), so the statement is kept.

Wired: WIRED (`cubbyllm/worlds/wd_store_world.py`).
"""
from __future__ import annotations

__wiring__ = "WIRED"

SAYABLE = {"wikibase:WikibaseItem", "wikibase:Time", "wikibase:Quantity", "wikibase:String", "wikibase:Monolingualtext"}
END_TIME = 582
JULIAN = "wd:Q1985786"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def kept(s: dict, changes: bool) -> bool:
    """A statement read as a fact: best rank, not deprecated; on a relation whose value changes (`changes`), not ended."""
    ended = any(q[0] == END_TIME for q in s.get("hot_qualifiers", []))
    return bool(s.get("best")) and s.get("rank") != 0 and not (changes and ended)


def entity_ids(statements) -> set:
    """Every item a statement's value names (their labels are fetched in one request)."""
    out = set()
    for s in statements:
        v = s.get("value")
        if isinstance(v, dict) and "entity" in v:
            out.add(v["entity"])
    return out


def unit_ids(statements, nodes: dict) -> set:
    """Every quantity's unit item."""
    out = set()
    for s in statements:
        n = nodes.get(s.get("value_node") or "") or {}
        u = n.get("wikibase:quantityUnit", "")
        if u.startswith("wd:Q") and u != "wd:Q199":
            out.add(u[3:])
    return out


def say_time(literal: str, node: dict | None) -> str | None:
    if not node:
        return None
    try:
        p = int(node.get("wikibase:timePrecision", ""))
    except ValueError:
        return None
    if p < 9:
        return None
    neg = literal.startswith("-")
    body = literal.lstrip("+-")
    try:
        y, m, d = int(body[0:body.index("-")]), *(int(x) for x in body[body.index("-") + 1:].split("T")[0].split("-"))
    except (ValueError, IndexError):
        return None
    year = f"{y} BC" if neg else str(y)
    if neg and y == 0:
        return None
    if p == 9 or node.get("wikibase:timeCalendarModel") == JULIAN:
        return year
    if p == 10:
        return f"{MONTHS[m - 1]} {year}" if 1 <= m <= 12 else None
    return f"{d} {MONTHS[m - 1]} {year}" if 1 <= m <= 12 and 1 <= d <= 31 else None


def say_quantity(amount: str, node: dict | None, labels: dict) -> str | None:
    a = amount.lstrip("+")
    if not a:
        return None
    u = (node or {}).get("wikibase:quantityUnit", "wd:Q199")
    if u in ("", "wd:Q199"):
        return a
    lab = labels.get(u[3:]) if u.startswith("wd:Q") else None
    return f"{a} {lab}" if lab else None


def say(value, node: dict | None, datatype: str | None, labels: dict) -> str | None:
    """A main value's words, or None (module doc). `labels`: item id -> its label."""
    if datatype not in SAYABLE or not isinstance(value, dict):
        return None                                      # some value, no value, an external id, a URL, ...
    kind, x = next(iter(value.items()))
    if kind == "entity":
        return labels.get(x)
    if kind == "mono":
        return x[0] if x[1] in ("en", "mul") else None
    if kind == "typed" and datatype == "wikibase:Time":
        return say_time(x[0], node)
    if kind == "typed" and datatype == "wikibase:Quantity":
        return say_quantity(x[0], node, labels)
    if kind == "str" and datatype == "wikibase:String":
        return x
    return None
