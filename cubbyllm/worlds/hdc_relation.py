"""cubbyllm/worlds/hdc_relation.py -- a shared name decided by the question's relation and its cue words (order 0037).

Nick: "a capital is usually in a country so it should pick the country France (there is only one)". A hop on a name
several items share reads ONE item when:
- that item's kind carries the hop's relation (``hdc_kinds.Table.carries``: a share well over the typed items' base,
  not "any member"), and goes with each of the question's cue words (``hdc_cues``: "Nobel" goes with people);
- no other namesake's kind does.

A namesake's kind is its recorded one (``instance of``), else the one read from its own facts (``Table.read``: a
citizenship and an occupation read as a person). An item whose facts say nothing of its kind does not compete (France's
Q49644, whose only fact is "country: France", asked a capital). For a place relation (``PLACING``: an item's country,
what it is located in) every item that holds such a fact competes, whatever its kind: a thing with a country is
somewhere, and a kind's share of "country" says nothing, since nearly every typed item holds one (Cowork's reading of
the audit, 2026-10-27: KATA, a village in Estonia picked over a village in Iran whose one fact was its country; and
the Eiffel Tower's replica in the United States, read alone when the tower's own kind did not carry "country"). Two or
more that fit are two of a kind: each is answered, labelled by one of its own facts (``reasoning.ask_namesakes``). None
that fits: the loop asks which, as since 0036.

Nothing is ranked: an item never wins for being famous or more completely described (Q142's 197 facts count for
nothing), and the order the items are named in is their ids'.

Wired: WIRED (``hdc_namesakes.Namesakes.hop_rows``; on by default since 2026-10-27, the audit read by Cowork at the
line -- 1 wrong pick in 50; ``CB_NAMESAKE_RELATION=0`` turns it off).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"

INSTANCE = "instance of"
SAY_KIND = {"human": "person"}                           # a kind as said in a sentence ("there is also a person named ...")
PLACING = frozenset(("country", "located in the administrative territorial entity", "located in"))   # facts that place an
#   item somewhere; an item of no kind that holds one competes for one of them (where it is), for nothing else


def qnum(item: str) -> int:
    """An item's place in the fixed order: its id's number (Q142 before Q49644)."""
    return int(item[1:]) if item[:1] == "Q" and item[1:].isdigit() else 1 << 60


def item_kinds(rows, table) -> tuple[set, str]:
    """(kinds, how): recorded (``instance of``), else read from the item's own relations ("read", or "read widely"
    when they go with no kind together: ``Table.read_how``), else none."""
    rec = {normalize(t.obj) for _f, t in rows if normalize(t.rel) == INSTANCE}
    if rec:
        return rec, "recorded"
    if table is None:
        return set(), "none"
    if hasattr(table, "read_how"):
        return table.read_how({t.rel for _f, t in rows})
    got = table.read({t.rel for _f, t in rows})
    return got, ("read" if got else "none")


def sayable(kinds, how: str) -> bool:
    """Whether an item may be named by its kind: one recorded, or one kind read (a read that leaves several, or one
    made widely, does not say which it is: order 0037's audit named a band "a film")."""
    return bool(kinds) and (how == "recorded" or (how == "read" and len(kinds) == 1))


def kind_word(kinds: set, table) -> str:
    """One word for an item's kinds in a sentence: the most widely held of them (a word for the item, never a rank of
    items), as said ("person" for human)."""
    size = getattr(table, "_size", None) or {}
    k = max(sorted(kinds), key=lambda x: size.get(x, 0)) if kinds else ""
    return SAY_KIND.get(k, k)


def decide(by: dict, rels: set, cues: list, table, cue_table) -> dict:
    """Order 0037's rule over a name's items (`by`: {item: [(fact, Triple)]}) for a hop on `rels` (the world's names of
    the hop's relation) with the question's cue words. Returns {"fit": [items that fit, in id order], "kinds": {item:
    [kinds, how]}, "ruled_out": {item: the cue word}, "cues": [the cue words the world uses], "placed": [the items of
    no kind that compete by a fact that places them, for a place relation]}."""
    used = [w for w in cues if cue_table is not None and cue_table.kinds(w) is not None]
    out = {"fit": [], "kinds": {}, "ruled_out": {}, "cues": used, "rels": sorted(rels), "placed": []}
    place = bool(set(rels) & PLACING)
    for q in sorted(by, key=qnum):
        ks, how = item_kinds(by[q], table)
        out["kinds"][q] = [sorted(ks), how]
        placed = place and bool({normalize(t.rel) for _f, t in by[q]} & PLACING)
        if placed:                                       # placed somewhere: it competes for where it is, whatever its
            out["placed"].append(q)                      #   kind (KATA; and a kind's share of "country" says nothing)
        elif not ks or not any(table.carries(k, r) for k in ks for r in rels):
            continue                                     # (no kind known, or none that carries the relation)
        miss = next((w for w in used if ks and not ks & cue_table.kinds(w)), None)   # (a cue rules out a KIND: an item
        #   of none, placed, stays in the running)
        if miss:
            out["ruled_out"][q] = miss                   # a cue word its kind never goes with: it does not compete
            continue
        out["fit"].append(q)
    # two of a kind compete together: an item that shares a kind with one that fits is in the running too, whether or
    # not its own facts make its kind carry the relation -- which of them is more completely described decides nothing
    # (order 0037's audit: two taxa named Thespis, one picked because its facts were read more widely)
    shared = set().union(*(set(out["kinds"][q][0]) for q in out["fit"])) if out["fit"] else set()
    for q in sorted(by, key=qnum):
        if q not in out["fit"] and q not in out["ruled_out"] and set(out["kinds"][q][0]) & shared:
            out["fit"].append(q)
    out["fit"].sort(key=qnum)
    return out


def others(decision: dict, by: dict, table, at_most: int = 2) -> list[dict]:
    """The other namesakes to name after a one-item answer: those whose kind can be said (`sayable`), in id order, at
    most `at_most`; each {"item", "label", "kind"} (an item with nothing certain of its kind is not named)."""
    fit = set(decision.get("fit") or ())
    got = []
    for q in sorted(by, key=qnum):
        ks, how = decision["kinds"].get(q, [[], "none"])
        if q in fit or not sayable(ks, how):
            continue
        got.append({"item": q, "label": by[q][0][1].subj, "kind": kind_word(set(ks), table)})
    return got[:at_most]


__all__ = ["PLACING", "decide", "item_kinds", "kind_word", "others", "qnum", "sayable"]
