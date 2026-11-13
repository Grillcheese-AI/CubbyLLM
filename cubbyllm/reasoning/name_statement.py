"""cubbyllm/reasoning/name_statement.py -- the certificate's name statement (order 0054, the shape Cowork set on
2026-11-11, and her read of the first build the same day): how the question's name reached the item the answer is
about.

- ``asked``: the name as the host cut it; ``store``: the world that read the name (the world that held the first
  fact, never the combined world), with its snapshot; the live source: ``"live"``, with ``read`` (its revision, else
  the time it read the item);
- ``item``: the id the world handed facts for (`worlds.world_handed`: the shared name's decision, else the first
  fact's subject id) and the world's own label for it; never found by matching a label. No id: ``id: null`` and
  ``reason``;
- ``first_fact``: the trace's first fact, its subject as said, and the ids of the items the world read it from;
- ``steps``: the name rules applied, in order, each carrying the string the world gave or the statement it rests on:
  ``label`` (the item's label), ``fold`` (the two strings taken as equal), ``alias`` (the world's alias; ``null`` when
  the world gave none), ``place_qualifier``, ``role`` (0066, check 3), ``original_first`` (0066, check 2),
  ``namesake`` (what the rule compared: its cues, relations, kinds, what it ruled out, the items it kept),
  ``first_by_id`` (several kept, the first in id order said), ``store_item`` (the live source asked about the
  store's item); ``unstated`` for a step this module cannot ground;
- ``others``: the world's other items for the name: the count (``null`` when not counted), and up to 10 with the
  step that dropped each.

A refusal over a name carries one too: ``ambiguous_entity`` lists its options (the record's own candidates);
``entity_unresolved`` the worlds asked and what each held. The checks over a statement are `name_ground`.

Built after the answer, by the same rules over the same snapshot: a record, never a rule (it changes no answer).

Wired: WIRED (`validation/system_eval/run_c.py` keeps it in each record; the replay reads it).
"""
from __future__ import annotations

from .name_ground import LIST, ground, klass
from .planner import normalize, parse_fact

__wiring__ = "WIRED"

POINTS = ("based on", "edition or translation of", "main subject", "derivative work")


def _qualifier(asked: str, label: str, rows) -> dict | None:
    """The asked name is the label and more words: a place that a fact of the item names."""
    a, lab = normalize(asked).split(), normalize(label).split()
    if len(a) <= len(lab) or a[:len(lab)] != lab:
        return None
    rest = " ".join(a[len(lab):])
    for s, r, o in rows:
        if rest in normalize(o).split() or normalize(o).startswith(rest) or rest == normalize(o):
            return {"rule": "place_qualifier", "qualifier": rest, "statement": f"{o} is the {r} of {s}"}
    return {"rule": "place_qualifier", "qualifier": rest, "statement": "unstated"}


def _name_step(asked: str, key: str, names: list) -> dict | None:
    """The step from the world's own name entries (lang, kind 0 label / 1 alias, text) that the asked name is."""
    for want in (0, 1):
        for lang, kind, text in names:
            if kind == want and normalize(text) in (key, normalize(f"the {key}")):
                if kind == 1:
                    return {"rule": "alias", "alias": text, "lang": lang}
                return {"rule": "label", "label": text, "lang": lang} if text == asked else \
                    {"rule": "fold", "equal": [asked, text], "lang": lang}
    return None


def _item_of(w, key: str, fact: str, fit) -> tuple:
    """(id, how, the first fact's ids): the item the world handed facts for, by its own record."""
    h = w.handed(key) if hasattr(w, "handed") else None
    ids = w.fact_items(fact) if hasattr(w, "fact_items") else []
    if h:
        return h[0], h[1], ids
    if len(ids) == 1:
        return ids[0], "the first fact's subject", ids
    kept = [q for q in ids if q in (fit or ())]
    if len(kept) == 1:
        return kept[0], "the first fact's subject, among the items kept", ids
    return None, ("several items state the first fact" if ids else "the world did not record the fact's item"), ids


def _pick(d: dict | None, by_n: int, item, how: str) -> list:
    if d is None:
        return [{"rule": "namesake", "how": "unstated"}] if by_n > 1 else []
    st = [{"rule": "namesake", "kept": list(d.get("fit") or []), "compared": {
        "cues": list(d.get("cues") or []), "relations": list(d.get("rels") or []),
        "kinds": {q: k for q, k in list((d.get("kinds") or {}).items())[:LIST]},
        "ruled_out": dict(d.get("ruled_out") or {}), "stepped_aside": list(d.get("stepped_aside") or []),
        "neutral": bool(d.get("neutral"))}}]
    if len(d.get("fit") or []) > 1:
        st.append({"rule": "first_by_id", "kept": list(d["fit"]), "chose": item, "how": how})
    return st


def _dropped(q, item, aside, d, labels, aliases) -> str:
    if q in aside:
        return "original_first"
    if d is not None:
        if q in (d.get("ruled_out") or {}):
            return f"namesake: the cue {d['ruled_out'][q]}"
        return "first_by_id" if q in (d.get("fit") or []) else "namesake"
    return "tier" if q in aliases and item in labels else "kept"


def _store(w, key: str, asked: str, fact: str, subj: str | None) -> dict:
    labels, aliases = w._tiers(key)
    pool = list(dict.fromkeys(labels + aliases))
    d = w.decision(key) if hasattr(w, "decision") else None
    item, how, ids = _item_of(w, key, fact, (d or {}).get("fit")) if fact else (None, "no fact", [])
    aside = sorted(set((w._rec().get("aside") or {}).get(key, ()))) if hasattr(w, "_rec") else []
    lab, rows, _keys = w._entity(item) if item else (None, [], {})
    st = {"asked": asked, "store": getattr(w, "name", "store"), "snapshot": getattr(w, "state", None),
          "item": {"id": item, "label": lab} if item else {"id": None, "label": None, "reason": how},
          "first_fact": {"fact": fact, "subject": subj, "subject_ids": ids} if fact else None, "steps": []}
    if item:
        st["item"]["how"] = how
        e = w.ask({"op": "get", "id": item}) or {}
        step = _name_step(asked, key, [tuple(n) for n in e.get("names", []) if n[0] in ("en", "fr", "mul")])
        st["steps"] = [step] if step else ([q] if (q := _qualifier(asked, lab or "", rows)) else [])
        if key in getattr(w, "_opened", set()):
            role = [f"{s} {rr} {o}" for s, rr, o in rows if rr in ("position held", "noble title", "occupation")][:1]
            st["steps"].append({"rule": "role", "statement": role[0] if role else "unstated"})
    for a in aside:
        arow = [f"{o} is the {rr} of {s}" for s, rr, o in w._entity(a)[1] if rr in POINTS]
        st["steps"].append({"rule": "original_first", "set_aside": a, "statement": arow[0] if arow else "unstated"})
    tier = labels if labels else aliases
    st["steps"] += _pick(d, len(tier), item, how) if item and len(pool) > 1 else []
    others = [q for q in pool + aside if q != item]
    st["others"] = {"count": len(others), "listed": [
        {"id": q, "label": w._entity(q)[0], "dropped_by": _dropped(q, item, aside, d, labels, aliases)} for q in others[:LIST]]}
    st["steps"] = st["steps"] or [{"rule": "unstated"}]
    return st


def _hdc(w, key: str, asked: str, fact: str, subj: str | None) -> dict:
    tier, by = w._named(key)
    d = w.decision(key) if hasattr(w, "decision") else None
    item, how, ids = _item_of(w, key, fact, (d or {}).get("fit"))
    lab = next((s for s, _r, _o in by.get(item, [])), None) if item else None
    steps = []
    if item and lab and tier == "label":
        steps = [{"rule": "label", "label": lab} if lab == asked else {"rule": "fold", "equal": [asked, lab]}]
    elif item and tier == "alias":
        steps = [{"rule": "alias", "alias": key}]        # the names table's own key
    steps += _pick(d, len(by), item, how) if item and len(by) > 1 else []
    others = [q for q in by if q != item]
    return {"asked": asked, "store": getattr(w, "name", "hdc"), "snapshot": getattr(w, "snapshot", None),
            "item": {"id": item, "label": lab, "how": how} if item else {"id": None, "label": None, "reason": how},
            "first_fact": {"fact": fact, "subject": subj, "subject_ids": ids}, "steps": steps or [{"rule": "unstated"}],
            "others": {"count": len(others), "listed": [{"id": q, "dropped_by": _dropped(q, item, [], d, [], [])}
                                                        for q in others[:LIST]]}}


def _unidentified(world, primary, asked: str, fact: str, subj: str | None) -> dict:
    """The fact came from a world whose facts name no item (the wiki world), or from none of the worlds."""
    from .name_ground import fold_step
    holds = primary is not None and fact in primary
    also = [{"store": o.name, "ids": o.fact_items(fact) if hasattr(o, "fact_items") else None}
            for o in getattr(world, "others", []) if fact in o]
    return {"asked": asked, "store": getattr(primary, "name", "wiki") if holds else None,
            "snapshot": getattr(primary, "snapshot", None) if holds else None,
            "item": {"id": None, "label": subj, "reason": "this world's facts name no item" if holds else
                     "no world holds the first fact"},
            "first_fact": {"fact": fact, "subject": subj, "subject_ids": None}, "also_held": also,
            "steps": [fold_step(asked, subj) if subj else {"rule": "unstated"}],
            "others": {"count": None, "listed": [], "reason": "this world lists no items"}}


def _live(src, asked: str, fact: str, subj: str | None, relations) -> dict:
    old = getattr(src, "last", None)
    src.last = {}
    try:
        got = src.resolve(asked, list(relations or []) or None)
        last = dict(src.last)
    finally:
        src.last = old
    qid, lab = got if got else (None, None)
    how = str(last.get("how") or "")
    from cubbyllm.sources.wikidata_resolve import name_key
    if "store's item" in how:
        steps = [{"rule": "store_item", "statement": f"the store made {asked} stand for {qid}"}]
    elif how.startswith(("namesake rule", "relation:")) or "label over alias" in how:
        steps = [{"rule": "namesake", "how": how}]
    elif "place" in how:
        steps = [{"rule": "place_qualifier", "how": how}]
    elif lab and name_key(lab) == name_key(asked):
        steps = [{"rule": "label", "label": lab} if lab == asked else {"rule": "fold", "equal": [asked, lab]}]
    elif lab:
        steps = [{"rule": "alias", "alias": None, "how": how}]   # the search matched an alias it does not return
    else:
        steps = []
    exact = last.get("exact")
    others = [q for q in (exact or []) if q != qid]
    return {"asked": asked, "store": "live", "snapshot": None, "read": src.read_of(qid) if qid and hasattr(src, "read_of") else {},
            "item": {"id": qid, "label": lab, "how": how} if qid else {"id": None, "label": None, "reason": "not resolved again"},
            "first_fact": {"fact": fact, "subject": subj, "subject_ids": None}, "steps": steps or [{"rule": "unstated"}],
            "others": {"count": None if exact is None else len(others), "listed": [{"id": q, "dropped_by": how} for q in others[:LIST]]}}


def _handed_here(w, key: str, fact: str) -> bool:
    """`w` recorded handing facts for the name, or read the fact from one item only."""
    return bool((hasattr(w, "handed") and w.handed(key)) or (hasattr(w, "fact_items") and len(w.fact_items(fact)) == 1))


def _refusal(loop, rec: dict) -> dict | None:
    world, reason = getattr(loop, "world", None), rec.get("reason")
    if reason == "ambiguous_entity":
        cands = rec.get("candidates") or []
        more = (rec.get("clarify") or {}).get("more") or 0
        ids = {c[2] for c in cands}
        found = next((o.name for o in getattr(world, "others", []) if hasattr(o, "namesakes")
                      and ids & set(o.namesakes(rec.get("seed") or ""))), "live")
        return {"asked": rec.get("seed"), "store": found, "item": {"id": None, "label": None, "reason": reason},
                "steps": [{"rule": "namesake", "how": "nothing decided: the asker is asked"}],
                "others": {"count": len(cands) + more, "listed": [{"id": c[2], "label": c[0], "detail": c[1], "dropped_by":
                                                                   "undecided"} for c in cands[:LIST]]}}
    name = rec.get("seed") or next(iter(rec.get("entities") or []), None)
    if not name:
        return None
    key, asked = normalize(name), []
    for o in [getattr(world, "primary", None), *getattr(world, "others", [])]:
        if o is None:
            continue
        n = len(set(o._tiers(key)[0] + o._tiers(key)[1])) if hasattr(type(o), "_tiers") else \
            len(o._named(key)[1]) if hasattr(o, "_named") else None
        asked.append({"store": getattr(o, "name", "wiki"), "items": n})
    asked.append({"store": "live", "items": None})
    counted = [a["items"] for a in asked if a["items"] is not None]
    return {"asked": name, "store": None, "worlds": asked, "item": {"id": None, "label": None, "reason": reason},
            "steps": [{"rule": "unresolved"}], "others": {"count": sum(counted) if counted else None, "listed": []}}


def build(loop, rec: dict, live: bool = False) -> dict | None:
    """The name statement for a record's seed, from the world that held its first fact (`live`: the live source gave
    it). None when there is nothing to state. A record: it never raises (an error is kept in it)."""
    try:
        return _build(loop, rec, live)
    except Exception as e:                               # noqa: BLE001 -- a record never breaks the run
        return {"asked": rec.get("seed"), "steps": [{"rule": "unstated", "error": str(e)[:120]}],
                "others": {"count": None, "listed": []}}


def _build(loop, rec: dict, live: bool = False) -> dict | None:
    seed, trace = rec.get("seed"), rec.get("trace") or []
    world = getattr(loop, "world", None)
    if not trace and rec.get("reason") in ("ambiguous_entity", "entity_unresolved"):
        return _refusal(loop, rec)
    if not seed or not trace:
        return None
    rels: dict = {}                                      # the relations the worlds hold: "author of Gospel of John"
    for w in [world, *getattr(world, "others", [])]:     # splits at the relation, not at the last "of"
        try:
            rels.update(w.relations() if hasattr(w, "relations") else {})
        except Exception:                                # noqa: BLE001
            pass
    t = parse_fact(trace[0], known=rels or None)
    subj = t.subj if t is not None else None
    key = normalize(seed)
    if live:
        return _live(loop.source, seed, trace[0], subj, rec.get("plan"))
    held = world.held_by(trace[0]) if hasattr(world, "held_by") else None
    if held is not None and not _handed_here(held, key, trace[0]):   # a world holds every fact it built for a name's
        held = next((o for o in getattr(world, "others", []) if o is not held and _handed_here(o, key, trace[0])
                     and trace[0] in o), held)           # items, handed or not: the world that handed it, if one did
    if held is not None and hasattr(type(held), "_tiers"):
        return _store(held, key, seed, trace[0], subj)
    if held is not None and hasattr(held, "_named"):
        return _hdc(held, key, seed, trace[0], subj)
    return _unidentified(world, getattr(world, "primary", world), seed, trace[0], subj)


__all__ = ["build", "ground", "klass"]
