"""cubbyllm/reasoning/name_statement.py -- the certificate's name statement (order 0054, the shape Cowork set on
2026-11-11 for the certificate replay's kill): how the question's name reached the item the answer is about.

The replay (`validation/system_eval/replay_cert.py`) found 17 of 86 spoken answers that their certificate does not
carry: the trace states facts about an item's label, never how "nashville tn" became Nashville. A name statement:
- ``asked``: the name as the host cut it; ``item``: the item's id and the label the answer uses;
- ``store``: the world that read the name, with its snapshot (the live source: ``"live"``, no revision kept);
- ``steps``: the name rules applied, in order, each with what it rests on (``statement``, or ``unstated``):
  ``label``, ``alias``, ``fold`` (an article, a casing, a punctuation: the two strings taken as equal),
  ``place_qualifier`` (the fact that ties the item to the place the extra words name), ``role`` (0066, check 3),
  ``original_first`` (0066, check 2: the statement by which each set-aside item points at the original),
  ``namesake`` (the rule that chose among several, as the world gave it), ``store_item`` (the live source asked
  about the store's item);
- ``others``: the world's other items for the name: the count, and up to 10 with the step that dropped each.

It is built after the answer, from the world that held the first fact, by the same rules over the same snapshot:
a record, never a rule (it changes no answer). A step this module cannot ground is marked ``unstated``.

Wired: WIRED (`validation/system_eval/run_c.py` keeps it in each record; the replay reads it).
"""
from __future__ import annotations

from .planner import normalize, parse_fact

__wiring__ = "WIRED"

LIST = 10
POINTS = ("based on", "edition or translation of", "main subject", "derivative work")


def _label_steps(asked: str, label: str, kind: int | None) -> list[dict]:
    if asked == label:
        return [{"rule": "label", "statement": f"{label} is the label of the item"}]
    if normalize(asked) == normalize(label) or normalize(f"the {asked}") == normalize(label):
        return [{"rule": "fold", "equal": [asked, label]}]
    if kind == 1:
        return [{"rule": "alias", "statement": f"{asked} is an alias of {label}"}]
    if kind == 0:
        return [{"rule": "fold", "equal": [asked, label], "note": "a label of the item in another form"}]
    return []


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


def _store(w, key: str, asked: str, subj: str) -> dict:
    labels, aliases = w._tiers(key)
    pool = list(dict.fromkeys(labels + aliases))
    item = next((q for q in pool if w._entity(q)[0] == subj), None)
    r = w._rec() if hasattr(w, "_rec") else {}
    aside = set((r.get("aside") or {}).get(key, ()))
    st = {"asked": asked, "store": getattr(w, "name", "store"), "snapshot": getattr(w, "state", None),
          "item": {"id": item, "label": subj}, "steps": []}
    if item:
        lab, rows, names = w._entity(item)
        st["steps"] = _label_steps(asked, lab, names.get(key)) or ([q] if (q := _qualifier(asked, lab, rows)) else [])
        if key in getattr(w, "_opened", set()):
            role = [f"{s} {rr} {o}" for s, rr, o in rows if rr in ("position held", "noble title", "occupation")][:1]
            st["steps"].append({"rule": "role", "statement": role[0] if role else "unstated"})
    for a in sorted(aside):
        arow = [f"{o} is the {rr} of {s}" for s, rr, o in w._entity(a)[1] if rr in POINTS]
        st["steps"].append({"rule": "original_first", "set_aside": a, "statement": arow[0] if arow else "unstated"})
    d = w.decision(key) if hasattr(w, "decision") else None
    if d and len(pool) > 1:
        st["steps"].append({"rule": "namesake", "how": str(d.get("how", {}).get(item) if isinstance(d.get("how"), dict) else d.get("how")),
                            "fit": list(d.get("fit") or [])})
    others = [q for q in pool + sorted(aside) if q != item]
    fit = set((d or {}).get("fit") or [])
    st["others"] = {"count": len(others), "listed": [
        {"id": q, "label": w._entity(q)[0], "dropped_by": "original_first" if q in aside else
         ("namesake" if d and q not in fit else "tier" if q in aliases and item in labels else "kept")}
        for q in others[:LIST]]}
    if not st["steps"]:
        st["steps"] = [{"rule": "unstated"}]
    return st


def _hdc(w, key: str, asked: str, subj: str) -> dict:
    tier, by = w._named(key)
    item = next((q for q, rows in by.items() if any(s == subj for s, _r, _o in rows)), None)
    steps = _label_steps(asked, subj, 0 if tier == "label" else 1 if tier == "alias" else None)
    d = w.decision(key) if hasattr(w, "decision") else None
    if d and len(by) > 1:
        steps.append({"rule": "namesake", "fit": list(d.get("fit") or [])})
    others = [q for q in by if q != item]
    return {"asked": asked, "store": getattr(w, "name", "hdc"), "snapshot": getattr(w, "snapshot", None),
            "item": {"id": item, "label": subj}, "steps": steps or [{"rule": "unstated"}],
            "others": {"count": len(others), "listed": [{"id": q, "dropped_by": "namesake" if d else "tier"} for q in others[:LIST]]}}


def _live(src, asked: str, subj: str, relations) -> dict:
    got = src.resolve(asked, list(relations or []) or None)
    how = str((getattr(src, "last", None) or {}).get("how") or "")
    qid = got[0] if got else None
    if "store's item" in how:
        steps = [{"rule": "store_item", "statement": f"the store made {asked} stand for {qid}"}]
    elif how.startswith("namesake rule"):
        steps = [{"rule": "namesake", "how": how}]
    elif how.startswith("exact"):
        steps = _label_steps(asked, subj, 0 if normalize(asked) == normalize(subj) else 1)
    elif how:
        steps = [{"rule": "place_qualifier" if "place" in how or "in " in how else "unstated", "how": how}]
    else:
        steps = []
    return {"asked": asked, "store": "live", "snapshot": None, "item": {"id": qid, "label": subj},
            "steps": steps or [{"rule": "unstated"}], "others": {"count": None, "listed": []}}


def build(loop, rec: dict, live: bool = False) -> dict | None:
    """The name statement for a record's seed, from the world that held its first fact (`live`: the live source gave
    it). None when there is nothing to state (no seed, no trace)."""
    seed, trace = rec.get("seed"), rec.get("trace") or []
    world = getattr(loop, "world", None)
    if seed and not trace and rec.get("reason") in ("ambiguous_entity", "name_unresolved"):
        stores = [w for w in [world, *getattr(world, "others", [])] if hasattr(type(w), "_tiers")]   # a refusal over a
        return _store(stores[0], normalize(seed), seed, None) if stores else None                    # name: its options
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
    if not subj:
        return {"asked": seed, "steps": [{"rule": "unstated"}], "others": {"count": None, "listed": []}}
    key = normalize(seed)
    try:
        if live:
            return _live(loop.source, seed, subj, rec.get("plan"))
        held = world.held_by(trace[0]) if hasattr(world, "held_by") else None
        if held is not None and hasattr(type(held), "_tiers"):
            return _store(held, key, seed, subj)
        if held is not None and hasattr(held, "_named"):
            return _hdc(held, key, seed, subj)
        return {"asked": seed, "store": getattr(world, "name", "world"), "item": {"id": None, "label": subj},
                "steps": _label_steps(seed, subj, None) or [{"rule": "unstated"}], "others": {"count": 0, "listed": []}}
    except Exception as e:                               # noqa: BLE001 -- a record: never breaks the answer
        return {"asked": seed, "steps": [{"rule": "unstated", "error": str(e)[:120]}], "others": {"count": None, "listed": []}}


def ground(st: dict, facts: list[str]) -> str:
    """The replay's fallback check (Cowork's shape): each step's ground, no more. ``re-derived``: every step stands on
    what the statement carries; ``stated only``: a pick among several, or a step it cannot ground; ``not the same``:
    a step's ground fails."""
    if not st:
        return "stated only"
    label = (st.get("item") or {}).get("label") or ""
    asked, out = st.get("asked") or "", "re-derived"
    for s in st.get("steps", []):
        rule = s.get("rule")
        if rule == "label" and asked != label:
            return "not the same"
        if rule == "fold" and normalize(s["equal"][0]) not in (normalize(s["equal"][1]), normalize(s["equal"][1]).removeprefix("the ")):
            out = "stated only" if out == "re-derived" else out          # a form the fold rule does not name
        if rule in ("namesake", "store_item", "unstated", "role", "original_first") or s.get("statement") == "unstated":
            out = "stated only"
        if rule == "place_qualifier" and s.get("statement") not in facts and s.get("statement") != "unstated":
            facts = facts + [s["statement"]]             # carried with the statement
    if (st.get("others") or {}).get("count") and (st["others"]["count"] or 0) > LIST:
        out = "stated only"
    return out


__all__ = ["build", "ground"]
