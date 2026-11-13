"""cubbyllm/reasoning/name_ground.py -- the replay's checks over a name statement (order 0054; Cowork's read of the
first build, 2026-11-11, item 2: a check that can fail, and reads what it is handed).

``ground(statement, facts)``:
- ``not the same``: the item is outside the items its namesake step kept; the item is not the first fact's subject
  (by id; by label where the world has no ids); a ``label``, ``fold`` or ``alias`` step whose world string is not the
  asked name (by the step's own rule);
- ``stated only``: a pick (``namesake``, ``first_by_id``, ``original_first``, ``role``, ``store_item``), an
  ``unstated`` step, an alias the world gave no string for, a ``place_qualifier`` with no fact, a statement with no
  item id, or more others than the 10 listed;
- ``re-derived``: every step stands on a string or a statement the statement carries.

``klass(statement)``: the three classes of the report (item 7): ``checked`` (a label or a fold of an identified item:
the replay compares the strings itself), ``store statement`` (an alias the world gave, a qualifier's fact),
``taken as stated`` (a pick, a role, a last choice, no item); ``not the same`` apart.

Wired: WIRED (`name_statement` re-exports both; `validation/system_eval/replay_cert.py` calls them).
"""
from __future__ import annotations

from .planner import normalize

__wiring__ = "WIRED"

LIST = 10
PICKS = ("namesake", "first_by_id", "original_first", "role", "store_item", "unstated", "unresolved")


def fold_step(asked: str, said: str) -> dict:
    return {"rule": "label", "label": said} if asked == said else {"rule": "fold", "equal": [asked, said]}


def _folds(a: str, b: str) -> bool:
    from cubbyllm.sources.wikidata_resolve import name_key
    na, nb = normalize(a or ""), normalize(b or "")
    return bool(na) and (na == nb or nb.removeprefix("the ") == na or na.removeprefix("the ") == nb
                         or name_key(a) == name_key(b))


def _step(s: dict, asked: str) -> str:
    rule = s.get("rule")
    if rule == "label":
        return "re-derived" if s.get("label") == asked else "not the same"
    if rule == "fold":
        e = s.get("equal") or [None, None]
        return "re-derived" if e[0] == asked and _folds(e[0], e[1]) else "not the same"
    if rule == "alias":
        if not s.get("alias"):
            return "stated only"                         # composed from the asked name: nothing to compare
        return "re-derived" if _folds(asked, s["alias"]) else "not the same"
    if rule == "place_qualifier":
        st = s.get("statement")
        return "stated only" if not st or st == "unstated" else "re-derived"
    if rule in PICKS or s.get("statement") == "unstated":
        return "stated only"
    return "stated only"                                 # a rule this check does not know: never re-derived


def ground(st: dict | None, facts: list[str] | None = None) -> str:
    """The statement's outcome (module doc). `facts`: the certificate's statements (a qualifier's fact may be among
    them as well as in the step)."""
    if not st:
        return "stated only"
    item, asked = st.get("item") or {}, st.get("asked") or ""
    iid = item.get("id")
    for s in st.get("steps", []):                        # the item is among the items its namesake step kept
        if s.get("rule") in ("namesake", "first_by_id") and s.get("kept") and iid and iid not in s["kept"]:
            return "not the same"
    ff = st.get("first_fact") or {}
    ids = ff.get("subject_ids")
    if iid and ids and iid not in ids:
        return "not the same"                            # the item is not the subject of the first fact (by id)
    said = ff.get("subject")
    if not ids and item.get("label") and said and not _folds(item["label"], said) and \
            not (st.get("store") == "live" and _folds(asked, said)):   # (the live source says an alias's facts of the
        return "not the same"                            #   asked name) -- by label, where the world gave no id
    out = "re-derived" if iid else "stated only"
    for s in st.get("steps", []):
        got = _step(s, asked)
        if got == "not the same":
            return got
        if got == "stated only":
            out = "stated only"
    if ((st.get("others") or {}).get("count") or 0) > LIST:
        out = "stated only"
    return out


def klass(st: dict | None) -> str:
    """The report's class of a statement (module doc)."""
    g = ground(st)
    if g == "not the same":
        return g
    if g == "stated only":
        return "taken as stated"
    rules = {s.get("rule") for s in (st or {}).get("steps", [])}
    return "store statement" if rules & {"alias", "place_qualifier"} else "checked"


__all__ = ["LIST", "fold_step", "ground", "klass"]
