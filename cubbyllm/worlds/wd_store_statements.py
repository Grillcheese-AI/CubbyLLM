"""cubbyllm/worlds/wd_store_statements.py -- the store's statements behind the facts it hands out (order 0076, step 2):
a record, never a rule (no answer reads it).

A certificate names each fact it rests on as one statement of the store at its snapshot: the item, the property,
the value as the store holds it, the statement's id and rank, the item's revision and modified time, and the
statement's own start, end and point-in-time qualifiers (their raw time literals and precision) where it has them.
The store world calls `_note_statement` for every statement it turns into a fact (`WdStoreWorld._entity_rows`);
``statements_of(subj, rel, obj)`` gives back every statement that said that fact (two statements can say one
sentence). Nothing here is read by a walk, a check or a reply.

Wired: WIRED (`wd_store_world.WdStoreWorld` takes it as a mixin; `validation/system_eval/run_c.py` records it with
``CB_CERT_FACTS=1``, and `cubbyllm/records/cheez_writer.py` reads that record).
"""
from __future__ import annotations

from .wd_store_names import END, POINT_IN_TIME, START, key3

__wiring__ = "WIRED"

KEPT = 200_000                                           # the facts whose statements are remembered (oldest dropped)
QUALIFIERS = {START: "start", END: "end", POINT_IN_TIME: "point in time"}


def _times(s: dict, nodes: dict) -> dict:
    """{"start" | "end" | "point in time": {"time": raw literal, "precision": int | None}} of a statement."""
    node_of = dict(s.get("hot_qualifier_nodes") or [])
    out = {}
    for p, v in s.get("hot_qualifiers") or []:
        if p in QUALIFIERS and isinstance(v, dict) and v.get("typed"):
            node = (nodes or {}).get(node_of.get(p) or "") or {}
            try:
                prec = int(node.get("wikibase:timePrecision", ""))
            except ValueError:
                prec = None
            out.setdefault(QUALIFIERS[p], {"time": v["typed"][0], "precision": prec})
    return out


class Statements:
    """For `WdStoreWorld`."""

    def _note_statement(self, qid: str, entity: dict, subj: str, rel: str, obj: str, s: dict, nodes: dict) -> None:
        got = self.__dict__.setdefault("_statements", {})
        rec = {"item": qid, "pid": f"P{s.get('pid')}", "guid": s.get("guid"), "rank": s.get("rank"),
               "best": bool(s.get("best")), "value": s.get("value"), "revision": entity.get("revision"),
               "modified": entity.get("modified"), "references": len(s.get("references") or []),
               "times": _times(s, nodes)}
        rows = got.setdefault(key3(subj, rel, obj), [])
        if all(r["guid"] != rec["guid"] or r["item"] != qid for r in rows):
            rows.append(rec)
        while len(got) > KEPT:
            got.pop(next(iter(got)))

    def statements_of(self, subj: str, rel: str, obj: str) -> list:
        """Every statement this world read the fact (subj, rel, obj) from ([] when it never read it)."""
        return list(self.__dict__.get("_statements", {}).get(key3(subj, rel, obj), ()))


__all__ = ["KEPT", "Statements"]
