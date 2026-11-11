"""cubbyllm/worlds/wd_store_names.py -- three of Cowork's proposals from the read of order 0054's "after" (2026-11-11,
approved by Nick the same day), on the hot Wikidata store's names and values. Each behind its own flag, off by default;
off, the store answers as before.

- **The original work first** (``CB_ORIGINAL_WORK=1``, proposal 2): a title shared by a work and works about it or
  made from it. An item that points at another item of the same name by "main subject" (P921), "based on" (P144),
  "edition or translation of" (P629), "recording or performance of" (P2550), "modified version of" (P5059) or
  "inspired by" (P941) steps aside from the fitting items, as a fictional item does (`wd_store_refine`): D. A.
  Carson's commentary "The Gospel According to John" (main subject: the gospel), Herbert Murrill's score of "Jana Gana
  Mana" (based on: the anthem). The original joins the name's items even when the name is only one of its aliases
  ("The Gospel According to John" is an alias of the Gospel of John). Then order 0037's rule decides among what is left,
  or the loop asks: the original is never picked for being the original, only its derivatives leave.
- **An alias match does not end the name search** (``CB_ALIAS_SEARCH=1``, proposal 3): a name found only as an alias
  and opening with a title ("Emperor Constantine", an alias of Constantine II alone) also gathers the items labelled
  with the rest of the name whose own description holds the title (the three Byzantine emperors labelled
  "Constantine"). The name is then shared, and the picking rules apply as to any shared name.
- **A value's own date** (``CB_OWN_DATE=1``, proposal 4): a statement with a "point in time" (P585) at year precision
  or finer keeps that year, `fact_year` gives it, and the loop says the value "as of" that year rather than the dump's
  (`ask_memory._snapshot`): Fayetteville's 208,501 is the 2020 count.

And for proposal 1 (``CB_ANSWER_FITS=1``, `cubbyllm/reasoning/ask_word_fits.py`): `value_kinds` reads what the value's
own item is, from its statements: a place (coordinates, a country, located in), a person (a human, a sex or gender, a
birth or death date), an organisation (headquarters, founder, industry, legal form, a league, a record label ...).

Wired: WIRED (`wd_store_world.WdStoreWorld`'s base; `ask_word_fits` and `ask_memory` read `value_kinds`, `fact_year`).
"""
from __future__ import annotations

import os
import re

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"

DERIVED = (921, 144, 629, 2550, 5059, 941)               # about it, based on it, an edition, a performance, a version
POINT_IN_TIME = 585
KEEP = 200_000                                           # entries each record keeps (oldest dropped)
TITLES = {"emperor": "emperor", "empress": "empress", "king": "king", "queen": "queen", "pope": "pope", "saint": "saint",
          "st": "saint", "prince": "prince", "princess": "princess", "tsar": "tsar", "czar": "tsar", "sultan": "sultan",
          "pharaoh": "pharaoh", "president": "president", "general": "general", "admiral": "admiral", "duke": "duke",
          "duchess": "duchess", "count": "count", "countess": "countess", "emir": "emir", "caliph": "caliph",
          "shah": "shah", "kaiser": "kaiser", "chancellor": "chancellor", "bishop": "bishop", "archbishop": "archbishop",
          "cardinal": "cardinal", "rabbi": "rabbi", "imam": "imam", "lord": "lord", "lady": "lady", "sir": "sir"}
PLACE = (625, 131, 17, 706, 30)                          # coordinates, located in, country, terrain feature, continent
PERSON = (21, 569, 570)                                  # sex or gender, born, died (and an instance of human)
ORGANISATION = (159, 112, 452, 1454, 749, 355, 488, 169, 1128, 2124, 118, 286, 115, 264)
HUMAN = {"Q5", "Q15632617"}                              # human, fictional human


def flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _put(d: dict, k, v) -> None:
    d[k] = v
    while len(d) > KEEP:
        d.pop(next(iter(d)))


def key3(subj: str, rel: str, obj: str) -> tuple:
    return normalize(subj or ""), normalize(rel or ""), normalize(obj or "")


def year_of(qualifiers: list, qnodes: list, nodes: dict, pid: int = POINT_IN_TIME) -> str | None:
    """The year of a statement's qualifier `pid` at year precision or finer ("2020", "500 BC"), or None."""
    node_of = {p: n for p, n in qnodes or []}
    for p, v in qualifiers or []:
        if p != pid or not isinstance(v, dict) or "typed" not in v:
            continue
        node = nodes.get(node_of.get(p) or "") or {}
        try:
            if int(node.get("wikibase:timePrecision", "")) < 9:
                return None
        except ValueError:
            return None
        m = re.match(r"^([+-])?0*(\d{1,})-", str(v["typed"][0]))
        if not m or m.group(2) == "0":
            return None
        return f"{m.group(2)} BC" if m.group(1) == "-" else m.group(2)
    return None


class Named:
    """For `WdStoreWorld` (uses its `ask`, `_entity`, `_tiers`, `_pid_label`)."""

    def _note_item(self, qid: str, statements: list) -> None:
        """What `_entity_rows` read of an item: its links to what it is made from or about, and the pids it states."""
        if not hasattr(self, "_links"):
            self._links, self._pids, self._obj_ids, self._years = {}, {}, {}, {}
        best = [s for s in statements if s.get("best") and s.get("rank") != 0]
        _put(self._links, qid, {s["value"]["entity"] for s in best if s.get("pid") in DERIVED
                                and isinstance(s.get("value"), dict) and "entity" in s["value"]})
        _put(self._pids, qid, ({s.get("pid") for s in best},
                               {s["value"]["entity"] for s in best if s.get("pid") == 31 and isinstance(s.get("value"), dict)
                                and "entity" in s["value"]}))

    def _note_row(self, subj: str, rel: str, obj: str, s: dict, nodes: dict) -> None:
        """A said row: its object's item, and its own year ("point in time")."""
        if not hasattr(self, "_links"):
            self._links, self._pids, self._obj_ids, self._years = {}, {}, {}, {}
        k = key3(subj, rel, obj)
        v = s.get("value")
        if isinstance(v, dict) and "entity" in v:
            _put(self._obj_ids, k, v["entity"])
        y = year_of(s.get("hot_qualifiers"), s.get("hot_qualifier_nodes"), nodes)
        if y:
            _put(self._years, k, y)

    # ---- proposal 4: a value's own date
    def fact_year(self, triple) -> str | None:
        """The year the source dated this fact's value at (its "point in time"), with ``CB_OWN_DATE=1``; else None."""
        if not flag("CB_OWN_DATE") or triple is None:
            return None
        return getattr(self, "_years", {}).get(key3(triple.subj, triple.rel, triple.obj))

    # ---- proposal 1: what the value's item is
    def value_kinds(self, triple) -> set | None:
        """{"place", "person", "organisation"} that the value's own item is (maybe empty), or None when the value is
        no item this world handed out (nothing to check against)."""
        if triple is None:
            return None
        q = getattr(self, "_obj_ids", {}).get(key3(triple.subj, triple.rel, triple.obj))
        if q is None:
            return None
        if q not in getattr(self, "_pids", {}):
            self._entity(q)
        pids, p31 = self._pids.get(q, (set(), set()))
        out = set()
        if pids & set(PLACE):
            out.add("place")
        if p31 & HUMAN or pids & set(PERSON):
            out.add("person")
        if pids & set(ORGANISATION):
            out.add("organisation")
        return out

    # ---- proposals 2 and 3: a name's items
    def _more_items(self, key: str, labels: list, aliases: list) -> tuple[list, list]:
        if flag("CB_ALIAS_SEARCH") and aliases and not labels:
            aliases = aliases + [q for q in self._titled(key) if q not in aliases]
        if flag("CB_ORIGINAL_WORK") and labels and aliases and len(labels) <= self.NAMESAKE_CAP:
            pointed = set().union(*(self._links_of(q) for q in labels))
            labels = labels + [q for q in aliases if q in pointed and q not in labels]
        return labels, aliases

    def _titled(self, key: str) -> list:
        """The items labelled with a titled name's rest whose description holds the title ("Emperor Constantine")."""
        words = key.split()
        title = TITLES.get(words[0]) if len(words) >= 2 else None
        if title is None:
            return []
        rest_labels, _ = self._tiers(" ".join(words[1:]))
        if not rest_labels or len(rest_labels) > 400:
            return []
        brief: dict = {}
        for i in range(0, len(rest_labels), 200):
            brief.update(self.ask({"op": "brief", "ids": rest_labels[i:i + 200]}) or {})
        rx = re.compile(rf"\b{title}\b", re.I)
        return [q for q in rest_labels if rx.search((brief.get(q) or {}).get("description") or "")]

    def _also_named(self, key: str, subj: str) -> bool:
        """`subj` is the label of an item the name gathered under these flags (the original from the alias tier, a
        titled name's namesake), whose rows a hop on the name may read."""
        if not (flag("CB_ORIGINAL_WORK") or flag("CB_ALIAS_SEARCH")):
            return False
        labels, aliases = self._tiers(normalize(key or ""))
        return any(normalize(self._entity(q)[0] or "") == normalize(subj or "") for q in labels + aliases)

    def _links_of(self, q: str) -> set:
        if q not in getattr(self, "_links", {}):
            self._entity(q)
        return getattr(self, "_links", {}).get(q, set())

    def _derived(self, items) -> list:
        """The items that point at another of the same items as their original (``CB_ORIGINAL_WORK=1``)."""
        if not flag("CB_ORIGINAL_WORK"):
            return []
        items = set(items)
        return sorted(q for q in items if self._links_of(q) & (items - {q}))


__all__ = ["DERIVED", "Named", "TITLES", "key3", "year_of"]
