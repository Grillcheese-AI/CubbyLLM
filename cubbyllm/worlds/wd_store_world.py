"""cubbyllm/worlds/wd_store_world.py -- the hot Wikidata store as a read-only world (order 0055, step 3).

The full Wikidata of the WikiKGQA 2026 dump (2025-12-29), read through `wdstore serve`, behind the same interface as
the Wikidata5M world (`hdc_store.HdcWorld`), which it subclasses: a hop, a subject's facts, an object's, a name's
items, an item's label. `Stores` puts it ahead of the Wikidata5M world, so the walk reads it first and the 2019 world
only for what it lacks. Built by the server behind ``CB_WD_STORE=1`` (`cubby_serve/cli.py`); off, it is never built
and every answer is what it was.

This module opens no socket and starts no process: the server hands it `ask`, a function that sends one request to
`wdstore serve` (a pipe the server owns) and returns its answer, the way a live source is handed its fetch.

What is the same as the Wikidata5M world (`hdc_namesakes`, orders 0033, 0036, 0037): a name stands for its items by
tier -- the items whose label it is, else those that carry it as an alias or a Q-id -- and a name two or more items
share gives none of their facts until the asker or a qualifier decides; nothing is ranked, an item never wins for
being famous. What differs:
- a name is matched as the Wikidata5M world's names table matches it: the store is asked for the name and for "the"
  + the name (the planner drops a leading article), its exact and accent-folded keys only, and a candidate is kept
  when one of its own English, French or `mul` labels (the label tier) or aliases (the alias tier) normalizes
  (`planner.normalize`) to the very key asked. The store's typo path is never read: a misspelt name finds nothing
  rather than something near;
- the relation-decision rule (0037) reads the store's own kinds table (`wd_store_kinds`, built from every hot item's
  classes and properties) when the server is started with ``CB_WD_KINDS=1`` and the table was built from this state
  of the store; then a shared name is decided as in Wikidata5M (the one item whose kind carries the asked relation),
  and a name with more items than `NAMESAKE_CAP` is first narrowed to the items the rule could pick
  (`wd_store_crowd`); the store's cue table (`wd_store_cues`, read with it) lets a question's words rule a kind
  out, as 0037's do ("Nobel" goes with people). Without the table, a
  shared name is held, as under ``CB_NAMESAKE_RELATION=0``, and ``CB_NAMESAKE_TIER=1``'s `applies` answers True (a
  kind may carry any relation), so its hop pick never decides either. Both err towards asking, never towards an item;
- the facts are the truthy statements, said exactly (`wd_store_values`), dated by the dump: a time-sensitive
  relation (`hdc_world.TIME_SENSITIVE`) is said "as of 2025" by the loop's snapshot rule (`ask_memory._snapshot`);
- an object's facts (who points at an item) are read up to `OBJECT_CAP` subjects, each checked against the subject's
  own truthy statements (a deprecated one, or an ended one on a relation whose value changes, is left out);
- a Wikimedia page (a disambiguation page, a category, a template, a list) is never one of a name's items: it is not
  a thing a question is about. The decision itself is refined for this world (`wd_store_refine`: a property that
  fits anything never picks; a fictional item steps aside when a real one fits; every item agreeing is said once). Without the kinds table, a name standing for more than `NAMESAKE_CAP` items gives none
  of them and lists none (full Wikidata: "Paris" is 248 items, paintings and albums among them).

Names are shared far more in the full Wikidata than in Wikidata5M ("Douglas Adams" is 10 people here, "France" 50
items), which is why the kinds table matters here; never fame.

Wired: WIRED (`cubby_serve/cli.py`, with ``CB_WD_STORE=1``).
"""
from __future__ import annotations

import functools
import re
import threading

from cubbyllm.reasoning.planner import normalize

from . import wd_store_values as vals
from .hdc_store import HdcWorld, _Rows
from .hdc_world import TIME_SENSITIVE
from .wd_store_crowd import Crowded
from .wd_store_refine import Refined

__wiring__ = "WIRED"

OBJECT_CAP = 200                                         # subjects read for an object's facts
NAMESAKE_CAP = 25                                        # more items than this under one name: none read
META = {"Q4167410", "Q22808320", "Q4167836", "Q11266439", "Q13406463", "Q14204246", "Q15184295", "Q17633526",
        "Q15407973", "Q16695773"}                        # disambiguation pages, categories, templates, lists, modules ...
_QID = re.compile(r"^q[1-9][0-9]*$")


class WdStoreWorld(Refined, Crowded, HdcWorld):
    name = "wdstore"
    NAMESAKE_CAP = NAMESAKE_CAP

    def __init__(self, ask, dump_date: str, statements: int = 0, cache: int = 50_000, kinds: str | None = None,
                 cues: str | None = None) -> None:
        """`ask`: one `wdstore serve` request -> its answer (a dict). `dump_date`: the dump's date (its manifest's).
        `kinds`: the kinds table's path (`wd_store_kinds`); `cues`: the cue table and fictional classes' path
        (`wd_store_cues`); each read when built from this state of the store."""
        self.ask = ask
        self._lock, self._pins = threading.Lock(), threading.local()
        self.snapshot = dump_date[:4]                    # the loop's snapshot rule compares a year
        self.provenance_name = f"wikidata (the WikiKGQA 2026 dump, {dump_date[:10]})"
        self.state = (ask({"op": "snapshot"}) or {}).get("snapshot")
        from .wd_store_kinds import for_store
        from .wd_store_cues import for_store as cues_for
        self._kinds = for_store(kinds, self.state)       # (no table: shared names are held)
        self._cues, self._fictional = cues_for(cues, self.state)   # (none: no cue rules out; nothing set aside)
        self._pid_label, self._datatype = {}, {}
        for lo in range(1, 14_001, 1000):                # every property: its label and datatype (2.6 s, full store)
            got = ask({"op": "brief", "ids": [f"P{i}" for i in range(lo, lo + 1000)], "lang": "en"})
            for pid, b in (got or {}).items():
                if pid.startswith("P") and b.get("label") and b.get("datatype") in vals.SAYABLE:
                    self._pid_label[int(pid[1:])], self._datatype[int(pid[1:])] = b["label"], b["datatype"]
        if not self._pid_label:
            raise RuntimeError(f"the store gave no properties: {got}")
        # every property is a relation stated many times: 2 stands for "more than once" (the paraphrase tier's guard)
        self._rel_n = {normalize(lab): 2 for lab in self._pid_label.values()}
        self._ts_pids = {int(p[1:]) for p in TIME_SENSITIVE if int(p[1:]) in self._pid_label}
        self.time_sensitive = {normalize(self._pid_label[p]) for p in self._ts_pids}
        self._n = statements
        self._p31: dict = {}                             # an item read -> its classes (ids), for the fictional rule
        self._entity = functools.lru_cache(maxsize=cache)(self._entity_rows)
        self._tiers = functools.lru_cache(maxsize=cache)(self._tiers)
        self._named = functools.lru_cache(maxsize=cache)(self._named_rows)
        self._applies = functools.lru_cache(maxsize=4096)(self._applies_rows)
        self._rows = functools.lru_cache(maxsize=cache)(self._query)
        self._by_subj, self._by_obj = _Rows(self._subject), _Rows(lambda k: self._rows("okey", k))

    # ---- the store's answers, as rows (subj, rel, obj) ----

    def _labels(self, ids) -> dict:
        ids, out = sorted(set(ids)), {}
        for lang in ("en", "mul"):
            want = [i for i in ids if i not in out]
            if want:
                got = self.ask({"op": "labels", "ids": want, "lang": lang}) or {}
                out.update({k: v for k, v in got.items() if isinstance(v, str)})
        return out

    def _entity_rows(self, qid: str) -> tuple:
        """An item's facts: (its label, [(subj, rel, obj)] of its truthy statements that can be said, {its names,
        normalized: 0 for a label, 1 for an alias} in English, French and `mul`)."""
        e = self.ask({"op": "get", "id": qid.upper(), "nodes": True}) or {}
        if not e.get("found"):
            return None, [], {}
        names = e.get("names", [])
        keys: dict = {}
        for lang, kind, text in names:
            if lang in ("en", "fr", "mul") and kind in (0, 1):
                k = normalize(text)
                keys[k] = min(kind, keys.get(k, kind))
        subj = next((n[2] for lang in ("en", "mul") for n in names if n[0] == lang and n[1] == 0), None)
        if subj is None:
            return None, [], keys                        # an item with no English or mul label is never named here
        self._p31[qid.upper()] = {s["value"]["entity"] for s in e.get("statements", []) if s.get("pid") == 31 and s.get("best")
                                  and s.get("rank") != 0 and isinstance(s.get("value"), dict) and "entity" in s["value"]}
        sts = [s for s in e.get("statements", []) if s.get("pid") in self._pid_label and vals.kept(s, s["pid"] in self._ts_pids)]
        nodes = e.get("nodes", {})
        labels = self._labels(vals.entity_ids(sts) | vals.unit_ids(sts, nodes))
        rows = []
        for s in sts:
            o = vals.say(s["value"], nodes.get(s.get("value_node") or ""), self._datatype[s["pid"]], labels)
            if o:
                rows.append((subj, self._pid_label[s["pid"]], o))
        return subj, rows, keys

    def _tiers(self, key: str) -> tuple[list, list]:
        """(the items one of whose labels a normalized name is, the items that carry it as an alias or a Q-id), each
        in id order (module doc)."""
        if _QID.match(key or ""):
            return [], [key.upper()]
        if not key:
            return [], []
        found, folded = {}, set()
        for text in (key, f"the {key}"):                 # an exact hit IS the key, normalized (no punctuation, no article)
            got = self.ask({"op": "name", "text": text}) or {}
            for h in got.get("hits", []):
                if h.get("lang") not in ("en", "fr", "mul") or h.get("kind") not in (0, 1):
                    continue
                if h.get("how") == "exact":
                    found[h["id"]] = min(h["kind"], found.get(h["id"], h["kind"]))
                elif h.get("how") == "folded":
                    folded.add(h["id"])
        if len(folded - found.keys()) > NAMESAKE_CAP:
            folded = set()                               # too many near forms to read: the exact ones only
        for q in folded - found.keys():                  # accents or punctuation: kept when its own name normalizes to it
            kind = self._entity(q)[2].get(key)
            if kind is not None:
                found[q] = kind
        if found:
            brief = self.ask({"op": "brief", "ids": sorted(found)}) or {}
            found = {q: k for q, k in found.items()                     # a Wikimedia page is not a thing
                     if q in brief and not set(brief[q].get("types", [])) & META}
        order = sorted(found, key=lambda x: int(x[1:]))
        return [q for q in order if found[q] == 0], [q for q in order if found[q] == 1]

    def _ids(self, key: str) -> tuple[str, list]:
        """("label", ids), else ("alias", ids), else ("none", [])."""
        labels, aliases = self._tiers(key)
        return ("label", labels) if labels else ("alias", aliases) if aliases else ("none", [])

    def _named_rows(self, key: str) -> tuple[str, dict]:
        """`hdc_namesakes.items` over the store: the items whose label the name is that hold facts; else those that
        carry it as an alias or a Q-id and hold facts; else ("none", {})."""
        labels, aliases = self._tiers(key)
        for tier, ids in (("label", labels), ("alias", aliases)):
            if len(ids) > NAMESAKE_CAP:
                return "none", {}                        # too many items under one name: none read, none listed
            by = {q: rows for q in ids for _subj, rows, _keys in [self._entity(q)] if rows}
            if by:
                return tier, by
        return "none", {}

    def _query(self, col: str, key: str) -> list:
        key = normalize(key or "")
        if col == "skey":
            _tier, by = self._named(key)
            return self._triples(next(iter(by.values()))) if len(by) == 1 else []
        _tier, ids = self._ids(key)                      # an object: the one item the name stands for, then who points at it
        if len(ids) != 1:
            return []
        obj_label = self._entity(ids[0])[0]
        got = self.ask({"op": "pointing_at", "value": ids[0], "limit": OBJECT_CAP}) or {}
        rows = []
        for pid, subj, rank, _best in got.get("hits", []):
            if pid not in self._pid_label or rank == 0:
                continue
            s_label, s_rows, _keys = self._entity(subj)
            row = (s_label, self._pid_label[pid], obj_label)
            if s_label and obj_label and row in s_rows:   # the subject's own truthy statement, said the same way
                rows.append(row)
        return self._triples(rows)

    def label(self, name: str) -> str | None:
        """The exact label a name (label, alias or Q-id) stands for: its one item's."""
        if self._pin(name):
            return self._pin(name)[0]
        _tier, ids = self._ids(normalize(name or ""))
        return self._entity(ids[0])[0] if len(ids) == 1 else None

    def _applies_rows(self, kind: str, rel: str) -> bool:
        if self._kinds is not None:
            return self._kinds.applies(kind, rel)
        return True                                      # no kinds table: a kind may carry any relation (module doc)
