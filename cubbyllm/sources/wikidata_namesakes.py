"""cubbyllm/sources/wikidata_namesakes.py -- a shared name at the source decided by the question's relation (order
0043: order 0037's rule on the Wikidata side; the world's side is ``worlds.hdc_relation``).

Several exact hits for a name ("uk": Uttarakhand and the United Kingdom; "Georgia": a state, a country, a given name).
Which of them fit the hop asked:
- **its kind** is its ``instance of`` (P31); a candidate with none is of no kind;
- **the kind carries the relation** when the candidate's own claims carry the property ("holds"); otherwise when the
  property's subject-type constraint (P2302 of kind Q21503250, its classes P2308) names a class the candidate's kind is
  or is one ``subclass of`` step under ("may"); otherwise it does not. The constraints are the property's own claims,
  read through the source's cache like any item's;
  A kind that only may carry it yields to a different kind whose item holds it (the novel over its editions); of one
  kind it competes all the same (the James Young rule);
- **placed** (0037's KATA guard): for a place relation (``PLACE``: the country, located in, location) every candidate
  whose claims place it competes, whatever its kind;
- **the label over an alias** (the source's own tier, kept): among several that fit and are of different kinds, the
  one labelled with the name over those that only carry it as an alias ("Spain" over "Second Spanish Republic");
  never between two of a kind, never between two labelled alike (Georgia, Georgia);
- **cue words** (0037 item 4): the question's capitalised words that are words of some candidate's kind only rule out
  (a candidate of no kind is never ruled out by one).

The outcome: one fits, the source resolves to it and names the others by kind (the loop's offer line); two or more
fit, the loop answers each with the source pinned to one item at a time (``reasoning.ask_source_namesakes``); none, the
strict refusal stands. The asker's ledger and a qualifier come first (``wikidata_resolve``). Nothing is ranked: fame,
sitelinks, search order and how completely an item is described decide nothing; two of a kind are never separated by
which is better described (the James Young rule), they are answered each.

Behind ``CB_SOURCE_NAMESAKES=1`` (off until Cowork's reading of the audit, ``validation/audit_source_namesakes.py``).

Wired: WIRED (a mixin of ``WikidataSource``; ``wikidata_resolve.WikiResolve.resolve``).
"""
from __future__ import annotations

import contextlib
import os
import threading

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"

INSTANCE, SUBCLASS = "P31", "P279"
CONSTRAINT, SUBJECT_TYPE, CLASSES = "P2302", "Q21503250", "P2308"
PLACE = ("P17", "P131", "P276")                          # the country, located in the administrative entity, location
KIND_PROPS = ("P31", "P106", "P27", "P17", "P131", "P495", "P136", "P279")   # the facts an item is told apart by


def on() -> bool:
    return os.environ.get("CB_SOURCE_NAMESAKES") == "1"


def _ids(claims: dict, pid: str) -> list[str]:
    out = []
    for st in claims.get(pid, []):
        v = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(v, dict) and v.get("id"):
            out.append(v["id"])
    return out


class WikiNamesakes:
    """WikidataSource's side of the rule (a mixin): the decision, a pin for one walk, the question's cue words."""

    def _ns(self):
        if getattr(self, "_ns_state", None) is None:
            self._ns_state = threading.local()
        return self._ns_state

    @contextlib.contextmanager
    def pinned(self, entity: str, qid: str, label: str | None = None):
        """Inside, `entity` resolves to `qid` (one of the items the rule left) and nothing is written to the askers'
        ledger: the loop answering each namesake apart. Per thread."""
        from .wikidata_resolve import name_key
        ns = self._ns()
        old = getattr(ns, "pins", None)
        ns.pins = {**(old or {}), name_key(entity): (qid, label)}
        try:
            yield
        finally:
            ns.pins = old

    def pin_of(self, entity: str):
        from .wikidata_resolve import name_key
        return (getattr(self._ns(), "pins", None) or {}).get(name_key(entity))

    @contextlib.contextmanager
    def cued(self, words):
        ns = self._ns()
        old = getattr(ns, "cues", None)
        ns.cues = list(words or [])
        try:
            yield
        finally:
            ns.cues = old

    def namesake_decision(self, entity: str) -> dict | None:
        """The rule's decision on `entity` in the walk under way (per thread), or None."""
        from .wikidata_resolve import name_key
        return (getattr(self._ns(), "decided", None) or {}).get(name_key(entity))

    def _remember(self, entity: str, d: dict) -> None:
        from .wikidata_resolve import name_key
        ns = self._ns()
        if getattr(ns, "decided", None) is None:
            ns.decided = {}
        ns.decided[name_key(entity)] = d

    def clear_decisions(self) -> None:
        self._ns().decided = {}

    # -- the rule -------------------------------------------------------------------------------
    def constraint_classes(self, pid: str) -> set:
        """The classes a property's subject-type constraint names (its own claims, cached as any item's)."""
        out = set()
        for st in self._claims(pid).get(CONSTRAINT, []):
            kind = ((st.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
            if kind != SUBJECT_TYPE:
                continue
            for q in (st.get("qualifiers") or {}).get(CLASSES, []):
                v = (q.get("datavalue") or {}).get("value")
                if isinstance(v, dict) and v.get("id"):
                    out.add(v["id"])
        return out

    def carries(self, claims: dict, kinds: list[str], pids: list[str]) -> str | None:
        """"holds" when the candidate's own claims carry one of `pids`; "may" when a constraint's class is one of its
        kinds or one `subclass of` step above them; None otherwise."""
        if any(p in claims for p in pids):
            return "holds"
        if not kinds:
            return None
        allowed = set().union(*(self.constraint_classes(p) for p in pids))
        if not allowed:
            return None
        if allowed & set(kinds):
            return "may"
        above = set().union(*(set(_ids(self._claims(k), SUBCLASS)) for k in kinds))
        return "may" if allowed & above else None

    def decide(self, entity: str, cands: list[dict], relations: list[str]) -> dict | None:
        """Order 0043's rule over the exact hits (`cands`: search hits) for a hop on `relations`. None when the
        relations name no property. Otherwise {"fit": [qids, in search order], "how": {qid: holds | may | placed},
        "kinds": {qid: [kind labels]}, "ruled_out": {qid: the cue word}, "others": [(qid, label, kind)],
        "labels": {qid: label}, "rels": relations, "cues": [the words used]}."""
        pids = [p for r in relations for p in (self.aliases.pids(r) if self.aliases is not None else [])]
        if not pids:
            return None
        place = bool(set(pids) & set(PLACE))
        claims = {h["id"]: self._claims(h["id"]) for h in cands}
        kinds = {q: _ids(c, INSTANCE) for q, c in claims.items()}
        names = self._labels_for([k for ks in kinds.values() for k in ks])
        words = {q: {w for k in ks for w in normalize(names.get(k, "")).split()} for q, ks in kinds.items()}
        cues = [w for w in (getattr(self._ns(), "cues", None) or []) if any(w in ws for ws in words.values())]
        d = {"fit": [], "how": {}, "kinds": {q: [names.get(k, k) for k in ks] for q, ks in kinds.items()}, "ruled_out": {},
             "others": [], "yields": [], "labels": {h["id"]: h.get("label") or entity for h in cands}, "rels": list(relations),
             "cues": cues}
        hows = {}
        for h in cands:
            q = h["id"]
            how = self.carries(claims[q], kinds[q], pids)
            if how is None and place and any(p in claims[q] for p in PLACE):
                how = "placed"
            if how is not None:
                hows[q] = how
        holders = [q for q, how in hows.items() if how == "holds"]
        for h in cands:
            q = h["id"]
            how = hows.get(q)
            if how is None:
                continue
            # a kind that MAY carry the relation yields to another kind whose item HOLDS it (the novel over its
            # editions, the painting over two people named Mona Lisa); of one kind it competes all the same (the
            # James Young rule: two people are never separated by which is better described)
            if how == "may" and holders and not any(set(kinds[q]) & set(kinds[o]) for o in holders):
                d["yields"].append(q)
                continue
            miss = next((w for w in cues if kinds[q] and w not in words[q]), None)
            if miss:
                d["ruled_out"][q] = miss
                continue
            d["fit"].append(q); d["how"][q] = how
        d["others"] = [(h["id"], d["labels"][h["id"]], d["kinds"][h["id"]][0]) for h in cands
                       if h["id"] not in d["fit"] and d["kinds"][h["id"]]]
        return d

    def by_label(self, entity: str, d: dict, cands: list[dict]) -> str | None:
        """Among several that fit and are of different kinds, the one whose LABEL is the name over those that carry
        it as an alias ("Spain" over "Second Spanish Republic"; the source's tier since 2026-09-14), or None: two of
        a kind are never separated so (the James Young rule), nor are two labelled alike (Georgia, Georgia)."""
        from .wikidata_resolve import name_key
        fit = [h for h in cands if h["id"] in d["fit"]]
        if len(fit) < 2 or self._one_kind(fit):
            return None
        labelled = [h["id"] for h in fit if name_key(h.get("label", "")) == name_key(entity)]
        return labelled[0] if len(labelled) == 1 else None

    def item_rows(self, qid: str, label: str) -> list:
        """A namesake's own facts that tell it apart (``KIND_PROPS``), as (fact, Triple) rows the loop labels by."""
        from cubbyllm.reasoning.planner import Triple
        claims = self._claims(qid)
        ids = [i for p in KIND_PROPS for i in _ids(claims, p)]
        names = self._labels_for(list(KIND_PROPS) + ids)
        rows = []
        for p in KIND_PROPS:
            rel = names.get(p)
            for i in _ids(claims, p):
                if rel and names.get(i):
                    rows.append((f"{names[i]} is the {rel} of {label}", Triple(obj=names[i], rel=rel, subj=label)))
        return rows


__all__ = ["KIND_PROPS", "PLACE", "WikiNamesakes", "on"]
