"""cubbyllm/worlds/hdc_namesakes.py -- one name, several items, in the hdc world (order 0033).

Wikidata5M labels are not unique. Three items are labelled "Globe Theatre": the theatre in Southwark, one in Boston, and
one whose only fact is its country. Seven are labelled "William Owen", three "France", eight "Japan". Before 0033 a
label's facts were read by the label alone, so "Where is the Globe Theatre?" was answered from all three at once
("London Borough of Southwark, Boston"): two items merged into one, and a wrong answer spoken.

Here a name stands for its items, by tier: the items whose label it is; else the items that carry it as an alias or a
Q-id. A name that stands for two or more items gives none of their facts until something decides which one:

- the asker (a clarify choice names the item: `item_facts`, `pinned`);
- the question's own context, a qualifier it carries ("Bambi (film)", "the Globe Theatre in Boston"):
  `qualifier_pick`, read by `reasoning.ask_namesakes`;
- for a walk's hop, the relation the hop reads (`hop_pick`): the online source's rule (`sources.wikidata_resolve`:
  of the book, the ferry and the physicist named Marie Curie, one has a date of birth), under three guards this
  snapshot needs. That rule reads a missing claim as a claim the item cannot have; with full claims that holds, but
  here most items carry one to three facts. So the one item that holds the relation is taken only when (1) its kind
  (`instance of`) is known, (2) no other item shares a kind with it (two of a kind: the James Young incident), and
  (3) no other item is of a kind that carries the relation anywhere in this world (`applies`): then its absence there
  is what the kind is, not what the snapshot kept. Without (3), "Where is Wantage?" went to the New Jersey township,
  because the English civil parish's location is simply not in the snapshot. With them, "What is the capital of
  France?" still reads Q142: the other two Frances are a constituency (a kind with no capital) and an item of no
  known kind.

A "where / what is X?" ask names no relation, so the third does not apply to it (`reasoning.ask_namesakes`).

**Strict since order 0036** (its audit: 21 of 50 of the tier's decisions were the item a person asking means, against
a pre-registered 49; 29 could not be told from the question, mostly because a namesake with no known kind is plainly
of the chosen one's kind). A hop on a shared name now reads nothing unless the asker or a qualifier decided, and its
holders are noted, so the loop asks which. ``CB_NAMESAKE_TIER=1`` turns the relation tier back on (``hop_pick`` as
above), for an order that audits a better guard; it is off by default.

Nothing is ranked. An item never wins for being famous, or for being more completely described.

Wired: WIRED (`hdc_store.HdcWorld` takes `Namesakes` as a mixin, `hdc_store.Stores` takes `StoresNamesakes`).
"""
from __future__ import annotations

import contextlib
import os

from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

INSTANCE = "instance of"
APPLIES_SCAN = 2000                                      # a kind's members read to see whether it carries a relation

# a place qualifier ("in Boston", "Paris, Texas") matches only where an item is, not any of its values
PLACES = ("country", "located in the administrative territorial entity", "continent", "located in or next to body of water",
          "located on terrain feature", "headquarters location", "location", "country of origin")


def items(db, key: str) -> tuple[str, dict]:
    """The items a normalized name stands for, each with its rows (subj, rel, obj).

    Returns ("label", {item: rows}) for the items whose label it is; else ("alias", {...}) for the items that carry it
    as an alias or a Q-id and hold facts here; else ("none", {})."""
    by: dict = {}
    for s, r, o, q in db.execute("SELECT subj, rel, obj, sq FROM facts WHERE skey = ?", (key,)).fetchall():
        by.setdefault(q, []).append((s, r, o))
    if by:
        return "label", by
    for (q,) in db.execute("SELECT DISTINCT qid FROM names WHERE key = ?", (key,)).fetchall():
        lab = db.execute("SELECT en FROM labels WHERE qid = ?", (q,)).fetchone()
        rows = [(s, r, o) for s, r, o, sq in db.execute("SELECT subj, rel, obj, sq FROM facts WHERE skey = ?",
                                                        (normalize(lab[0]),)).fetchall() if sq == q] if lab else []
        if rows:
            by[q] = rows
    return ("alias" if by else "none"), by


def object_items(db, key: str) -> dict:
    """The items a normalized name stands for as an OBJECT, each with the rows (subj, rel, obj) that point at it (order
    0036): by the object's label; else, for an alias or a Q-id, the rows that point at that one item."""
    by: dict = {}
    for s, r, o, q in db.execute("SELECT subj, rel, obj, oq FROM facts WHERE okey = ?", (key,)).fetchall():
        by.setdefault(q, []).append((s, r, o))
    if by:
        return by
    for (q,) in db.execute("SELECT DISTINCT qid FROM names WHERE key = ?", (key,)).fetchall():
        lab = db.execute("SELECT en FROM labels WHERE qid = ?", (q,)).fetchone()
        rows = [(s, r, o) for s, r, o, oq in db.execute("SELECT subj, rel, obj, oq FROM facts WHERE okey = ?",
                                                        (normalize(lab[0]),)).fetchall() if oq == q] if lab else []
        if rows:
            by[q] = rows
    return by


def kinds(rows) -> set:
    """An item's kinds, from its `(fact, Triple)` rows."""
    return {normalize(t.obj) for _f, t in rows if normalize(t.rel) == INSTANCE}


def qualifier_pick(by: dict, qualifier: str, place: bool = False) -> str | None:
    """The one item one of whose values IS the qualifier ("Boston", "film"), or None. `place`: only the values that
    say where it is count. A value that only contains the qualifier ("London Borough of Southwark" for "London") is
    no match: a word inside a name is not the name."""
    want = normalize(qualifier or "")
    hit = [q for q, rows in by.items()
           if want and any(normalize(t.obj) == want and (not place or normalize(t.rel) in PLACES) for _f, t in rows)]
    return hit[0] if len(hit) == 1 else None


def hop_pick(held: dict, by: dict, applies) -> tuple[str | None, list]:
    """The one item whose rows a hop reads, among a name's several (`by`; `held`: each item's rows the hop keeps), or
    None; and the items that hold any. `applies(kind, relation)`: does any item of that kind carry the relation?"""
    holders = [q for q, rows in held.items() if rows]
    if len(holders) != 1:
        return None, holders
    h = holders[0]
    kh, rels = kinds(by[h]), {t.rel for _f, t in held[h]}
    if not kh:
        return None, holders                             # (1) a holder of no known kind
    for q, rows in by.items():
        kq = kinds(rows) if q != h else set()
        if kq & kh or any(applies(k, r) for k in kq for r in rels):
            return None, holders                         # (2) two of a kind; (3) a kind that carries it, missing by chance
    return h, holders


class Namesakes:
    """HdcWorld's names that stand for several items (a mixin: `db`, `_lock`, `_pins` and `fact` are HdcWorld's)."""

    def _named_rows(self, key: str) -> tuple[str, dict]:
        with self._lock:
            return items(self.db, key)

    def _triples(self, rows) -> list:
        return [(self.fact(s, r, o), Triple(obj=o, rel=r, subj=s)) for s, r, o in rows]

    def namesakes(self, name: str) -> dict:
        """{item: [(fact, Triple)]} when `name` stands for two or more items, else {}."""
        _tier, by = self._named(normalize(name or ""))
        return {q: self._triples(rows) for q, rows in by.items()} if len(by) > 1 else {}

    def pick_by(self, name: str, qualifier: str, place: bool = False) -> str | None:
        """The item among a name's several whose value is the qualifier (`qualifier_pick`), or None."""
        by = self.namesakes(name)
        return qualifier_pick(by, qualifier, place) if by else None

    def item_facts(self, name: str, item: str) -> list:
        """One item's facts, when `name` stands for it."""
        _tier, by = self._named(normalize(name or ""))
        return self._triples(by.get(item) or [])

    def applies(self, kind: str, rel: str) -> bool:
        """Does any item of `kind` carry `rel` in this world? From the kinds table when the world has one (complete;
        ``hdc_kinds``), else the kind's first `APPLIES_SCAN` members are read, which can miss a relation only a few of a
        large kind carry (and the hop then decides where it should not)."""
        return self._applies(normalize(kind), rel)

    def _applies_rows(self, kind: str, rel: str) -> bool:
        if getattr(self, "_kinds", None) is not None:            # the built table (hdc_kinds, order 0036): complete
            return self._kinds.applies(kind, rel)
        with self._lock:
            members = self.db.execute("SELECT subj, sq FROM facts WHERE okey = ? AND rel = ? LIMIT ?",
                                      (kind, INSTANCE, APPLIES_SCAN)).fetchall()
            return any(self.db.execute("SELECT 1 FROM facts WHERE skey = ? AND rel = ? AND sq = ? LIMIT 1",
                                       (normalize(s), rel, q)).fetchone() for s, q in members)

    def hop_rows(self, name: str, keep, rel: str | None = None) -> list:
        """The rows a hop reads for `name` (`keep`: the hop's own test of a row; `rel`: its relation): its one item's
        (or the item the asker or a qualifier pinned); for a name several items share, none, and the items that held
        some are noted (with ``CB_NAMESAKE_TIER=1``, those of the one item `hop_pick` allows). With
        ``CB_NAMESAKE_RELATION=1`` (order 0037, off until its audit), the one item order 0037's rule leaves
        (``hdc_relation.decide``: its kind carries the relation and goes with the question's cue words, no other's
        does), even when it holds no such fact; two or more that fit are noted to be answered each."""
        got = self._by_subj.get(name)
        if got:
            return [(f, t) for f, t in got if keep(f, t)]
        by = self.namesakes(name)
        held = {q: [(f, t) for f, t in rows if keep(f, t)] for q, rows in by.items()}
        if by and rel and os.environ.get("CB_NAMESAKE_RELATION") == "1" and getattr(self, "_kinds", None) is not None:
            from .hdc_relation import decide
            d = decide(by, self._world_rels(rel), getattr(self._pins, "cues", None) or [], self._kinds,
                       getattr(self, "_cues", None))
            self._decisions()[normalize(name)] = d
            if len(d["fit"]) == 1:
                return held[d["fit"][0]]
            self._notes()[normalize(name)] = d["fit"] or [h for h, rows in held.items() if rows]
            return []
        if os.environ.get("CB_NAMESAKE_TIER") == "1":        # the relation tier, off since 0036's audit (module doc)
            q, holders = hop_pick(held, by, self.applies) if by else (None, [])
        else:
            q, holders = None, [h for h, rows in held.items() if rows]
        if q is None and holders:
            self._notes()[normalize(name)] = holders
        return held[q] if q else []

    def _notes(self) -> dict:
        if getattr(self._pins, "withheld", None) is None:
            self._pins.withheld = {}
        return self._pins.withheld

    def _decisions(self) -> dict:
        if getattr(self._pins, "decided", None) is None:
            self._pins.decided = {}
        return self._pins.decided

    def decision(self, name: str) -> dict | None:
        """Order 0037's decision on a shared name in the walk just made (``hdc_relation.decide``), or None."""
        return self._decisions().get(normalize(name or ""))

    def _world_rels(self, rel: str) -> set:
        """This world's relations a question's relation names: the same words, or near enough (``relation_matches``)."""
        from cubbyllm.reasoning.planner import relation_matches
        return {r for r in self._rel_n if normalize(r) == normalize(rel) or relation_matches(rel, r)}

    @contextlib.contextmanager
    def cued(self, words):
        """Inside, a hop's decision on a shared name reads the question's cue words (order 0037). Per thread."""
        old = getattr(self._pins, "cues", None)
        self._pins.cues = list(words or [])
        try:
            yield
        finally:
            self._pins.cues = old

    def withheld(self, name: str) -> list:
        """The items whose facts a hop on `name` would have read and did not (nothing decided which)."""
        return list(self._notes().get(normalize(name or ""), []))

    def clear_withheld(self) -> None:
        self._pins.withheld = {}
        self._pins.decided = {}

    def _pin(self, key: str):
        return (getattr(self._pins, "map", None) or {}).get(normalize(key or ""))

    def _subject(self, key: str) -> list:
        """A subject's facts as `_by_subj` serves them; a name pinned for the walk under way reads as its one item."""
        pin = self._pin(key)
        return self.item_facts(*pin) if pin else self._rows("skey", key)

    @contextlib.contextmanager
    def pinned(self, key: str, name: str, item: str):
        """Inside, `key` (the walk's seed) reads as `item`, one of the items `name` stands for: the asker's choice, or
        the question's qualifier, for one walk. Per thread, so another ask is never pinned by this one."""
        old = getattr(self._pins, "map", None)
        self._pins.map = {**(old or {}), normalize(key): (name, item)}
        try:
            yield
        finally:
            self._pins.map = old


class StoresNamesakes:
    """`hdc_store.Stores`'s view of its read-only worlds' shared names (a mixin: `others` is Stores')."""

    def _all(self, attr):
        return [getattr(o, attr) for o in self.others if hasattr(o, attr)]

    def namesakes(self, name: str) -> dict:
        """A read-only world's items for a name two or more share: {item: [(fact, Triple)]}."""
        return next((got for got in (fn(name) for fn in self._all("namesakes")) if got), {})

    def item_facts(self, name: str, item: str) -> list:
        return next((got for got in (fn(name, item) for fn in self._all("item_facts")) if got), [])

    def pick_by(self, name: str, qualifier: str, place: bool = False) -> str | None:
        return next((q for q in (fn(name, qualifier, place) for fn in self._all("pick_by")) if q), None)

    def withheld(self, name: str) -> list:
        return next((got for got in (fn(name) for fn in self._all("withheld")) if got), [])

    def decision(self, name: str) -> dict | None:
        return next((got for got in (fn(name) for fn in self._all("decision")) if got), None)

    def cued(self, words):
        """Every read-only world reads the question's cue words, for the walk inside."""
        stack = contextlib.ExitStack()
        for o in self.others:
            if hasattr(o, "cued"):
                stack.enter_context(o.cued(words))
        return stack

    def clear_withheld(self) -> None:
        for fn in self._all("clear_withheld"):
            fn()

    def pinned(self, key: str, name: str, item: str):
        """Every world that holds `item` under `name` reads `key` as it, for the walk inside."""
        stack = contextlib.ExitStack()
        for o in self.others:
            if hasattr(o, "pinned") and o.item_facts(name, item):
                stack.enter_context(o.pinned(key, name, item))
        return stack
