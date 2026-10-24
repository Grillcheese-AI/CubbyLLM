"""standin/hdc_store.py -- the hdc world on disk (order 0014), and the one store that puts it behind the wiki world.

`HdcWorld`: read-only over `standin/data/hdc_world.py`'s SQLite. A subject's facts come back by its normalized label,
any of its aliases or its Q-id (the names table), as `(fact, Triple)` the way `TripleIndex` holds them -- one item's,
never a merge of a name's several (`hdc_namesakes`, order 0033); `hop` answers
a walk's hop as `TripleIndex.hop` does (the exact tier, then the paraphrase tier for a relation stated twice or
more, `planner.accepts` on both); `relations()` counts. Nothing is held in memory but an LRU of the subjects asked
about, so 4.3M facts cost the index file, not RAM.

`Stores`: the ask loop's one world made of several -- the wiki world first (in memory: where the loop writes what
it learns, its provenance and its dates), then the hdc world. `index.hop` asks each in order and keeps the first
world's facts first; `_by_subj` / `_by_obj` merge; `paths` walks the merged graph; `relations()` sums. A hdc fact's
provenance is the snapshot's ("wikidata (Wikidata5M ..., ~2019)"); its relation may be one whose value changes
(`time_sensitive`): the loop speaks it "as of 2019" (`ask.AskLoop`).

Wired: WIRED (`serve_api --world hdc`; `CUBBY_HDC_WORLD` names the file).
"""
from __future__ import annotations

import functools
import os
import sqlite3
import threading

from cubbyllm.local_paths import data_dir
from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.planner import accepts, normalize, parse_fact, tail_splits

from .hdc_cues import for_world as cues_for
from .hdc_kinds import for_world
from .hdc_namesakes import Namesakes, StoresNamesakes, object_items

__wiring__ = "WIRED"

DEFAULT = data_dir("hdc_world.sqlite")


class _Rows:
    """A mapping-like view of one key column: `.get(key, default)` -> [(fact, Triple)]."""

    def __init__(self, fetch) -> None:
        self._fetch = fetch

    def get(self, key, default=()):
        rows = self._fetch(key)
        return rows if rows else (list(default) if default else [])


class HdcWorld(Namesakes):
    name = "hdc"

    def __init__(self, path=None, cache: int = 50_000) -> None:
        self.path = str(path or os.environ.get("CUBBY_HDC_WORLD") or DEFAULT)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        meta = dict(self.db.execute("SELECT k, v FROM meta"))
        self.snapshot, self.provenance_name = meta.get("snapshot", "2019"), meta.get("provenance", "wikidata")
        rels = self.db.execute("SELECT rel, n, ts FROM rels").fetchall()
        self._rel_n = {normalize(r): n for r, n, _ts in rels}
        self.time_sensitive = {normalize(r) for r, _n, ts in rels if ts}
        self._n = sum(n for _r, n, _ts in rels)
        self._named = functools.lru_cache(maxsize=cache)(self._named_rows)
        self._applies = functools.lru_cache(maxsize=4096)(self._applies_rows)
        self._kinds = for_world(self.path)               # the kinds table, when built beside this world (order 0036)
        self._cues = cues_for(self.path)                 # the cue words' table (order 0037), likewise
        self._rows = functools.lru_cache(maxsize=cache)(self._query)
        self._pins = threading.local()                   # a seed read as one item for one walk (`pinned`, order 0033)
        self._by_subj, self._by_obj = _Rows(self._subject), _Rows(lambda k: self._rows("okey", k))

    @staticmethod
    def fact(subj: str, rel: str, obj: str) -> str:
        return f"{obj} is the {rel} of {subj}"

    def _query(self, col: str, key: str) -> list:
        key = normalize(key or "")
        if col == "skey":
            # a subject: the one item the name stands for (its label, else an alias or a Q-id). A name two or more
            # items share gives none of them (order 0033: the Globe Theatre's three items were said as one).
            _tier, by = self._named(key)
            return self._triples(next(iter(by.values()))) if len(by) == 1 else []
        with self._lock:                                 # an object: the same rule, by the item pointed at (order 0036)
            by = object_items(self.db, key)
        return self._triples(next(iter(by.values()))) if len(by) == 1 else []

    def label(self, name: str) -> str | None:
        """The exact label a name (label, alias or Q-id) stands for."""
        if self._pin(name):
            return self._pin(name)[0]
        with self._lock:
            q = self.db.execute("SELECT qid FROM names WHERE key = ?", (normalize(name),)).fetchone()
            lab = self.db.execute("SELECT en FROM labels WHERE qid = ?", (q[0],)).fetchone() if q else None
        return lab[0] if lab else None

    def hop(self, plan, hop: int, entity: str | None) -> list:
        """A hop as `TripleIndex.hop` serves it. A tail that names its entity by an alias or a Q-id ("rurtania") is
        accepted as the tail with the entity's label: the set writes every entity by one of its aliases."""
        if hop == 0:
            cands, seen = [], set()
            for exact in (True, False):
                for rel, ent in tail_splits(plan.tail):
                    lab = self.label(ent)
                    as_label = plan if not lab or normalize(lab) == ent else type(plan)(
                        relations=plan.relations, tail=f"{rel} of {lab}", n_hop=plan.n_hop, answer_class=plan.answer_class)
                    def keep(f, t, rel=rel, as_label=as_label, exact=exact):
                        same = normalize(t.rel) == normalize(rel)
                        return (same if exact else self._rel_n.get(normalize(t.rel), 0) >= 2) and accepts(as_label, hop, entity, t)
                    for f, t in self.hop_rows(ent, keep, rel):   # a shared name: the one item the hop may read (0033)
                        if f not in seen:
                            cands.append((f, t)); seen.add(f)
            return cands
        return self.hop_rows(normalize(entity or ""), lambda f, t: accepts(plan, hop, entity, t), plan.relations[hop])

    def relations(self) -> dict[str, int]:
        return dict(self._rel_n)

    def __contains__(self, fact: str) -> bool:
        """Held by any item of the subject's name: a fact is in the world whichever namesake it is about."""
        f = " ".join(fact.split())
        t = parse_fact(f, known=self._rel_n)
        _tier, by = self._named(normalize(t.subj)) if t is not None else ("none", {})
        return any(" ".join(self.fact(s, r, o).split()) == f for rows in by.values() for s, r, o in rows)   # "St.\xa0Louis"

    def __len__(self) -> int:
        return self._n


class _Merged:
    def __init__(self, views) -> None:
        self.views = views

    def get(self, key, default=()):
        out, seen = [], set()
        for v in self.views:
            for f, t in v.get(key, []):
                if f not in seen:
                    out.append((f, t)); seen.add(f)
        return out or (list(default) if default else [])


class _Index:
    """The merged index the loop walks: the first world's facts first."""
    paths = TripleIndex.paths                            # a DFS over `_by_subj`, which merges here

    def __init__(self, stores: "Stores") -> None:
        self.stores = stores
        p = stores.primary.index
        self._by_subj = _Merged([p._by_subj] + [o._by_subj for o in stores.others])
        self._by_obj = _Merged([p._by_obj] + [o._by_obj for o in stores.others])

    def __getattr__(self, name):                          # _seen, _reused, _by_tail, declare_relation, add, ...
        return getattr(self.stores.primary.index, name)

    def hop(self, plan, hop: int, entity: str | None) -> list:
        out, seen = [], set()
        for w in [self.stores.primary.index] + self.stores.others:
            for f, t in w.hop(plan, hop, entity):
                if f not in seen:
                    out.append((f, t)); seen.add(f)
        return out

    def relations(self) -> dict[str, int]:
        out = dict(self.stores.primary.index.relations())
        for o in self.stores.others:
            for r, n in o.relations().items():
                out[r] = out.get(r, 0) + n
        return out

    def about(self, subject: str) -> list:
        return self._by_subj.get(normalize(subject))

    def by_object(self, entity: str) -> list:
        return self._by_obj.get(normalize(entity))

    def __len__(self) -> int:
        return len(self.stores.primary.index) + sum(len(o) for o in self.stores.others)


class _Provenance(dict):
    """The first world's provenance, and for a fact only another world holds, that world's name."""

    def __init__(self, stores: "Stores") -> None:
        super().__init__()
        self.stores = stores

    def _own(self):
        return self.stores.primary.provenance

    def get(self, key, default=None):
        v = self._own().get(key)
        if v is None:
            for o in self.stores.others:
                if key in o:
                    return o.provenance_name
        return default if v is None else v

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key, value):
        self._own()[key] = value

    def __contains__(self, key):
        return key in self._own() or any(key in o for o in self.stores.others)

    def setdefault(self, key, value=None):
        return self._own().setdefault(key, value)

    def items(self):
        return self._own().items()

    def values(self):
        return self._own().values()

    def __len__(self):
        return len(self._own())

    def __bool__(self):
        return True


def disagreements(db_path, triples=None) -> tuple[set, dict]:
    """(subject, relation) pairs on which the wiki world and the hdc world give no common value (order 0015, addendum
    3): held -- never said -- until a rule per kind settles them. Values are compared folded (case, spacing and
    diacritics off: "Mc Cullers" and "McCullers" are one). Both relation names go in: the wiki world's noun and hdc's
    label. -> (the pairs, counts)."""
    import sqlite3
    from . import wikikg as wk
    from .hdc_world import WIKI
    from .hygiene import fold
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out, c = set(), {"compared": 0, "agree": 0, "agree_once_folded": 0, "held": 0}
    for s, r, o in triples if triples is not None else wk.load_triples():
        rel = WIKI.get(r)
        if not rel:
            continue
        subj = wk.decamel(s)
        vals = [v for (v,) in db.execute("SELECT obj FROM facts WHERE skey = ? AND rel = ?", (normalize(subj), rel))]
        if not vals:
            continue
        c["compared"] += 1
        o = wk.decamel(o)
        if normalize(o) in {normalize(v) for v in vals}:
            c["agree"] += 1
        elif fold(o) in {fold(v) for v in vals}:
            c["agree_once_folded"] += 1
        else:
            c["held"] += 1
            out |= {(normalize(subj), normalize(rel)), (normalize(subj), normalize(wk.relation_entry(r)[0]))}
    return out, c


class Stores(StoresNamesakes):
    """The wiki world (written to) before the read-only worlds behind it."""

    def __init__(self, primary, *others) -> None:
        self.primary, self.others = primary, list(others)
        self.name = "+".join([getattr(primary, "name", "wiki")] + [o.name for o in self.others])
        self.index = _Index(self)
        self.provenance = _Provenance(self)

    def __getattr__(self, name):                          # texts, times, _key, oracle, ... are the first world's
        return getattr(self.primary, name)

    def lookup(self, plan, hop, entity):
        return self.index.hop(plan, hop, entity)

    def add(self, fact: str) -> bool:
        return self.primary.add(fact)

    def __contains__(self, fact: str) -> bool:
        return fact in self.primary or any(fact in o for o in self.others)

    def held_by(self, fact: str):
        """The read-only world a fact comes from (None: the first world's own)."""
        if fact in self.primary:
            return None
        return next((o for o in self.others if fact in o), None)

    def known_relations(self, known) -> None:
        """Add the other worlds' relations to a `StoreRelations` built over the first world's facts."""
        for o in self.others:
            for r, n in o.relations().items():
                known._rels.add(r)
                known._n[r] = known._n.get(r, 0) + n
