"""standin/hdc_store.py -- the hdc world on disk (order 0014), and the one store that puts it behind the wiki world.

`HdcWorld`: read-only over `standin/data/hdc_world.py`'s SQLite. A subject's facts come back by its normalized label,
any of its aliases or its Q-id (the names table), as `(fact, Triple)` the way `TripleIndex` holds them; `hop` answers
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
import pathlib
import sqlite3
import threading

from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.planner import Triple, accepts, normalize, parse_fact, tail_splits

__wiring__ = "WIRED"

DEFAULT = pathlib.Path(__file__).resolve().parent / "data" / "out" / "hdc_world.sqlite"


class _Rows:
    """A mapping-like view of one key column: `.get(key, default)` -> [(fact, Triple)]."""

    def __init__(self, fetch) -> None:
        self._fetch = fetch

    def get(self, key, default=()):
        rows = self._fetch(key)
        return rows if rows else (list(default) if default else [])


class HdcWorld:
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
        self._rows = functools.lru_cache(maxsize=cache)(self._query)
        self._by_subj, self._by_obj = _Rows(lambda k: self._rows("skey", k)), _Rows(lambda k: self._rows("okey", k))

    @staticmethod
    def fact(subj: str, rel: str, obj: str) -> str:
        return f"{obj} is the {rel} of {subj}"

    def _query(self, col: str, key: str) -> list:
        key = normalize(key or "")
        with self._lock:
            got = self.db.execute(f"SELECT subj, rel, obj FROM facts WHERE {col} = ?", (key,)).fetchall()
            if not got:                                  # an alias or a Q-id: its label's facts
                for (q,) in self.db.execute("SELECT qid FROM names WHERE key = ? LIMIT 2", (key,)).fetchall()[:1]:
                    lab = self.db.execute("SELECT en FROM labels WHERE qid = ?", (q,)).fetchone()
                    if lab and normalize(lab[0]) != key:
                        got = self.db.execute(f"SELECT subj, rel, obj FROM facts WHERE {col} = ?", (normalize(lab[0]),)).fetchall()
        return [(self.fact(s, r, o), Triple(obj=o, rel=r, subj=s)) for s, r, o in got]

    def label(self, name: str) -> str | None:
        """The exact label a name (label, alias or Q-id) stands for."""
        with self._lock:
            q = self.db.execute("SELECT qid FROM names WHERE key = ?", (normalize(name),)).fetchone()
            lab = self.db.execute("SELECT en FROM labels WHERE qid = ?", (q[0],)).fetchone() if q else None
        return lab[0] if lab else None

    def hop(self, plan, hop: int, entity: str | None) -> list:
        if hop == 0:
            cands, seen = [], set()
            for exact in (True, False):
                for rel, ent in tail_splits(plan.tail):
                    for f, t in self._by_subj.get(ent):
                        same = normalize(t.rel) == normalize(rel)
                        if f not in seen and (same if exact else self._rel_n.get(normalize(t.rel), 0) >= 2):
                            cands.append((f, t)); seen.add(f)
        else:
            cands = self._by_subj.get(normalize(entity or ""))
        return [(f, t) for f, t in cands if accepts(plan, hop, entity, t)]

    def relations(self) -> dict[str, int]:
        return dict(self._rel_n)

    def __contains__(self, fact: str) -> bool:
        t = parse_fact(" ".join(fact.split()), known=self._rel_n)
        return t is not None and any(f == " ".join(fact.split()) for f, _t in self._by_subj.get(t.subj))

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


class Stores:
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
