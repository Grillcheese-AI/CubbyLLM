"""standin/worlds_store.py -- one store interface over every world the ingestion wrote (order 0015).

`WorldsStore(root)` reads the registry (`worlds_ingest`) and gives:
- `reality()`: the ask loop's store -- the wiki world first, then the hdc world (`hdc_store.Stores`); the passage worlds
  are sources a loop reads through `book_source.BookSource`, by `passage_index(world)`;
- `fiction`: the fiction worlds (`worlds_fiction.FictionWorlds`), consulted only when a question names its work;
- `lookup(entity, relation, worlds=None)`: facts about an entity across the worlds that hold facts as rows (the hdc
  world, the fiction worlds, the events and quotes worlds), filtered by world -- reality ones unless a fiction world
  is named in `worlds`;
- `ask(question, loop_for)`: a question that names a fiction work to that work's world, every other to reality.

Wired: STANDALONE (`validation/exp_worlds_bench.py`; the serve stack keeps its own mounts until a world is promoted).
"""
from __future__ import annotations

import pathlib
import sqlite3

__wiring__ = "STANDALONE"


class WorldsStore:
    def __init__(self, root) -> None:
        self.root = pathlib.Path(root)
        self.reg = sqlite3.connect(f"file:{self.root / 'registry.sqlite'}?mode=ro", uri=True, check_same_thread=False)
        self.worlds = {w: (kind, store, tier, as_of) for w, kind, store, tier, as_of in self.reg.execute("SELECT * FROM worlds")}
        fic = self.root / "fiction" / "fiction.sqlite"
        from .fiction import FictionWorlds
        self.fiction = FictionWorlds(fic) if fic.exists() else None
        self._reality = None

    def store_of(self, world: str) -> str | None:
        return (self.worlds.get(world) or (None, None))[1]

    def passage_index(self, world: str):
        from .books import BookIndex
        path = self.store_of(world)
        return BookIndex(path) if path and pathlib.Path(path).exists() else None

    def reality(self):
        """The wiki world, then the hdc world behind it (order 0014's composite)."""
        if self._reality is None:
            from . import wikikg as wk
            from .hdc_store import HdcWorld, Stores
            hdc = self.store_of("reality")
            if hdc and pathlib.Path(hdc).exists():
                from .hdc_store import disagreements
                self._reality = Stores(wk.wiki_world(), HdcWorld(hdc))
                self._reality.contested, self.disagreement_counts = disagreements(hdc)   # held until a rule per kind
            else:
                self._reality = wk.wiki_world()
        return self._reality

    def lookup(self, entity: str, relation: str | None = None, worlds=None) -> list[dict]:
        """Rows about `entity` (and `relation`), each with its world. `worlds`: the worlds to read (default: reality's)."""
        from cubbyllm.reasoning.planner import normalize
        out, want = [], set(worlds) if worlds else None
        hdc = self.store_of("reality")
        if hdc and (want is None or "reality" in want):
            db = sqlite3.connect(f"file:{hdc}?mode=ro", uri=True)
            q = "SELECT subj, rel, obj FROM facts WHERE skey = ?" + (" AND rel = ?" if relation else "")
            out += [{"world": "reality", "subj": s, "rel": r, "obj": o}
                    for s, r, o in db.execute(q, (normalize(entity),) + ((relation,) if relation else ()))]
        if self.fiction is not None and want:
            for w in want:
                if w.startswith("fiction:"):
                    q = "SELECT subj, rel, obj FROM fiction_facts WHERE work = ? AND lower(subj) = lower(?)"
                    out += [{"world": w, "subj": s, "rel": r, "obj": o} for s, r, o in self.fiction.db.execute(q, (w[8:], entity))
                            if relation is None or r == relation]
        quotes = self.root / "quotes.sqlite"
        if quotes.exists() and (want is None or "reality:quotes" in want) and relation in (None, "quote"):
            db = sqlite3.connect(f"file:{quotes}?mode=ro", uri=True)
            out += [{"world": "reality:quotes", "subj": s, "rel": "quote", "obj": q, "tier": "attributed"}
                    for s, q in db.execute("SELECT speaker, quote FROM quotes WHERE lower(speaker) = lower(?)", (entity,))]
        return out

    def ask(self, question: str, loop_for) -> dict:
        reality = loop_for(self.reality())
        if self.fiction is None:
            return {**reality.ask(question), "world": "reality"}
        return self.fiction.ask(question, reality, loop_for)
