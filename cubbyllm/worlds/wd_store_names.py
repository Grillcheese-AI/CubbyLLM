"""cubbyllm/worlds/wd_store_names.py -- order 0066's checks on the hot Wikidata store's names and values (Cowork, approved
by Nick 2026-11-11), each behind its own flag, off by default; off, the store answers as before. Classes are read
through the class-roots table (`wd_store_roots`); without it, checks 1 to 3 do nothing.

- **Check 1's kinds** (`value_kind`, read by `ask_qword` with ``CB_QWORD_KIND=1``): the value's own item is a place
  (coordinates, or a class under "geographic location"), a person, a group or an organisation (a class that is or is
  under "human", "group of humans", "organization" or "fictional character"); a time value is a date. A value that is
  no item and no time this world handed out: unknown.
- **Check 2, the original first** (``CB_ORIGINAL_FIRST=1``): a name's items, label and alias tiers read together, that
  are works (a class under "work") and point at one another ("based on" P144, "edition or translation of" P629, "main
  subject" P921, or the original's own "derivative work" P4969): the work pointed at that points at none of them is
  the original, and the name stands for it (with any item that is no work, which the existing rules still weigh). No
  original, or several: as today (a shared name is asked about). Never by fame or age.
- **Check 3, an alias match does not end the search** (``CB_ALIAS_OPEN=1``): a name that matches no label, one item
  by alias, and opens with a role word (the label of an item that is, or is an instance of, a class under "position",
  "title", "noble title" or "military rank") reads the bare name too, keeps its items that hold or held the role
  ("position held" P39, "noble title" P97, "occupation" P106: the role, or a class under it by subclass of), and the
  alias's item joins them. Then the existing rules decide, or the loop asks; a year the question names keeps the one
  whose term or life covers it (`pick_year`). None held the role: the alias stands, as today.
- **Check 4, a value's own date** (``CB_VALUE_DATE=1``): a statement's "point in time" (P585), else its "start time"
  (P580), at year precision or finer, is the year the loop says it "as of" (`ask_memory._snapshot`), not the dump's.

Wired: WIRED (`wd_store_world.WdStoreWorld`'s base; `ask_qword`, `ask_memory` and `ask_namesakes` read it).
"""
from __future__ import annotations

import os
import re

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"

POINTS = (144, 629, 921)                                 # based on, edition or translation of, main subject
DERIVATIVE = 4969                                        # the original's own "derivative work"
HELD = (39, 97, 106)                                     # position held, noble title, occupation
POINT_IN_TIME, START, END = 585, 580, 582
KEEP = 200_000                                           # entries each record keeps (oldest dropped)
ROLE_ITEMS, BARE_MAX, UP = 25, 400, 6


def flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _put(d: dict, k, v) -> None:
    d[k] = v
    while len(d) > KEEP:
        d.pop(next(iter(d)))


def key3(subj: str, rel: str, obj: str) -> tuple:
    return normalize(subj or ""), normalize(rel or ""), normalize(obj or "")


def year_of(literal) -> int | None:
    m = re.match(r"^([+-])?0*(\d+)-", str(literal or ""))
    if not m or m.group(2) == "0":
        return None
    return -int(m.group(2)) if m.group(1) == "-" else int(m.group(2))


def say_year(y: int) -> str:
    return f"{-y} BC" if y < 0 else str(y)


def qual_year(s: dict, nodes: dict, pid: int) -> int | None:
    """A statement's qualifier `pid` (a time) as a year, at year precision or finer when its node says, else None."""
    node_of = dict(s.get("hot_qualifier_nodes") or [])
    for p, v in s.get("hot_qualifiers") or []:
        if p == pid and isinstance(v, dict) and "typed" in v:
            node = nodes.get(node_of.get(p) or "") if nodes else None
            try:
                if node is not None and int(node.get("wikibase:timePrecision", "")) < 9:
                    return None
            except ValueError:
                return None
            return year_of(v["typed"][0])
    return None


def _ent(s: dict) -> str | None:
    v = s.get("value")
    return v["entity"] if isinstance(v, dict) and "entity" in v else None


class Named:
    """For `WdStoreWorld` (uses its `ask`, `_entity`, `_tiers`, `_p31`, `_roots`, `namesakes`, NAMESAKE_CAP)."""

    def _rec(self) -> dict:
        if not hasattr(self, "_named_rec"):
            self._named_rec = {k: {} for k in ("links", "derivs", "pids", "life", "held", "obj", "time", "year", "up")}
            self._opened = set()
        return self._named_rec

    def _note_item(self, qid: str, statements: list) -> None:
        r, best = self._rec(), [s for s in statements if s.get("best") and s.get("rank") != 0]
        _put(r["links"], qid, {_ent(s) for s in best if s.get("pid") in POINTS and _ent(s)})
        _put(r["derivs"], qid, {_ent(s) for s in best if s.get("pid") == DERIVATIVE and _ent(s)})
        _put(r["pids"], qid, {s.get("pid") for s in best})
        born = [year_of((s.get("value") or {}).get("typed", [None])[0]) for s in best if s.get("pid") == 569]
        died = [year_of((s.get("value") or {}).get("typed", [None])[0]) for s in best if s.get("pid") == 570]
        _put(r["life"], qid, (born[0] if born else None, died[0] if died else None))
        _put(r["held"], qid, [(_ent(s), qual_year(s, {}, START), qual_year(s, {}, END))
                              for s in statements if s.get("pid") in HELD and s.get("rank") != 0 and _ent(s)])

    def _note_row(self, subj: str, rel: str, obj: str, s: dict, nodes: dict) -> None:
        r, k = self._rec(), key3(subj, rel, obj)
        if _ent(s):
            _put(r["obj"], k, _ent(s))
        if "dateTime" in str((s.get("value") or {}).get("typed", "")):
            _put(r["time"], k, True)
        y = qual_year(s, nodes, POINT_IN_TIME)
        y = qual_year(s, nodes, START) if y is None else y
        if y is not None:
            _put(r["year"], k, y)

    # ---- check 4
    def fact_year(self, triple) -> str | None:
        """The year the source dated this fact's value at (its point in time, else its start), with
        ``CB_VALUE_DATE=1``; else None."""
        if not flag("CB_VALUE_DATE") or triple is None:
            return None
        y = self._rec()["year"].get(key3(triple.subj, triple.rel, triple.obj))
        return None if y is None else say_year(y)

    # ---- check 1
    def item_roots(self, q: str) -> set:
        if getattr(self, "_roots", None) is None:
            return set()
        self._entity(q)
        return set().union(*(self._roots.of(c) for c in self._p31.get(q, set()) | {q}))

    def value_kind(self, triple) -> set | None:
        """{"time"} for a time value; the roots of the value's item (with "place" for coordinates); None: no item and
        no time this world handed out, or no roots table (nothing to check against)."""
        if triple is None:
            return None
        r, k = self._rec(), key3(triple.subj, triple.rel, triple.obj)
        if r["time"].get(k):
            return {"time"}
        q = r["obj"].get(k)
        if q is None or getattr(self, "_roots", None) is None:
            return None
        got = self.item_roots(q) - {"role", "work"}
        return got | ({"place"} if 625 in r["pids"].get(q, set()) else set())

    # ---- checks 2 and 3: a name's items
    def _more_items(self, key: str, labels: list, aliases: list) -> tuple[list, list]:
        if getattr(self, "_roots", None) is None:
            return labels, aliases
        if flag("CB_ALIAS_OPEN"):
            more = self._alias_open(key, labels, aliases)
            if more:
                aliases = aliases + more
        if flag("CB_ORIGINAL_FIRST"):
            orig = self._original(labels, aliases)
            if orig is not None:
                works = self._works(labels + aliases)
                labels, aliases = [orig] + [q for q in labels if q not in works and q != orig], []
        return labels, aliases

    def _works(self, items) -> set:
        return {q for q in items if "work" in self.item_roots(q)}

    def _original(self, labels: list, aliases: list) -> str | None:
        items = labels + [q for q in aliases if q not in labels]
        if not 2 <= len(items) <= self.NAMESAKE_CAP:
            return None
        works = self._works(items)
        if len(works) < 2:
            return None
        r = self._rec()
        points = {q: (r["links"].get(q, set()) & works) - {q} for q in works}
        for o in works:
            for w in (r["derivs"].get(o, set()) & works) - {o}:
                points[w].add(o)
        pointed = set().union(*points.values())
        originals = [q for q in pointed if not points[q]]
        return originals[0] if len(originals) == 1 else None

    def _alias_open(self, key: str, labels: list, aliases: list) -> list:
        words = key.split()
        if labels or len(aliases) != 1 or len(words) < 2:
            return []
        roles = self._role_items(words[0])
        if not roles:
            return []
        bare, _ = self._tiers(" ".join(words[1:]))
        if not bare or len(bare) > BARE_MAX:
            return []
        held = [q for q in bare if q not in aliases and self._holds(q, roles)]
        if held:
            self._rec()
            self._opened.add(key)
        return held

    def _role_items(self, word: str) -> set:
        items, _ = self._tiers(word)
        return {q for q in items[:ROLE_ITEMS] if "role" in self.item_roots(q)}

    def _holds(self, q: str, roles: set) -> bool:
        self._entity(q)
        return any(v in roles or roles & self._up(v) for v, _s, _e in self._rec()["held"].get(q, []))

    def _up(self, c: str) -> set:
        """The classes above `c` by subclass of, at most UP steps (read from the store, remembered)."""
        memo = self._rec()["up"]
        if c in memo:
            return memo[c]
        seen, frontier = set(), {c}
        for _ in range(UP):
            nxt = set()
            for x in frontier:
                e = self.ask({"op": "get", "id": x}) or {}
                nxt |= {_ent(s) for s in e.get("statements", []) if s.get("pid") == 279 and s.get("best") and _ent(s)}
            frontier = nxt - seen
            seen |= nxt
            if not frontier or len(seen) > 200:
                break
        _put(memo, c, seen)
        return seen

    def pick_year(self, name: str, year: int) -> str | None:
        """Among a name check 3 opened, the one item whose term in the role, else whose life, covers the year."""
        key = normalize(name or "")
        if not flag("CB_ALIAS_OPEN") or key not in getattr(self, "_opened", set()):
            return None
        r, by = self._rec(), self.namesakes(name)

        def term(q):
            return any(s is not None and s <= year and (e is None or year <= e) for _v, s, e in r["held"].get(q, []))

        def life(q):
            b, d = r["life"].get(q, (None, None))
            return b is not None and d is not None and b <= year <= d
        for test in (term, life):
            hit = [q for q in by if test(q)]
            if hit:
                return hit[0] if len(hit) == 1 else None
        return None

    def _also_named(self, key: str, subj: str) -> bool:
        """`subj` is the label of an item a name gathered under checks 2 or 3, whose rows a hop on the name may read."""
        if not (flag("CB_ORIGINAL_FIRST") or flag("CB_ALIAS_OPEN")):
            return False
        labels, aliases = self._tiers(normalize(key or ""))
        return any(normalize(self._entity(q)[0] or "") == normalize(subj or "") for q in labels + aliases)


__all__ = ["Named", "key3", "qual_year", "say_year", "year_of"]
