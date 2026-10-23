"""Scene -- named things at cells: exact geometry for the walk, one Hilbert-space vector for
the trunk, and the facts the ask loop fetches from both.

Wired: STANDALONE (2026-10-17, Nick: "to build a village geo spatial understanding is important").

Two readings of the same scene, kept in step by construction:

* **exact** -- a dict of cells. ``where``, ``at``, ``adjacent``, ``nearest``, ``within``,
  ``offset`` ("3 east of the well") and ``path`` (a shortest walk over free cells) are plain
  geometry, and they are what the VM is given: a spatial fact is admitted because the host
  computed it, never because something said it;
* **vector** -- ``V = sum_i code(name_i) (*) S(cell_i)``, one ``SSP`` vector for the whole
  scene. "What is at p" is ``V (*) S(p)^-1`` against the codes, "where is X" is
  ``V (*) code(X)^-1`` decoded on the lattice, "anything in this box" is one inner product
  with the box's closed-form vector. These are the trunk's reading -- the same algebra
  ``cubbyllm.model.spatial`` binds into its embeddings -- and ``tests/spatial/test_scene.py``
  measures how far they agree with the exact reading (a capacity, not a theorem).

``facts(entity)`` is the ask loop's Source protocol (``cubbyllm.reasoning.learn``): template
lines ``OBJ is the REL of SUBJ`` derived on every call from the exact reading, never stored
here, so the loop fetches, gates, stores with provenance and verifies them like any other.
Relations stated: ``position``, ``<direction> neighbour`` for the six faces, ``nearest
<kind>`` per kind in the scene, ``region`` for a named box the thing lies in.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from ..core.protocols import Wiring
from . import hilbert
from .ssp import SSP

__all__ = ["Scene", "DIRECTIONS", "Cell"]

Cell = tuple[int, int, int]

#: compass of the lattice: x east, y north, z up (BlockWorld's; z = 0 is the floor)
DIRECTIONS: dict[str, Cell] = {"east": (1, 0, 0), "west": (-1, 0, 0), "north": (0, 1, 0),
                               "south": (0, -1, 0), "above": (0, 0, 1), "below": (0, 0, -1)}


def _norm(name: str) -> str:
    n = " ".join(str(name).lower().split())
    for art in ("the ", "a ", "an "):
        if n.startswith(art):
            n = n[len(art):]
    return n


def _cell(c) -> Cell:
    x, y, z = (int(v) for v in c)
    return (x, y, z)


class Scene:
    """Things at cells. ``ssp`` is the space the vector lives in (``SSP()`` by default)."""

    name = "scene"

    def __init__(self, ssp: SSP | None = None, free=None) -> None:
        self.ssp = ssp or SSP()
        self.things: dict[str, Cell] = {}          # normalized name -> cell
        self.kinds: dict[str, str | None] = {}
        self.cells: dict[Cell, str] = {}
        self.regions: dict[str, tuple[Cell, Cell]] = {}
        self._codes: dict[str, np.ndarray] = {}
        self.vector = np.zeros(self.ssp.d)
        self.free = free                            # predicate cell -> bool for `path`; None = any unoccupied cell

    # -- codes and the vector ---------------------------------------------------------
    def code(self, name: str) -> np.ndarray:
        """A fixed random unit vector per name (from a hash of it, so every process agrees)."""
        key = _norm(name)
        if key not in self._codes:
            seed = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
            v = np.random.default_rng(seed).standard_normal(self.ssp.d)
            self._codes[key] = v / np.linalg.norm(v)
        return self._codes[key]

    def place(self, name: str, cell, kind: str | None = None) -> None:
        """Put ``name`` at ``cell`` (moving it if it was elsewhere); the vector follows."""
        key, c = _norm(name), _cell(cell)
        if c in self.cells and self.cells[c] != key:
            raise ValueError(f"cell {c} already holds {self.cells[c]!r}")
        if key in self.things:
            self.remove(key)
        self.things[key], self.kinds[key], self.cells[c] = c, kind, key
        self.vector = self.vector + self.ssp.bind(self.code(key), self.ssp.encode(np.asarray(c, float)))

    def remove(self, name: str) -> None:
        key = _norm(name)
        c = self.things.pop(key)
        self.kinds.pop(key, None)
        del self.cells[c]
        self.vector = self.vector - self.ssp.bind(self.code(key), self.ssp.encode(np.asarray(c, float)))

    def region(self, name: str, lo, hi) -> None:
        """Name a box (inclusive corners), for ``within`` and the ``region`` fact."""
        self.regions[_norm(name)] = (_cell(lo), _cell(hi))

    @classmethod
    def from_json(cls, obj, ssp: SSP | None = None) -> "Scene":
        """A scene from a scene file: a dict, a JSON string or a path (``scene_file``, order 0034)."""
        from .scene_file import from_json
        return from_json(obj, ssp)

    def to_json(self) -> dict:
        """This scene as a scene file's dict (``scene_file.FORMAT``)."""
        from .scene_file import to_json
        return to_json(self)

    # -- exact reading ------------------------------------------------------------------
    def where(self, name: str) -> Cell | None:
        return self.things.get(_norm(name))

    def at(self, cell) -> str | None:
        return self.cells.get(_cell(cell))

    def offset(self, name: str, direction: str, n: int = 1) -> Cell | None:
        """The cell ``n`` steps ``direction`` from ``name`` ("3 east of the well")."""
        c = self.where(name)
        d = DIRECTIONS[direction]
        return None if c is None else (c[0] + n * d[0], c[1] + n * d[1], c[2] + n * d[2])

    def adjacent(self, name: str) -> dict[str, str]:
        """``{direction: neighbour name}`` for the faces of ``name`` that hold something."""
        out = {}
        for direction in DIRECTIONS:
            c = self.offset(name, direction)
            if c is not None and c in self.cells:
                out[direction] = self.cells[c]
        return out

    @staticmethod
    def distance(a: Cell, b: Cell) -> int:
        """Steps along the lattice (Manhattan)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def nearest(self, name: str, kind: str | None = None) -> str | None:
        """The nearest other thing (of ``kind`` when given); ties by name, so deterministic."""
        c = self.where(name)
        if c is None:
            return None
        best = None
        for other, oc in sorted(self.things.items()):
            if other == _norm(name) or (kind is not None and self.kinds.get(other) != kind):
                continue
            d = self.distance(c, oc)
            if best is None or d < best[0]:
                best = (d, other)
        return None if best is None else best[1]

    def within(self, lo, hi) -> list[str]:
        """Names in the box, in curve order (so a listing reads as a walk through it)."""
        lo, hi = _cell(lo), _cell(hi)
        inside = [c for c in self.cells if all(lo[i] <= c[i] <= hi[i] for i in range(3))]
        if not inside:
            return []
        return [self.cells[tuple(int(v) for v in c)] for c in hilbert.order(np.array(inside))]

    def regions_of(self, name: str) -> list[str]:
        c = self.where(name)
        if c is None:
            return []
        return sorted(r for r, (lo, hi) in self.regions.items() if all(lo[i] <= c[i] <= hi[i] for i in range(3)))

    def path(self, a, b, limit: int = 100_000) -> list[Cell] | None:
        """A shortest 6-connected walk from cell ``a`` to cell ``b`` over free cells (the
        destination may be occupied: a walk *to* a thing). ``free`` decides what is walkable;
        by default any cell holding nothing. None when no walk exists within ``limit`` cells."""
        a, b = _cell(a), _cell(b)
        free = self.free or (lambda c: c not in self.cells)
        prev: dict[Cell, Cell | None] = {a: None}
        q = deque([a])
        while q and len(prev) <= limit:
            c = q.popleft()
            if c == b:
                out = []
                while c is not None:
                    out.append(c)
                    c = prev[c]
                return out[::-1]
            for d in DIRECTIONS.values():
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n not in prev and (n == b or free(n)):
                    prev[n] = c
                    q.append(n)
        return None

    def serialize(self, lo, hi) -> list[tuple[Cell, str]]:
        """The things in the box as ``(cell, name)`` along the Hilbert curve: the order the
        trunk reads them in, each with the cell its embedding is bound to."""
        return [(self.things[n], n) for n in self.within(lo, hi)]

    # -- vector reading -----------------------------------------------------------------
    def recall_at(self, cell) -> tuple[str | None, float]:
        """The name whose code best explains ``V (*) S(cell)^-1``, and its similarity."""
        if not self.things:
            return None, 0.0
        probe = self.ssp.unbind(self.vector, self.ssp.encode(np.asarray(_cell(cell), float)))
        names = sorted(self.things)
        s = np.stack([self._codes[n] for n in names]) @ probe
        i = int(np.argmax(s))
        return names[i], float(s[i])

    def lattice(self, extra=(), max_cells: int = 65_536) -> np.ndarray:
        """The candidate cells a place is read back over: every cell of the scene's bounding
        box (with ``extra`` corners), along the curve; past ``max_cells`` of them, the
        occupied cells alone (a decode over a whole large world is the index's job)."""
        cs = np.array(list(self.cells) + [_cell(c) for c in extra])
        lo, hi = cs.min(axis=0), cs.max(axis=0)
        if int(np.prod(hi - lo + 1)) > max_cells:
            return np.array(list(self.cells))
        return hilbert.curve(lo, hi)

    def recall_where(self, name: str, candidates=None) -> tuple[Cell | None, float]:
        """The cell whose ``S`` best explains ``V (*) code(name)^-1``, over ``candidates``
        (the scene's lattice by default), and its similarity (1 for a clean read)."""
        key = _norm(name)
        if key not in self._codes or not self.things:
            return None, 0.0
        probe = self.ssp.unbind(self.vector, self.code(key))
        c, s = self.ssp.decode(probe, self.lattice() if candidates is None else candidates)
        return _cell(c), s

    def recall_within(self, lo, hi, floor: float = 0.5) -> list[str]:
        """Names whose recalled cell lies in the box, read with similarity above ``floor``."""
        lo, hi = _cell(lo), _cell(hi)
        cands = self.lattice(extra=(lo, hi))
        out = []
        for name in sorted(self.things):
            c, s = self.recall_where(name, cands)
            if s > floor and all(lo[i] <= c[i] <= hi[i] for i in range(3)):
                out.append(name)
        return out

    # -- the Source protocol ------------------------------------------------------------
    def relations(self, text: str) -> list[str]:
        """The relation labels this source states that ``text`` names (lever 4's table tier)."""
        t = _norm(text)
        labels = ["position", "region"] + [f"{d} neighbour" for d in DIRECTIONS]
        labels += [f"nearest {k}" for k in sorted({k for k in self.kinds.values() if k})]
        return [r for r in labels if t == r or t == r.replace(" neighbour", "")]

    def kind(self, label: str) -> str | None:
        return "place" if _norm(label) in ("position", "region") else "thing"

    def wordings(self, label: str) -> list[str]:
        r = _norm(label)
        return [r, r.replace(" neighbour", " of")] if r.endswith(" neighbour") else [r]

    def facts(self, entity: str, relations: list[str] | None = None) -> list[str]:
        """Template facts about ``entity``, derived now from the exact reading."""
        key = _norm(entity)
        c = self.things.get(key)
        if c is None:
            return []
        subj = key
        out = [f"{c[0]} {c[1]} {c[2]} is the position of {subj}"]
        for direction, other in self.adjacent(key).items():
            out.append(f"{other} is the {direction} neighbour of {subj}")
        for k in sorted({k for k in self.kinds.values() if k}):
            near = self.nearest(key, k)
            if near is not None:
                out.append(f"{near} is the nearest {k} of {subj}")
        for r in self.regions_of(key):
            out.append(f"{r} is the region of {subj}")
        if relations:
            wanted = {_norm(r) for r in relations}
            first = [f for f in out if any(f" is the {w} of " in f for w in wanted)]
            out = first + [f for f in out if f not in first]
        return out


__wiring__ = Wiring.STANDALONE
