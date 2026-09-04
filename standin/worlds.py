"""worlds — incremental fact stores, world routing, and the plugin surface.

Wired: WIRED (stand-in serve path; nothing in cubbyllm/ imports this).

`FactStore` is the incremental successor to exp_m3's `make_retriever` closure
(same math: unit-cosine over `enc.encode(text)` rows) with `add()` so the
serve loop can LEARN — a fact accepted at turn N is retrievable at turn N+1.
Without an encoder it falls back to IDF-weighted token overlap (pure python),
so tests and encoder-less hosts still get a working store.

`CubbyPlugin` is the mount contract — MindForge style: a plugin sits ON TOP
of the trunk and specializes it; the trunk never imports the plugin.
cubbyverse implements this in its own repo (its world models become mounted
worlds; its web brain reads /state) and `CubbyServe.mount()` attaches it.
A plugin contributes named worlds and may observe finished turns; it never
gets to inject replies — every spoken reply still goes through CubbyTalk's
ASK and the voice rules, whatever is mounted.
"""
from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

__wiring__ = "WIRED"


class FactStore:
    """A named, growable retrieval world: (query, k) -> [(score, fact)].

    With `enc` (a FastWordEncoder-shaped object): unit-cosine over encoded
    rows, mirroring exp_m3_cot_pipeline.make_retriever. Without: IDF-weighted
    token overlap. Callable, so anything that took a retriever takes a store.
    """

    def __init__(self, texts: list[str] | None = None, enc=None, name: str = "facts") -> None:
        from cubbyllm.reasoning import TripleIndex
        self.name = name
        self.enc = enc
        self.texts: list[str] = []
        self._rows: list = []                            # np row vectors when enc is set
        self._seen: set[str] = set()
        self.index = TripleIndex()                       # retrieval as LOOKUP for template facts (exp_m4, 2026-09-04)
        for t in texts or []:
            self.add(t)

    def __len__(self) -> int:
        return len(self.texts)

    @staticmethod
    def _key(text: str) -> str:
        return " ".join(text.split())

    def add(self, text: str) -> bool:
        """Append one fact; False if it is already stored (whitespace-normalized)."""
        key = self._key(text)
        if not key or key in self._seen:
            return False
        self._seen.add(key)
        self.texts.append(key)
        self.index.add(key)                              # a learned fact is looked up next turn, not only searched
        if self.enc is not None:
            import numpy as np
            v = self.enc.encode(key).reshape(-1).astype(np.float32)
            self._rows.append(v / (np.linalg.norm(v) + 1e-12))
        return True

    def __contains__(self, text: str) -> bool:
        return self._key(text) in self._seen

    def lookup(self, plan, hop: int, entity):
        """`TripleIndex.hop`: the exact candidates for a hop; the walk tries this before cosine."""
        return self.index.hop(plan, hop, entity)

    def __call__(self, query: str, k: int) -> list[tuple[float, str]]:
        if not self.texts:
            return []
        if self.enc is not None:
            import numpy as np
            M = np.stack(self._rows)
            qv = self.enc.encode(query).reshape(-1).astype(np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-12)
            scores = M @ qv
            k = min(k, len(self.texts))
            idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
            idx = idx[np.argsort(-scores[idx])]
            return [(float(scores[i]), self.texts[i]) for i in idx]
        return self._overlap(query, k)

    def _overlap(self, query: str, k: int) -> list[tuple[float, str]]:
        """IDF-weighted token overlap fallback (no encoder)."""
        df: dict[str, int] = {}
        toks = [set(re.findall(r"[\w']+", t.lower())) for t in self.texts]
        for ts in toks:
            for t in ts:
                df[t] = df.get(t, 0) + 1
        n = len(self.texts)
        q = set(re.findall(r"[\w']+", query.lower()))
        if not q:
            return []
        scored = []
        for ts, text in zip(toks, self.texts):
            w = sum(math.log(1 + n / df[t]) for t in q & ts)
            norm = sum(math.log(1 + n / df.get(t, 1)) for t in q)
            scored.append((w / (norm + 1e-12), text))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(s, t) for s, t in scored[:k] if s > 0]


def route_world(worlds: dict[str, "FactStore"], query: str) -> tuple[str, float]:
    """Pick the world whose best fact scores highest for the query (MoWM v0:
    best-top-score routing; the margin gate lives in the caller's route_tau)."""
    best_name, best = next(iter(worlds)), 0.0
    for name, w in worlds.items():
        hits = w(query, 1)
        if hits and hits[0][0] > best:
            best_name, best = name, float(hits[0][0])
    return best_name, best


@runtime_checkable
class CubbyPlugin(Protocol):
    """What a mounted plugin provides. `worlds()` runs once at mount and the
    returned stores are routed alongside the default; `on_turn` (optional —
    checked with hasattr) observes each finished turn record. Plugins never
    produce replies; speech stays behind the VM's ASK and the voice rules."""

    name: str

    def worlds(self) -> dict[str, FactStore]: ...
