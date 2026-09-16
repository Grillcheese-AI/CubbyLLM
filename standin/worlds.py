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
from array import array
from typing import Protocol, runtime_checkable

__wiring__ = "WIRED"


class FactStore:
    """A named, growable retrieval world: (query, k) -> [(score, fact)].

    With `enc` (a FastWordEncoder-shaped object): unit-cosine over encoded
    rows, mirroring exp_m3_cot_pipeline.make_retriever. Without: IDF-weighted
    token overlap. Callable, so anything that took a retriever takes a store.
    """

    _TOKEN = re.compile(r"[\w']+")
    BIG, COMMON = 10_000, 0.2      # on a store past BIG facts, a token in more than COMMON of them is not traversed by the fallback

    def __init__(self, texts: list[str] | None = None, enc=None, name: str = "facts") -> None:
        from cubbyllm.reasoning import TripleIndex
        self.name = name
        self.enc = enc
        self.texts: list[str] = []
        self._rows: list = []                            # np row vectors when enc is set
        self._seen: set[str] = set()
        self._post: dict[str, array] = {}                # token -> fact ids: the lexical fallback's inverted index
        self.times: dict[str, dict] = {}                 # fact key -> {start, end, point}: WHEN the fact was true, when the
                                                         # source said so (2026-09-14; the fact text never carries the time)
        self.provenance: dict[str, str] = {}             # fact key -> the source(s) that stated it (learn.py fills it; a
                                                         # latent source's fact stays '<name> (latent)' until another agrees)
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
        i = len(self.texts) - 1
        for tok in set(self._TOKEN.findall(key.lower())):
            post = self._post.get(tok)
            if post is None:
                post = self._post[tok] = array("I")
            post.append(i)
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
        """IDF-weighted token overlap fallback (no encoder), through the inverted index: the same score as the
        original per-fact scan (sum of IDF over shared tokens / the query's IDF mass), O(postings) per query
        instead of O(store). Past BIG facts, near-universal tokens ('is', 'the', 'of' in a template store) are
        not traversed — they carry no discrimination and would cost the whole store per query (the wikikg
        world, 2026-09-04)."""
        n = len(self.texts)
        q = set(self._TOKEN.findall(query.lower()))
        if not q:
            return []
        norm = sum(math.log(1 + n / (len(self._post[t]) if t in self._post else 1)) for t in q)
        acc: dict[int, float] = {}
        for t in q:
            post = self._post.get(t)
            if post is None or (n > self.BIG and len(post) > self.COMMON * n):
                continue
            w = math.log(1 + n / len(post))
            for i in post:
                acc[i] = acc.get(i, 0.0) + w
        scored = sorted(((w / (norm + 1e-12), self.texts[i]) for i, w in acc.items()), key=lambda s: (-s[0], s[1]))
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


# ── the worlds he can ASK, and what keeps that from being an oracle ────────
#
# Nick, 2026-09-15:
#
#   "the science world is having gravity inside, cubby dont know it tries stuff
#    then all of a sudden oh... let me ask the world: 'how can I know when
#    something is about to fall on me?' then the science world sends the gravity
#    + attraction laws so it understands ... then cubby stores it in long term
#    memory so it knows that part and dont have to ask already. In reality its
#    cubby building its own world via interations via other worlds outside its
#    own."
#
# A `FactStore` above is a world he RETRIEVES from. A `Knows` world is one he
# can put a QUESTION to and get back facts phrased as laws. The difference from
# the oracle WO-2.10 spent a day removing is not the plumbing, it is what may
# travel down it:
#
#   knowledge  — how falling things behave.      ALLOWED. He could in principle
#                have found it out himself, slowly, by dropping things. It can
#                be wrong, and later evidence can refute it. It tells him what
#                to PERCEIVE.
#   state      — where the ghost is standing now. NEVER. He could not have
#                found it out; it arrives unearned, cannot be checked, and
#                makes perceiving unnecessary.
#
# That line is not enforceable by a type, so it is enforced by the worlds
# themselves (a world is written to answer in laws) and made auditable here:
# every ask is recorded in `Worlds.asked`, which the percept tripwire reads.


@runtime_checkable
class Knows(Protocol):
    """A world that can be asked a question, not just searched.

    `covers` is how it says the question is in its domain at all — returning 0
    is a real and common answer, because a question NO world covers has to stay
    unanswered. `answer` returns the facts, already in his fact language, or an
    empty list."""

    name: str
    domain: str

    def covers(self, question: str) -> float: ...

    def answer(self, question: str) -> list[str]: ...


class Worlds:
    """The worlds outside his own, and the routing that picks one.

    Deliberately thin. It does not hold knowledge, it does not decide what is
    true, and it never writes to anybody's map — the asker learns the answer
    through its own gate, the same one every percept passes. All this does is
    carry a question to whoever's domain it is, and remember that it did."""

    TAU = 0.35                                   # below this, no world claims the question

    def __init__(self, trace=None, tau: float | None = None) -> None:
        self.by_name: dict[str, Knows] = {}
        self.tau = self.TAU if tau is None else float(tau)
        self.trace = trace or (lambda kind, **d: None)
        self.asked: list[dict] = []              # every question, where it went, what came back

    def __len__(self) -> int:
        return len(self.by_name)

    def mount(self, world: Knows) -> None:
        if not isinstance(world, Knows):
            raise TypeError(f"{world!r} is not askable: it needs name, domain, covers(), answer()")
        self.by_name[world.name] = world

    def route(self, question: str) -> tuple[Knows | None, float]:
        """The world whose domain this is, or None. Ties break by name so the
        routing is reproducible."""
        best, score = None, 0.0
        for name in sorted(self.by_name):
            w = self.by_name[name]
            try:
                s = float(w.covers(question))
            except Exception:                    # a broken world is not an answer
                continue
            if s > score:
                best, score = w, s
        return (best, score) if score >= self.tau else (None, score)

    def ask(self, question: str) -> dict:
        """Put the question to whoever knows. Returns
        {world, score, facts} — facts empty when nobody covers it, which is
        a legitimate outcome and the reason he can still be ignorant."""
        w, score = self.route(question)
        facts: list[str] = []
        if w is not None:
            try:
                facts = [f for f in (w.answer(question) or []) if f and f.strip()]
            except Exception as e:
                self.trace("ask_error", question=question, world=w.name, error=str(e)[:160])
                facts = []
        rec = {"question": question, "world": (w.name if w else None),
               "score": round(score, 3), "facts": facts}
        self.asked.append(rec)
        self.trace("ask", question=question, world=rec["world"], score=rec["score"],
                   got=len(facts), facts=facts[:4] or None)
        return rec

    def domains(self) -> dict[str, str]:
        return {n: w.domain for n, w in sorted(self.by_name.items())}
