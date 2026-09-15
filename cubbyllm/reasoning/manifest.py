"""manifest -- the per-request capability manifest. WO-2.1.

Wired: STANDALONE until a serving path constructs one. It is a drop-in for the
`known=` seam that `exp_r25` proved is clean, so nothing is rewired to adopt it;
`__wiring__` says STANDALONE rather than WIRED on purpose, because WO-0.6 found
three modules claiming the latter with no caller and the new CI guard now checks.

WHY THIS EXISTS
---------------
`exp_r26` relabelled every relation to an opaque token and both emitter arms died
on the rate clause -- 0.247 and 0.262 of their labelled score. The string-argument
form does not carry the generalization: an emitter asked for `damuzo` either
copies the token out of the question or falls back on a relation it memorized,
and `StoreRelations.match` will happily fuzzy-match the fallback onto something
real. WO-0.3 measured the same thing statically: 95-98% of role identifiers were
per-relation, so a relation absent from training has nothing to bind to.

The fix is not a better emitter. It is to stop asking the emitter to *know* the
relation at all:

    The host supplies, per request, the admissible relation strings plus
    direction and arity. The string is evaluated EXACTLY ONCE, as a key into
    this index. A miss is a hard error, never a nearest-neighbour fallback.

That is "evaluated, not parsed" made mechanical. `StoreRelations` has two tiers
on purpose -- exact membership and a Jaccard>=0.6 paraphrase tier -- and the
second tier is precisely what lets a memorized relation slip through wearing a
real one's confidence. A manifest has one tier.

WHAT IT BUYS, BEYOND GENERALIZATION
-----------------------------------
The same mechanism is what layered worlds need. A counterfactual world (a fact
retracted, the closure recomputed) and an invented world (a fiction's own facts)
each have a DIFFERENT admissible relation set, and the manifest is how the host
tells the emitter which world it is planning in. One substrate, both goals.

WHAT IT CANNOT DO
-----------------
WO-2.5's own self-indictment applies and is worth stating on the class rather
than in a doc nobody opens: **once the host supplies the manifest, a pure copier
passes an unseen-relation test.** Correctness migrates silently into manifest
construction. `with_decoys` exists so that migration is measurable -- inject k
plausible relations the entity does not have, and an emitter that is selecting
rather than copying should still beat the 1/(k+1) chance baseline.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .planner import normalize, parse_fact, reused_relations

__wiring__ = Wiring.STANDALONE


@dataclass(frozen=True)
class Entry:
    """One admissible relation, with the metadata WO-2.1 names."""

    relation: str
    n_facts: int = 0
    direction: str = "forward"          # forward: OBJ is the REL of SUBJ
    arity: str = "multi"                # 'functional' (one value) | 'multi'

    def line(self) -> str:
        a = "one value" if self.arity == "functional" else "may have several"
        return f"- {self.relation} ({a})"


class Manifest:
    """The admissible relations for ONE request. Exact lookup, one tier.

    Satisfies `plan_verify.KnownRelations`: `rel in manifest` is exact
    (normalized) membership, and `match` returns the relation itself on a hit or
    **None** on a miss. There is deliberately no fuzzy tier -- `_fuzzy_match` is
    what lets a memorized relation wear an exact hit's confidence, and removing
    it is the entire proposal.
    """

    def __init__(self, entries) -> None:
        self._by_key: dict[str, Entry] = {}
        for e in entries:
            self._by_key[normalize(e.relation)] = e

    # -- the KnownRelations protocol ---------------------------------------
    def __contains__(self, rel: str) -> bool:
        return normalize(rel) in self._by_key

    def match(self, rel: str, *, reused_only: bool = False) -> str | None:
        """Evaluated, not parsed. One dict lookup; a miss is a miss."""
        e = self._by_key.get(normalize(rel))
        return e.relation if e else None

    # -- the rest of the surface `learn_and_answer` touches ------------------
    def declare(self, rel: str) -> None:
        """A manifest is fixed for the request it was built for. A source that
        wants to widen it must build a new one -- silently growing the admissible
        set mid-walk is how 'the host supplies' turns back into 'the model
        decides'."""

    def add(self, fact: str) -> None:
        """Facts are the store's business, not the manifest's."""

    def reused(self):
        return self

    def words(self) -> frozenset[str]:
        return frozenset(w for k in self._by_key for w in k.split())

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key)

    def __repr__(self) -> str:
        return f"Manifest({len(self._by_key)} relations)"

    # -- what the host shows the emitter ------------------------------------
    def prompt_block(self, header: str | None = None) -> str:
        """The admissible set, rendered for the emitter's system prompt.

        This is the interim form. WO-2.2's generated grammar makes an unknown
        relation *undecodable* rather than merely discouraged, which is strictly
        better: a prompt asks, a grammar enforces. Until then this measures
        whether the information is enough, separately from whether the
        constraint is enforced.
        """
        head = header or ("The ONLY relations available for this question are listed below. "
                          "Use one of them exactly as written. If none of them answers the "
                          "question, emit no plan.")
        body = "\n".join(e.line() for e in sorted(self._by_key.values(),
                                                  key=lambda e: e.relation))
        return f"{head}\n{body}"

    # -- construction --------------------------------------------------------
    @staticmethod
    def index_store(texts) -> dict[str, dict[str, int]]:
        """subject -> {relation: n_facts}, in one pass over the store.

        Built once and reused: a scan per question over a 563k-fact store would
        make the manifest cost more than the walk it constrains.

        Uses `parse_fact` with the two-pass `reused_relations` split, exactly as
        `StoreRelations` does -- NOT a regex. The first cut here used
        `^(.+?) is the (.+?) of (.+)$`, which is non-greedy and therefore splits
        `1890-12-08 is the date of birth of Bohuslav Martinu` at the FIRST ' of ',
        yielding relation `date` and subject `birth of Bohuslav Martinu`. The
        ' of '-split ambiguity is a known hazard in this codebase (it is what put
        entity names inside role identifiers in WO-1.3's audit), and the project
        already has the parser that handles it. Reimplementing it was the mistake.
        """
        texts = list(texts)
        known = reused_relations(texts)
        out: dict[str, dict[str, int]] = {}
        for t in texts:
            tr = parse_fact(t, known=known)
            if tr is None:
                continue
            d = out.setdefault(normalize(tr.subj), {})
            rel = tr.rel.strip()
            d[rel] = d.get(rel, 0) + 1
        return out

    @classmethod
    def for_entity(cls, entity: str, index: dict[str, dict[str, int]]) -> "Manifest":
        """Everything the store can actually answer about this entity."""
        rels = index.get(normalize(entity), {})
        return cls(Entry(relation=r,
                         n_facts=n,
                         arity="functional" if n == 1 else "multi")
                   for r, n in rels.items())

    def with_decoys(self, pool, k: int, rng: random.Random) -> "Manifest":
        """k plausible relations this entity does NOT have, added to the set.

        The control for the copier problem. A manifest listing exactly one
        relation is answerable by copying it; a manifest listing the right one
        among k+1 plausible ones is not. Selection accuracy against the
        1/(k+1) chance baseline is the measurement WO-2.5 asks for and could
        not run before a manifest existed.
        """
        have = set(self._by_key)
        cand = [r for r in pool if normalize(r) not in have]
        rng.shuffle(cand)
        extra = [Entry(relation=r, n_facts=0, arity="multi") for r in cand[:k]]
        return Manifest(list(self._by_key.values()) + extra)
