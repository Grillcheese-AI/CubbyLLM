"""possibility -- the worlds layer's answer to one fact: possible, impossible, or unknown (H-F2 M4, 2026-09-29).

Wired: STANDALONE -- the interface the capsule stores, the world router and plan_verify read; nothing on the
forward path imports it until `serve.py` mounts an oracle.

WHAT IT BRINGS
--------------
Today `standin/worlds.py` routes a text to the fact store whose best hit scores highest ("MoWM v0"), and a new
fact is kept if it parses, is not a duplicate and does not contradict a stored (subject, relation). The world
model behind `bridges/world_model.py` is a fake. This module puts ONE question in front of every write and
every route -- "which world is this, and is it possible there?" -- and lets MoWM answer it
(`mowm/bridges/possibility.py`; dependency direction mowm -> cubbyllm, never the reverse) or, without MoWM,
the stores themselves (`StorePossibility` below: the same verdicts, membership rule only). The advantage over
a conventional model's "does this sound right": the answer names the world that gave it, the members that
back it, and the margin by which it won, so it can be audited, refused, or held as latent until verified.

WHICH WORLDS (one interface, three kinds)
-----------------------------------------
  fact worlds   a FactStore / CapsuleStore per domain, era or source: what someone STATED. Covers a fact by
                its best hit; supports it by members; contradicts it by a stored (subject, relation) with a
                different object (`serve.contradiction`'s rule, applied here once for every writer).
  law worlds    a MoWM expert World with axioms (or a `Knows` world): how things BEHAVE. Covers by cosine to
                its axiom CENTROID (exp_m3_open_set: 0.680 macro, 92% route precision at the margin gate);
                supports by `best_member`; predicts by its transition model, cleaned up against its members.
                Knowledge only -- a law world never returns state (the oracle line in worlds.py).
  time worlds   a fact world sliced by when the fact held (`FactStore.times`; the era/year worlds of
                exp_m3_temporal_causal): the Branch's `T^t` role. `exclude X` in a branch is removing X from
                its year-world before asking again.

WHAT A POSSIBILITY BRANCH RETURNS
---------------------------------
    Possibility(world_id, verdict, score, margin, support, predicted, latent, why)

  verdict    POSSIBLE    the routed world covers the fact and holds it (why='member') or a member clears
                         `tau_answer` (why='resembles'), and nothing it holds contradicts it
             IMPOSSIBLE  the routed world covers the fact and a member CONTRADICTS it (fact worlds: same
                         subject and relation, different object; law worlds: the injected `contradicts`)
             UNKNOWN     no world covers it (best score below `tau_match`, or an ambiguous margin), or the
                         routed world neither holds nor contradicts it -- a latent world takes it
  score      the best member similarity (support[0]) or, unmatched, the routing score
  margin     best world minus runner-up (the route-vs-spawn signal); None with one world
  support    ((member, similarity), ...) backing the verdict, best first -- symbols and scores only
  predicted  ((member, similarity), ...) what the world expects next of this fact, cleaned up against its
             members; empty for a fact world (no transition model)
  latent     the world the fact waits in until verified, or None

Only symbols and scores cross to the host: no vector leaves a world (the symbolic boundary). `possible` means
"in a world's domain and not ruled out by it", never "true": truth still comes from a verified walk.

HOW THE CAPSULE STORES READ IT
------------------------------
`CapsuleStore.add(fact)` with an oracle: IMPOSSIBLE -> refused and written to `store.ledger`; UNKNOWN ->
stored with `meta.latent = <world>` and provenance '(latent)' (the latent tier learn.py already honors: a
chain resting on it is not spoken until an attesting source agrees); POSSIBLE -> stored with `meta.support`.
`worlds.route_world` asks `oracle.route` first and falls back to best-top-score. `PossibleRelations` is the
`plan_verify.KnownRelations` that asks the oracle whether a relation is possible anywhere (the
spawn-a-latent-world path is the unknown_relations != [] branch of that same verdict).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..core.protocols import Wiring

__wiring__ = Wiring.STANDALONE

LATENT = "latent"                       # the default latent world's name


class Verdict(str, Enum):
    POSSIBLE = "possible"
    IMPOSSIBLE = "impossible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Possibility:
    """One world's answer to one fact. Symbols and scores only (see the module)."""
    world_id: str
    verdict: Verdict
    score: float
    margin: float | None = None
    support: tuple[tuple[str, float], ...] = ()
    predicted: tuple[tuple[str, float], ...] = ()
    latent: str | None = None
    why: str = ""

    @property
    def possible(self) -> bool:
        return self.verdict is Verdict.POSSIBLE

    @property
    def impossible(self) -> bool:
        return self.verdict is Verdict.IMPOSSIBLE

    @property
    def unknown(self) -> bool:
        return self.verdict is Verdict.UNKNOWN

    def as_dict(self) -> dict:
        """The trace / ledger form: plain values, rounded."""
        return {"world": self.world_id, "verdict": self.verdict.value, "score": round(float(self.score), 4),
                "margin": (None if self.margin is None else round(float(self.margin), 4)),
                "support": [[m, round(float(s), 4)] for m, s in self.support],
                "predicted": [[m, round(float(s), 4)] for m, s in self.predicted],
                "latent": self.latent, "why": self.why}


@runtime_checkable
class PossibilityOracle(Protocol):
    """What the capsule stores, the router and plan_verify need from a world model.

    `kind` says what `text` is: 'fact' (a statement to keep), 'relation' (an edge a plan asks for),
    'question' (a turn to route). `context` is the turn the text came from, for routing when the text
    alone is too short; an oracle may ignore it."""

    def possible(self, text: str, kind: str = "fact", context: str | None = None) -> Possibility: ...

    def route(self, text: str) -> tuple[str, float, float | None]:
        """(world_id, score, margin) -- the world a text belongs to, whether or not it is possible there."""
        ...

    def worlds(self) -> list[str]: ...


# ── the contradiction rule, stated once ────────────────────────────────────────

def same_subject_clash(fact: str, members) -> str | None:
    """The member with the same (subject, relation) as `fact` and a different object, else None -- the
    anti-poisoning rule `serve.contradiction` applies, factored out so every world (a store here, MoWM's
    adapter there) refuses by the same test. Every relation is treated as functional: that is the existing
    gate's choice, and the hypothesis gates measure what it costs (true facts refused)."""
    from ..reasoning.planner import normalize, parse_fact
    t = parse_fact(fact)
    if t is None:
        return None
    subj, rel, obj = normalize(t.subj), normalize(t.rel), normalize(t.obj)
    for known in members:
        k = parse_fact(known)
        if k is not None and normalize(k.subj) == subj and normalize(k.rel) == rel and normalize(k.obj) != obj:
            return known
    return None


def relation_members(rel: str, members) -> list[str]:
    """The members stating relation `rel` ('OBJ is the REL of SUBJ'), normalized -- the fact grammar's one
    appearance on the world-model side, imported by MoWM's adapter so a relation's reuse is judged by the
    same split `TripleIndex` uses."""
    from ..reasoning.planner import normalize, parse_fact
    r = normalize(rel)
    out = []
    for known in members:
        t = parse_fact(known)
        if t is not None and normalize(t.rel) == r:
            out.append(known)
    return out


def _members_about(store, fact: str):
    """The store's facts with the same subject as `fact` (its TripleIndex `about`), else every text."""
    from ..reasoning.planner import parse_fact
    index = getattr(store, "index", None)
    t = parse_fact(fact)
    if index is not None and t is not None and hasattr(index, "about"):
        return [f for f, _t in index.about(t.subj)]
    return list(getattr(store, "texts", []))


def _relations_of(store) -> dict[str, int]:
    index = getattr(store, "index", None)
    if index is not None and hasattr(index, "relations"):
        return index.relations()
    return {}


class StorePossibility:
    """The oracle without MoWM: the fact stores answer for themselves -- the same verdicts, membership rule
    only. A store covers a text by its best hit, holds a fact exactly (`fact in store`), supports it by
    members at or above `tau_answer`, and contradicts it by `same_subject_clash`. No transition model, so
    `predicted` is always empty. The world named `latent` is never routed to: what waits there is
    unverified. `tau_margin` defaults to 0 (best-top-score, today's `route_world`); MoWM's adapter uses the
    validated 0.02. `worlds` is the LIVE dict (serve's `self.worlds`): a world mounted later is routed."""

    def __init__(self, worlds: dict, tau_match: float = 0.35, tau_margin: float = 0.0,
                 tau_answer: float = 0.6, latent: str = LATENT, k: int = 5, cross_world: bool = True) -> None:
        self._worlds = worlds
        self.tau_match, self.tau_margin, self.tau_answer = float(tau_match), float(tau_margin), float(tau_answer)
        self.latent = latent
        self.k = int(k)
        # a contradiction is looked for in the routed world first, then -- `cross_world` -- in every other
        # world that holds the subject (an index lookup each, not a search): a fact whose new object pulls
        # its best member into another world would otherwise pass the world that actually holds its subject
        # (exp_f2m4, 2026-09-29: 29 of 33 escaped plants routed away from home). Time worlds are the case to
        # turn it off: the same subject legitimately has a different object in another year.
        self.cross_world = bool(cross_world)
        self.asked: list[dict] = []                           # every possible() call, for the trace

    def worlds(self) -> list[str]:
        return [n for n in self._worlds if n != self.latent]

    def _scores(self, text: str) -> list[tuple[str, float, list]]:
        out = []
        for name in self.worlds():
            w = self._worlds[name]
            try:
                hits = list(w(text, self.k))
            except Exception:                                  # a broken world is not an answer
                continue
            out.append((name, float(hits[0][0]) if hits else 0.0, hits))
        out.sort(key=lambda r: (-r[1], r[0]))
        return out

    def route(self, text: str) -> tuple[str, float, float | None]:
        ranked = self._scores(text)
        if not ranked:
            return self.latent, 0.0, None
        name, s1, _ = ranked[0]
        margin = (s1 - ranked[1][1]) if len(ranked) > 1 else None
        return name, s1, margin

    def _record(self, text: str, kind: str, p: Possibility) -> Possibility:
        self.asked.append({"text": text, "kind": kind, **p.as_dict()})
        return p

    def possible(self, text: str, kind: str = "fact", context: str | None = None) -> Possibility:
        if kind == "relation":
            return self._record(text, kind, self._relation(text))
        ranked = self._scores(text)
        if not ranked:
            return self._record(text, kind, Possibility(self.latent, Verdict.UNKNOWN, 0.0, None,
                                                        latent=self.latent, why="no_worlds"))
        name, s1, hits = ranked[0]
        margin = (s1 - ranked[1][1]) if len(ranked) > 1 else None
        if s1 < self.tau_match:
            return self._record(text, kind, Possibility(self.latent, Verdict.UNKNOWN, s1, margin,
                                                        latent=self.latent, why="below_tau_match"))
        if margin is not None and margin < self.tau_margin:
            return self._record(text, kind, Possibility(self.latent, Verdict.UNKNOWN, s1, margin,
                                                        latent=self.latent, why="ambiguous_margin"))
        store = self._worlds[name]
        if kind == "question":                                 # a turn: routed, and the hits are its support
            support = tuple((f, s) for s, f in hits if s >= self.tau_answer)
            return self._record(text, kind, Possibility(name, Verdict.POSSIBLE if support else Verdict.UNKNOWN,
                                                        s1, margin, support=support, latent=None if support else name,
                                                        why="hits" if support else "covered_unsupported"))
        key = " ".join(text.split())
        if hasattr(store, "__contains__") and key in store:
            return self._record(text, kind, Possibility(name, Verdict.POSSIBLE, 1.0, margin,
                                                        support=((key, 1.0),), why="member"))
        clash = same_subject_clash(key, _members_about(store, key))
        if clash is not None:
            cs = next((s for s, f in hits if f == clash), s1)
            return self._record(text, kind, Possibility(name, Verdict.IMPOSSIBLE, float(cs), margin,
                                                        support=((clash, float(cs)),), why="contradiction"))
        if self.cross_world:
            for other in self.worlds():
                if other == name:
                    continue
                clash = same_subject_clash(key, _members_about(self._worlds[other], key))
                if clash is not None:
                    return self._record(text, kind, Possibility(other, Verdict.IMPOSSIBLE, s1, margin,
                                                                support=((clash, s1),), why="contradiction_elsewhere"))
        support = tuple((f, float(s)) for s, f in hits if s >= self.tau_answer)
        if support:
            return self._record(text, kind, Possibility(name, Verdict.POSSIBLE, support[0][1], margin,
                                                        support=support, why="resembles"))
        return self._record(text, kind, Possibility(name, Verdict.UNKNOWN, s1, margin,
                                                    latent=self.latent, why="covered_unsupported"))

    def _relation(self, rel: str) -> Possibility:
        """A relation is possible where a store REUSES it (>= 2 facts: `StoreRelations.reused`'s guard --
        a one-off relation is usually an entity fragment mis-split at an ' of '); a paraphrase of a reused
        relation (`planner.relation_matches`) is possible with the matched name as its support."""
        from ..reasoning.planner import normalize, relation_matches
        r = normalize(rel)
        best: tuple[str, str, int] | None = None                # (world, relation, n)
        for name in self.worlds():
            n = _relations_of(self._worlds[name]).get(r, 0)
            if n >= 2 and (best is None or n > best[2]):
                best = (name, r, n)
        if best is not None:
            return Possibility(best[0], Verdict.POSSIBLE, 1.0, None, support=((best[1], 1.0),), why="member")
        for name in self.worlds():
            for known, n in sorted(_relations_of(self._worlds[name]).items()):
                if n >= 2 and relation_matches(r, known):
                    return Possibility(name, Verdict.POSSIBLE, 1.0, None, support=((known, 1.0),), why="paraphrase")
        return Possibility(self.latent, Verdict.UNKNOWN, 0.0, None, latent=self.latent, why="unknown_relation")


class PossibleRelations:
    """`plan_verify.KnownRelations` answered by an oracle: `rel in known` is exact (why='member'), `match`
    is the paraphrase tier (why='paraphrase') and returns the relation it matched -- two tiers, as the
    protocol requires, so an approximate match never wears an exact hit's confidence. A relation the oracle
    finds UNKNOWN lands in `verify_plan`'s unknown_relations: the spawn-a-latent-world branch."""

    def __init__(self, oracle: PossibilityOracle) -> None:
        self.oracle = oracle

    def __contains__(self, rel: str) -> bool:
        p = self.oracle.possible(rel, kind="relation")
        return p.possible and p.why == "member"

    def match(self, rel: str) -> str | None:
        p = self.oracle.possible(rel, kind="relation")
        if p.possible and p.support:
            return p.support[0][0]
        return None
