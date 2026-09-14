"""
Wired: WIRED (stand-in serve stack; implements `cubbyllm.reasoning.learn.Source`).

`SelfSource`: the loop's own record, as facts -- so the model can ask about its surroundings
with the same programs it uses to ask about the world. Nick, 2026-09-14: "the model needs to be
able to use the vm on its own so it can reason at all time and be concious of its surroundings
by analysing it via its own 'power vm'."

The whole thing rests on one observation: **the loop is the only subject it can witness
directly.** Every other fact in this system needs a source, a gate and a provenance, because
the loop is taking somebody's word for it. Its own history it does not take anybody's word for
-- it was there. So a self fact needs no external source, and `provenance` says `self`.

That is a privilege, and it is fenced accordingly:

1. **A self fact is never a world fact.** They live in different stores and a question reaches
   this one only by asking for it (`bind frame, WORLD, "self"`). "What do you know about Quebec
   City" and "what is Quebec City" are different questions with different answers, and letting
   the second be answered by the first is how a system starts mistaking its notes for the world.
2. **It reports, it does not judge.** `refused 3 times` is a fact. `is unreliable` is an opinion,
   and there is no relation here that carries one.
3. **It is derived, never stored.** The facts are rebuilt from the loop's history on every ask,
   so they cannot go stale and cannot be edited into something the loop did not do.
"""
from __future__ import annotations

import collections

__wiring__ = "WIRED"

from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402

# The subject that means the loop itself, rather than something it was asked about.
LOOP = "this loop"

# What a self ask may return, by kind. The `who`/`what`/`where` of the world become
# "what do I know / what did I do / where did it come from".
SELF_KINDS = {
    "what": ("asked", "answered", "refused", "last verdict", "last answer", "fact learned",
             "source", "relation asked"),
    "who": ("asked", "answered", "refused", "last verdict", "source"),
    "where": ("source", "publisher", "fact learned"),
    "when": ("asked",),
}


class SelfSource:
    """`facts(entity)` -> what the loop DID about `entity`, or about itself when `entity` is
    `this loop`. Reads an AskLoop's history; asks nothing of the network."""

    name = "self"

    def __init__(self, loop) -> None:
        self.loop = loop
        self.last: dict = {}
        self.times: dict[str, dict] = {}          # the loop's own record carries no qualifiers

    # -- the record ----------------------------------------------------------------------
    def _history(self) -> list[dict]:
        return list(getattr(self.loop, "history", None) or [])

    def about(self, entity: str) -> list[dict]:
        """The records in which `entity` was the seed, or one of the entities the loop fetched."""
        e = _normalize(entity)
        out = []
        for rec in self._history():
            names = {_normalize(rec.get("seed") or "")} | {_normalize(x) for x in (rec.get("entities") or [])}
            if e in names:
                out.append(rec)
        return out

    # -- the Source contract -------------------------------------------------------------
    def facts(self, entity: str, relations: list[str] | None = None, **_kw) -> list[Triple]:
        self.last = {"entity": entity, "n": 0, "how": None, "unresolved": False}
        self.times = {}
        hist = self._history()
        if _normalize(entity) == _normalize(LOOP):
            out = self._about_self(hist)
            self.last.update(n=len(out), how=f"the loop's own record over {len(hist)} question(s)")
            return out
        recs = self.about(entity)
        if not recs:
            # Nothing was ever asked about it. That is a true and useful answer -- and it is NOT
            # "I don't know what that is", which would be a claim about the world.
            self.last.update(unresolved=True, how="the loop has no record of that subject")
            return []
        out = self._about_entity(entity, recs)
        self.last.update(n=len(out), how=f"the loop's own record over {len(recs)} question(s)")
        return out

    def _about_entity(self, entity: str, recs: list[dict]) -> list[Triple]:
        subj = entity
        out: list[Triple] = []

        def add(rel: str, obj) -> None:
            if obj not in (None, "", []):
                out.append(Triple(obj=str(obj), rel=rel, subj=subj))

        answered = [r for r in recs if r.get("answer")]
        refused = [r for r in recs if not r.get("answer")]
        add("asked", len(recs))
        add("answered", len(answered))
        add("refused", len(refused))
        last = recs[-1]
        add("last verdict", last.get("reason") or ("answered" if last.get("answer") else "refused"))
        add("last answer", last.get("answer"))
        # what the loop LEARNED about it, and from whom -- counts and names, never a judgement
        learned = [l for r in recs for l in (r.get("learned") or []) if l.get("status") == "accepted"]
        add("fact learned", len(learned))
        for src, n in collections.Counter(l.get("source") for l in learned if l.get("source")).most_common(4):
            add("source", f"{src} ({n})")
        rels = collections.Counter(rel for r in recs for rel in (r.get("plan") or []) if rel)
        for rel, n in rels.most_common(4):
            add("relation asked", f"{rel} ({n})")
        return out

    def _about_self(self, hist: list[dict]) -> list[Triple]:
        subj = LOOP
        out: list[Triple] = []

        def add(rel: str, obj) -> None:
            if obj not in (None, "", []):
                out.append(Triple(obj=str(obj), rel=rel, subj=subj))

        answered = [r for r in hist if r.get("answer")]
        add("asked", len(hist))
        add("answered", len(answered))
        add("refused", len(hist) - len(answered))
        # WHY it refuses, commonest first. This is the single most useful thing the loop can tell
        # anyone about itself, and it is the one number a person cannot get by reading an answer.
        for reason, n in collections.Counter(r.get("reason") for r in hist
                                             if r.get("reason") and not r.get("answer")).most_common(5):
            add("refusal", f"{reason} ({n})")
        store = getattr(self.loop, "world", None)
        idx = getattr(store, "index", None)
        seen = getattr(idx, "_seen", None)
        if seen is not None:
            add("fact held", len(seen))
        times = getattr(store, "times", None)
        if times is not None:
            add("fact dated", len(times))
        prov = getattr(store, "provenance", None)
        if prov:
            for src, n in collections.Counter(prov.values()).most_common(5):
                add("provenance", f"{src} ({n})")
        calls = getattr(self.loop, "calls", None)
        if calls:
            for k, n in sorted(dict(calls).items()):
                add("call", f"{k} ({n})")
        for name, attr in (("source", "source"), ("date source", "news")):
            s = getattr(self.loop, attr, None)
            if s is not None:
                add(name, getattr(s, "name", type(s).__name__))
        return out
