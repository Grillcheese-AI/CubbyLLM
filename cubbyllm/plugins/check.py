"""standin/plugins_check.py -- the second check: a plugin the router woke loads only when it claims the question
(order 0028, review 2).

The router scores a question against what each plugin says it is for; a high score is a guess. A woken plugin acts --
a source fetches, a world answers from its pack, an adapter words the reply -- so it is asked first:

- a ``source`` or ``world``: its own ``covers(question)`` (a facts world without one: its best fact's score, at the
  worlds' ``TAU``), on the object built but not yet mounted -- building fetches nothing; mounting and asking do;
- an ``adapter`` (the ``talk`` role): the thalamus (``CubbyBrain.needs_facts``) must say the turn needs no facts --
  a factual answer is the VM's to give, never a talk adapter's wording;
- a ``tool`` checks its own arguments when called; the router's score stands.

A plugin that declines is not counted as woken and loads nothing; why is kept (``Loader.declined``) for the audit
view and the board's "which plugin would wake?".

Wired: WIRED (``plugins_load.Loader.wake``).
"""
from __future__ import annotations

__wiring__ = "WIRED"


def cover(obj, question: str) -> float:
    if hasattr(obj, "covers"):
        return float(obj.covers(question) or 0.0)
    if callable(obj):                                     # a FactStore: (question, k) -> [(score, fact)]
        from ..worlds.base import Worlds
        got = obj(question, 1)
        return float(got[0][0]) if got and got[0][0] >= Worlds.TAU else 0.0
    return 0.0


def probe(loader, p, kind: str):
    """The plugin's source or world built, not mounted; None (and why, in ``loader.refused``) when it may not run."""
    from . import guard
    why = loader.cleared(p)
    if why != "ok":
        loader.refused[p.name] = why
        return None
    ctx = guard.Ctx(p, loader.data_root, loader.log, loader.fetch)
    try:
        return loader.make_source(p, ctx) if kind == "source" else loader.make_world(p, ctx)
    except Exception as e:                                # noqa: BLE001 -- a plugin that fails stays out, with why
        loader.refused[p.name] = f"{type(e).__name__}: {str(e)[:200]}"
        return None


def claims(loader, name: str, question: str):
    """``(claims it, why, objects built)``: the objects go to ``Loader.load`` so nothing is built twice."""
    p = loader.index[name]
    if "adapter" in p.kinds:
        if loader.thalamus is None:
            return False, "no thalamus to say whether the turn needs facts", None
        facts, why = loader.thalamus(question)
        if facts:
            return False, f"the turn needs facts ({why}): not an adapter's to word", None
    made = {}
    for kind in ("source", "world"):
        if kind not in p.kinds:
            continue
        obj = ((loader.loaded.get(name) or {}).get("kinds") or {}).get(kind)
        if obj is None:
            obj = probe(loader, p, kind)
            if obj is None:
                return False, loader.refused.get(name, "could not be built"), None
            made[kind] = obj
        score = cover(obj, question)
        if score <= 0:
            return False, f"its {kind} does not claim the question (covers {score:.2f})", None
    return True, "claims it", made
