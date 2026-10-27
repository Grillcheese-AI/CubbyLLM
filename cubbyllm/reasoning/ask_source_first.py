"""cubbyllm/reasoning/ask_source_first.py -- the live source's value over a snapshot's (Nick, 2026-10-27; order 0037,
item 3).

A relation whose value changes (a capital, a head of state), held by a read-only snapshot world (the hdc world, ~2019),
is said "As of 2019, Paris" (``ask_memory.AskMemory._checked``). The source may know today's. Before the snapshot's
value is said, the walk is made once more with that world's rows on the relation set aside (``worlds.aside``): it
stalls there and asks the source as any walk does (the books first, the web after). A value it verifies is the answer,
said as any source's value is (the tense check and the time rule apply to it in their turn); none (offline, not
cached, no such fact, refused) and the snapshot's value stands, as of its year. The namesake rule still decides which
item is meant (``worlds.hdc_relation``); this decides only whose value is said. The record keeps what the snapshot
said (``over_snapshot``) or why it stood (``snapshot_stood``).

On by default (Nick's call); ``CB_SOURCE_OVER_SNAPSHOT=0`` turns it off (the snapshot's value said as of its year, as
before 2026-10-27).

Wired: WIRED (``ask_memory.AskMemory._checked``, handed the walk by ``ask.AskLoop._ask`` and
``ask_namesake_answer.several_fit``).
"""
from __future__ import annotations

import os

__wiring__ = "WIRED"


def on() -> bool:
    return os.environ.get("CB_SOURCE_OVER_SNAPSHOT") != "0"


def source_wins(loop, question: str, lr, rec: dict, again, as_of: str) -> bool:
    """The walk once more (`again`: the same walk, under the same pin or cue) with the snapshot's rows on the answer's
    relation aside. True when the source's value replaced the snapshot's in `rec` and went through the loop's checks
    in its turn; False, `rec` untouched but for `snapshot_stood`, when the source gave none."""
    from cubbyllm.worlds.aside import aside
    h = lr.result.trace[-1] if lr.result.trace else None
    holder = (loop.world.held_by(h.fact) if h is not None and h.triple is not None and hasattr(loop.world, "held_by")
              else None)
    if not on() or again is None or holder is None or getattr(loop.world, "_aside", None) is None:
        return False
    with aside(loop.world, holder, h.triple.rel):
        lr2 = again()
    h2 = lr2.result.trace[-1] if lr2.result.verified and lr2.result.trace else None
    if h2 is None or loop.world.held_by(h2.fact) is holder:
        rec["snapshot_stood"] = {"as_of": as_of, "source": lr2.result.reason or "the snapshot's own value"}
        return False
    from .ask_claims import _learned
    rec["over_snapshot"] = {"answer": lr.result.answer, "as_of": as_of}
    rec.update(answer=lr2.result.answer, verified=True, reason=None, refused=None,
               trace=[x.fact for x in lr2.result.trace if x.fact],
               learned=list(rec.get("learned") or []) + [_learned(p) for p in lr2.learned],
               entities=list(dict.fromkeys(list(rec.get("entities") or []) + list(lr2.entities))))
    rec.pop("tier", None)
    rec.pop("reply", None)
    loop._checked(question, lr2, rec)                   # the veto, the tense, the time rule, the hedge: once more
    return True


__all__ = ["on", "source_wins"]
