"""cubbyllm/reasoning/answer_values.py -- a question that names the class of its answer narrows a relation with many
values to the values of that class (order 0055, the Marie Curie test of change 3 in Cowork's audit addendum, approved
by Nick 2026-11-10), behind ``CB_ANSWER_CLASS=1`` (off by default; off, the walk is unchanged).

"What Nobel Prize did Marie Curie win?": her "award received" holds 16 values in the full store. The walk refuses a
last hop with several values (a set of answers is not a path to choose: `ask_claims.branch_clarify`), and past its 12
branches it cannot even explore them. The question names its answer's class ("Nobel Prize":
`ask_namesake_answer.class_words`, the capitalised words of "What/Which <class> did/is ..."); only the values whose
own label holds every word of it are what was asked. That is the question doing its job, as a year it names is
(`pipeline._covering`), never a pick:
- **one value left:** the hop takes it;
- **two to SAY_MAX left:** each is walked and certified as its own branch (`pipeline._explore`), and when every branch
  is certified with a value of its own, they are said together, in their labels' order, never by fame: "Nobel Prize
  in Chemistry and Nobel Prize in Physics";
- **none left:** the walk goes on as before (the value's class check, `ask_class`, and `ask_namesake_answer.one_fits`
  refuse a value not of the class); **more than SAY_MAX:** the refusal, naming the narrowed values.
Only the last hop is narrowed: the class describes the answer, not a step on the way to it. Only where the walk would
refuse: after a year the question names and a date the source gives (`pipeline._latest_dated`), so on, it never
changes an answer the walk already gives. Only a capitalised class ("Nobel Prize", "Grammy Award"): "which city" names
none, so "Quezon City" is never kept over "Manila" for holding the word. A plural ("Which Nobel Prizes ...") reads as
its singular. A value is matched by its label's words, so "Ig Nobel Prize" would read as a Nobel
Prize (none is in the store for a laureate today; the value's class check is the stricter one, online).

Wired: WIRED (``pipeline._walk`` and ``pipeline.answer``).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

SAY_MAX = 3                                              # values of the class said together; more: refused, named
_WORD = re.compile(r"[\w'-]+")


def on() -> bool:
    return os.environ.get("CB_ANSWER_CLASS") == "1"


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def wanted(question: str, tail: str) -> tuple[str, ...]:
    """The words of the class the question names for its answer, less the subject's ((): none named, or the flag
    off). `tail`: the plan's "<relation> of <subject>"."""
    if not on():
        return ()
    from .ask_namesake_answer import class_words
    seed = tail.rsplit(" of ", 1)[-1] if " of " in (tail or "") else ""
    return tuple(dict.fromkeys(_stem(w) for w in class_words(question, seed)))


def fits(value: str, want) -> bool:
    words = {_stem(w.lower()) for w in _WORD.findall(value or "")}
    return bool(want) and all(w in words for w in want)


def narrow(cands: list, want) -> list:
    """The (fact, triple) candidates whose value holds every word of the class asked."""
    return [(f, t) for f, t in cands if fits(t.obj, want)]


def several(branches: list[dict], overflow: bool) -> list[dict] | None:
    """The branches to say together: every one finished and certified, each with a value of its own, 2..SAY_MAX of
    them; None otherwise (the refusal stands)."""
    if overflow or not branches or any(b["status"] != "verified" for b in branches):
        return None
    by = {}
    for b in branches:
        by.setdefault(" ".join(str(b["answer"] or "").lower().split()), b)
    if not 2 <= len(by) <= SAY_MAX:
        return None
    return [by[k] for k in sorted(by)]


def said(live: list[dict]) -> tuple[str, list]:
    """(the values said together, the trace: the first branch's hops, then each other branch's last hop)."""
    vals = [b["answer"] for b in live]
    answer = vals[0] if len(vals) == 1 else ", ".join(vals[:-1]) + " and " + vals[-1]
    trace = list(live[0]["_trace"]) + [b["_trace"][-1] for b in live[1:]]
    return answer, trace


__all__ = ["SAY_MAX", "fits", "narrow", "on", "said", "several", "wanted"]
