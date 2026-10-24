"""cubbyllm/reasoning/ask_namesake_answer.py -- how an answer reads when the question's relation decided a shared name
(order 0037; the world's side is ``worlds.hdc_relation``). Behind ``CB_NAMESAKE_RELATION=1``, as the hop is.

- The walk reads the question's **cue words** (``cue_words``: its content words, less the name's and the relation's):
  "Nobel" rules out a namesake whose kind never goes with it.
- **One fits:** the answer, then one line naming at most two other namesakes by their kind, with an offer (Nick: "as
  Gemini does"): "Paris. If you meant something else, there is also a constituency of the European Parliament named
  France. Want to know about it?" An item with nothing known of its kind is not named. A value the question's own
  description of its answer rules out is no answer ("What **Nobel** Prize did Marie Curie win?": the one award she holds
  here is another), so the reply says it does not know (``class_words``).
- **Several fit (two of a kind):** each is walked apart (pinned, so every value stays tied to its own item: order
  0033's rule) and answered, labelled by one of its own facts that tells them apart, in their ids' order, never by
  fame: "Steve Mills (occupation: baseball player): 1950; Steve Mills (occupation: politician): 1962." More than three,
  or no fact that tells them apart: the loop asks which, as before.

Wired: WIRED (``ask_namesakes.seed_for`` and ``ask_namesakes.after_walk``).
"""
from __future__ import annotations

import os
import re as _re

from .ask_words import say

__wiring__ = "WIRED"

EACH_MAX = 3                                             # two of a kind answered each; more: asked which
TELLS_APART = ("occupation", "country of citizenship", "country", "located in the administrative territorial entity",
               "country of origin", "genre", "instance of")
QUESTION_WORDS = frozenset("""what which who whom whose where when why how did does do is are was were has have had can
could will would tell me please name give list the a an of in on at to for from by with about""".split())
_DESC = _re.compile(r"^\s*(?:what|which)\s+(?P<desc>.+?)\s+(?:did|does|do|is|was|were|are|has|have|had|can|could|will|would)\b",
                    _re.I)


def on() -> bool:
    return os.environ.get("CB_NAMESAKE_RELATION") == "1"


def cue_words(question: str, seed: str, rels) -> list[str]:
    """The question's own words that may narrow a shared name: its content words, less the name's, the relation's
    and the question's own ("what", "did")."""
    from cubbyllm.worlds.hdc_cues import words
    drop = set(words(seed)) | {w for r in rels or () for w in words(r or "")} | QUESTION_WORDS
    return list(dict.fromkeys(w for w in words(question) if w not in drop))


def class_words(question: str, seed: str) -> list[str]:
    """The capitalised words of the question's own description of its answer ("What Nobel Prize did ..." -> nobel,
    prize), less the name's: a value without them is not what was asked."""
    m = _DESC.match(question or "")
    if not m:
        return []
    name = {w.lower() for w in (seed or "").split()}
    return [w.lower() for w in m.group("desc").split() if w[:1].isupper() and w.lower() not in name]


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def also_line(named: list[dict]) -> str:
    """The line after a one-item answer: the other namesakes, by their kind, and an offer."""
    if not named:
        return ""
    parts = [f"{_a(o['kind'])} named {o['label']}" for o in named]
    which = "it" if len(named) == 1 else "one of them"
    return f"If you meant something else, there is also {' and '.join(parts)}. Want to know about {which}?"


def label_for(items: list[str], by: dict) -> dict | None:
    """{item: "label (relation: value)"} by the first relation each item holds once, with a value of its own that no
    other item shares; None when no fact tells them all apart."""
    for rel in TELLS_APART + tuple(sorted({t.rel for q in items for _f, t in by[q]} - set(TELLS_APART))):
        vals = {q: [t.obj for _f, t in by[q] if t.rel == rel] for q in items}
        if all(len(v) >= 1 for v in vals.values()) and len({v[0] for v in vals.values()}) == len(items):
            return {q: f"{by[q][0][1].subj} ({say(rel)}: {vals[q][0]})" for q in items}
    return None


def one_fits(loop, rec: dict, seed: str, d: dict, by: dict) -> None:
    """The record of a walk on the one item that fit: its answer checked against the question's own description of
    it, then the other namesakes named."""
    from cubbyllm.worlds.hdc_relation import others
    item = d["fit"][0]
    named = others(d, by, getattr(loop.world, "_kinds", None) or _kinds_of(loop.world))
    rec["namesakes"] = {"rule": "one fits", "item": item, "others": named, "kinds": d["kinds"], "cues": d["cues"]}
    if not rec.get("verified"):
        return                                           # the item holds no such fact: the reply says it does not know
    want = class_words(rec.get("question") or "", seed)
    if want and not all(w in str(rec.get("answer") or "").lower() for w in want):
        rec.update(answer=None, verified=False, reason="not_what_was_asked",
                   refused={"answer": rec.get("answer"), "why": f"the question asks for a {' '.join(want)}"})
        return
    line = also_line(named)
    if line:
        rec["namesakes"]["also"] = line                  # said once the loop's checks have shaped the reply (`also`)


def also(rec: dict) -> None:
    """The line naming the other namesakes, after the answer as the loop's checks left it ("As of 2019, Paris."), and
    only while it is still said."""
    line = (rec.get("namesakes") or {}).get("also")
    if line and rec.get("verified"):
        rec["reply"] = f"{str(rec.get('reply') or rec['answer']).rstrip('.')}. {line}"


def several_fit(loop, rec: dict, seed: str, d: dict, by: dict, learn) -> bool:
    """Each item that fit walked apart and answered, labelled; False (the loop asks which) when there are more than
    EACH_MAX, no walk to make, or no fact that tells them apart."""
    items = list(d["fit"])
    labels = label_for(items, by) if len(items) <= EACH_MAX and learn is not None else None
    if not labels:
        return False
    answers = []
    for q in items:
        with loop._lock, loop.world.pinned(seed, by[q][0][1].subj, q):
            lr = learn()
        ok = bool(lr.result.verified)
        one = {"answer": lr.result.answer if ok else None, "verified": ok, "reason": None if ok else lr.result.reason}
        if hasattr(loop, "_checked"):                    # the checks any walk passes: an ended value is not said
            loop._checked(rec.get("question") or "", lr, one)
        ok = bool(one["verified"])
        answers.append({"item": q, "label": labels[q], "answer": one["answer"], "verified": ok, "reason": one["reason"],
                        "said": str(one.get("reply") or one["answer"]).rstrip(".") if ok else None,
                        "trace": [h.fact for h in lr.result.trace if h.fact] if ok else []})
    if not any(a["verified"] for a in answers):
        return False
    said = "; ".join(f"{a['label']}: {a['said'] if a['verified'] else 'I do not know'}" for a in answers)
    rec.update(answer=said, reply=f"{said}.", verified=True, reason=None, answers=answers,
               trace=[f for a in answers for f in a["trace"]],
               namesakes={"rule": "several fit", "items": items, "kinds": d["kinds"], "cues": d["cues"]})
    rec.pop("clarify", None), rec.pop("candidates", None), rec.pop("refused", None)
    return True


def _kinds_of(world):
    return next((o._kinds for o in getattr(world, "others", []) if getattr(o, "_kinds", None) is not None), None)


__all__ = ["also", "also_line", "class_words", "cue_words", "label_for", "on", "one_fits", "several_fit"]
