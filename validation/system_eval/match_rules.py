"""validation/system_eval/match_rules.py -- order 0075: the rule behind each "right", and a strict matcher behind a flag.

``rule(answer, golds, bench)`` names the rule that matched, or None: exactly where ``exp_everyday_webq.matches`` (the
lenient matcher 0054 graded with) matches, so the verdicts do not change (step 1). Rules, strongest first:
  ``equal``           the answer and a gold are the same after normalization (``exp_everyday_webq._norm``);
  ``answer in gold``  the answer, over 3 characters, is inside a gold;
  ``gold in answer``  a gold, over 3 characters, is inside the answer;
  ``year``            the first years of the two are the same;
  CLUTRR: ``relation`` (the relation word, normalized, equal) and ``relation word`` (a whole word of the answer).

``rule(..., strict=True)`` (step 4) takes less, never more:
  - ``equal`` as before;
  - ``gold in answer`` only on word boundaries, and only when the answer names one candidate: no list (a comma, a
    semicolon, a slash, " or ", " and "), no second year, no negation ("not", "no", "never", "n't");
  - ``answer in gold`` is not taken: "Paris" against "Paris, Texas" is a different place as often as the same;
  - ``year`` only when the gold is a date (a year, or a day and a month with a year) and the answer holds exactly one
    year;
  - CLUTRR as before (a relation word has no partial).
What the strict matcher does not take goes to the read, as unmatched answers do.

``non_answer(answer)``: empty, "I don't know", "The facts don't say", "not attempted", "N/A", "unknown", "none
given": under the strict matcher it is refused or declined for every arm, never read as an answer.

Wired: STANDALONE (order 0054's harness, `grade.py`).
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

RULES = ("equal", "answer in gold", "gold in answer", "year", "relation", "relation word")
YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")
LIST = re.compile(r"[,;/]|\b(or|and|ou|et)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|none|neither|nor|pas|jamais)\b|n't\b", re.I)
MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec").split()
NON_ANSWERS = ("i don't know", "i do not know", "the facts don't say", "the facts do not say", "not attempted",
               "not_attempted", "n/a", "unknown", "none given", "no answer", "je ne sais pas")


def _norm(s: str) -> str:
    from exp_everyday_webq import _norm as n
    return n(s)


def non_answer(answer: str | None) -> bool:
    a = " ".join(str(answer or "").lower().replace("’", "'").split()).strip(" .!?")
    return not a or a.startswith(NON_ANSWERS) or a in ("na", "none", "-", "?")


def _date_gold(g: str) -> bool:
    words = _norm(g).split()
    return bool(words) and len(YEAR.findall(g)) == 1 and all(w.isdigit() or w in MONTHS for w in words)


def _one_candidate(raw: str) -> bool:
    return not LIST.search(raw) and not NEGATION.search(raw) and len(YEAR.findall(raw)) <= 1


def rule(answer: str | None, golds: list[str], bench: str, strict: bool = False) -> str | None:
    if bench == "clutrr":
        from cubbyllm.reasoning.planner import normalize
        a = normalize(answer or "")
        if not a:
            return None
        if any(a == normalize(g) for g in golds):
            return "relation"
        return "relation word" if any(f" {normalize(g)} " in f" {a} " for g in golds) else None
    raw = str(answer or "")
    a = _norm(raw)
    if not a:
        return None
    found = []
    for g in golds:
        n = _norm(g)
        if not n:
            continue
        if a == n:
            found.append("equal")
            continue
        if strict:
            if len(n) > 3 and f" {n} " in f" {a} " and _one_candidate(raw):
                found.append("gold in answer")
            elif _date_gold(g) and len(YEAR.findall(raw)) == 1 and YEAR.findall(raw) == YEAR.findall(g):
                found.append("year")
            continue
        if len(a) > 3 and a in n:
            found.append("answer in gold")
        elif len(n) > 3 and n in a:
            found.append("gold in answer")
        else:
            ya, yg = YEAR.findall(a), YEAR.findall(n)
            if ya and yg and ya[0] == yg[0]:
                found.append("year")
    return min(found, key=RULES.index) if found else None


__all__ = ["NON_ANSWERS", "RULES", "non_answer", "rule"]
