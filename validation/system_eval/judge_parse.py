"""validation/system_eval/judge_parse.py -- order 0075 step 6: a judge model's grade, read so that a non-answer is never
right. Built for the later frontier run (its own order); used by nothing yet.

NeMo Gym's SimpleQA and AA-Omniscience verifiers (issue #5430) took the first of A, B, C, D found anywhere in the
judge's text, so ``NOT_ATTEMPTED`` (its A), ``C: PARTIAL_ANSWER`` and ``Answer: B`` all scored as A or worse. Here:
- only a standalone grade letter counts: the whole reply ("B"), or a letter at the start of a line, alone or before
  ``:``, ``.``, ``)`` or ``-`` ("B: INCORRECT"), optionally after "Answer", "Grade" or "Final grade" ("Answer: B");
  a letter inside a word or an article in a sentence ("A correct answer ...") does not;
- several distinct letters, or none, mean not attempted;
- a letter whose label the reply contradicts (it names another class word: "C: PARTIAL_ANSWER" on SimpleQA, whose C is
  NOT_ATTEMPTED) means not attempted.

``parse(text, bench)`` -> {"letter": "B" | None, "grade": "CORRECT" | "INCORRECT" | "PARTIAL_ANSWER" | "NOT_ATTEMPTED"}.
Only "CORRECT" is right.

Wired: STANDALONE (for the frontier run's order; nothing calls it yet).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

LABELS = {
    "simpleqa": {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"},
    "aa": {"A": "CORRECT", "B": "INCORRECT", "C": "PARTIAL_ANSWER", "D": "NOT_ATTEMPTED"},
}
CLASS_WORDS = {"CORRECT": r"(?<!IN)CORRECT", "INCORRECT": r"INCORRECT", "PARTIAL_ANSWER": r"PARTIAL",
               "NOT_ATTEMPTED": r"NOT[ _]ATTEMPTED"}
LETTER = re.compile(r"^\s*(?:(?:final\s+)?(?:answer|grade)\s*[:=-]?\s*)?\(?([A-D])\)?\s*(?:[:.)\-]|$)", re.I | re.M)


def parse(text: str | None, bench: str = "simpleqa") -> dict:
    labels = LABELS[bench]
    t = str(text or "")
    letters = {m.group(1).upper() for m in LETTER.finditer(t) if m.group(1).upper() in labels}
    if len(letters) != 1:
        return {"letter": None, "grade": "NOT_ATTEMPTED"}
    letter = letters.pop()
    grade = labels[letter]
    up = t.upper()
    named = {c for c, rx in CLASS_WORDS.items() if re.search(rx, up)}
    if named and named != {grade}:
        return {"letter": letter, "grade": "NOT_ATTEMPTED"}
    return {"letter": letter, "grade": grade}


__all__ = ["LABELS", "parse"]
