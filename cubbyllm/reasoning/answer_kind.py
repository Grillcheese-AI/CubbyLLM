"""cubbyllm/reasoning/answer_kind.py -- order 0078's addendum, part 1: a try whose VM value cannot answer the question
does not vote. `kept` turns such a value into None, which `speak_vote.vote` counts as a try and never as a vote.

The three rules, fixed in the order (nothing else is added):
- ``negative``: a negative value is out;
- ``whole``: a "how many" question whose asked noun is not a measure (``MEASURES``, the order's closed list) gets a
  whole number;
- ``money``: a "how much" question with a money word (``MONEY``: $, dollars, cost, pay, spend, earn, save, price) gets
  at most 2 decimals.

How the question is read (this module's choices, reported with the read):
- The ask is the last "how many" or "how much" in the question. A question with neither is checked by ``negative`` only.
- The asked noun is the first word after "how many", before any punctuation, that is not a modifier (``SKIP``: more,
  fewer, total, square, of, the, ...). A verb there first ("how many did she buy") means no noun was asked, and no noun
  is not a measure.
- A money word is looked for in the whole question (the prices are in the story; the ask is often "how much money").
  The words' inflections count (paid, spent, costs); so do a measure's singular and its British spelling. The lists are
  not widened.
- Whole and 2 decimals are read within ``speak_vote.TOL`` (relative), as the vote groups values.

Each rule is checked on GSM8K train's 7,473 golds before the dev read (`validation/exp_0078_dev_score.gold_check`); a
rule refusing more than 7 of them (0.1%) is dropped there.

Wiring: STANDALONE (order 0078's dev read, `validation/exp_0078_dev_score.py --rules`; the serve path takes it only as
part of the speak rule frozen before test, on Nick's word).
"""
from __future__ import annotations

import re

from cubbyllm.reasoning.speak_vote import TOL

__wiring__ = "STANDALONE"

RULES = ("negative", "whole", "money")

MEASURES = frozenset("dollars cents hours minutes seconds days weeks months years miles kilometers meters feet inches "
                     "pounds ounces kilograms grams liters gallons cups percent degrees".split())
_SPELLING = {"foot": "feet", "inch": "inches", "kilometres": "kilometers", "kilometre": "kilometers",
             "metres": "meters", "metre": "meters", "litres": "liters", "litre": "liters"}

SKIP = frozenset("more fewer less total extra additional different other whole full complete square cubic of the a an "
                 "his her their its my your our these those this that such".split())
_VERB = frozenset("did does do is are was were will would can could should has have had must might may".split())

MONEY = re.compile(r"\$|\b(?:dollars?|costs?|costing|costed|pays?|paid|paying|spends?|spent|spending|earns?|earned|"
                   r"earning|saves?|saved|saving|prices?|priced)\b")
_ASK = re.compile(r"\bhow (many|much)\b")


def is_measure(noun: "str | None") -> bool:
    if not noun:
        return False
    w = _SPELLING.get(noun, noun)
    return w in MEASURES or w + "s" in MEASURES


def asked(question: str) -> "tuple | None":
    """("many", the asked noun or None) | ("much", a money word in the question) | None (neither is asked)."""
    q = question.lower()
    hits = list(_ASK.finditer(q))
    if not hits:
        return None
    last = hits[-1]
    if last.group(1) == "much":
        return ("much", bool(MONEY.search(q)))
    clause = re.split(r"[?.!,;:]", q[last.end():], maxsplit=1)[0]
    for w in re.findall(r"[a-z]+", clause):
        if w in SKIP:
            continue
        return ("many", None if w in _VERB else w)
    return ("many", None)


def _within(v: float, scale: float) -> bool:
    x = v * scale
    return abs(x - round(x)) <= TOL * max(1.0, abs(x))


def broken(value, question: str, rules=RULES) -> "str | None":
    """The first rule ``value`` breaks for ``question`` (in ``RULES``' order), or None. A None value breaks nothing."""
    if value is None:
        return None
    v = float(value)
    if "negative" in rules and v < 0:
        return "negative"
    a = asked(question)
    if a and a[0] == "many" and "whole" in rules and not is_measure(a[1]) and not _within(v, 1):
        return "whole"
    if a and a[0] == "much" and a[1] and "money" in rules and not _within(v, 100):
        return "money"
    return None


def kept(values: list, question: str, rules=RULES) -> list:
    """The tries' values with each one breaking a rule replaced by None (a try, never a vote)."""
    return [None if v is not None and broken(v, question, rules) else v for v in values]


__all__ = ["MEASURES", "MONEY", "RULES", "SKIP", "asked", "broken", "is_measure", "kept"]
