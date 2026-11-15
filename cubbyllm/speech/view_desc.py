"""cubbyllm/speech/view_desc.py -- a check's description on one line (order 0073): what the reader writes for a check
and the host reads back into the spec language (`view_spec`).

    n <view> <op> <number>
    count <view> <op> <number> where <condition>[ and <condition>]
    any | none | all <view> where <condition>[ and <condition>]
    at <view> <position> <column> <op> <value>[ absent pass]

A condition is ``<column> <op> <value>``; a position is ``1``, ``2``, ... or ``last``; a number is digits (at most
`MAX_DIGITS`); a string is in double quotes as JSON writes it (at most `MAX_STRING` characters); ``<op>`` is one of
six for a number column and ``==`` or ``!=`` for a string column; a column is one of its view's. Every slot is from a
closed set, so a description either reads back or names the slot that is not in its set.

**Arm W** (as said) may write ``not `` before the quantifier, before a condition and before a position's column;
**arm M** (folded) never does. `view_spec.fold` turns the first into the second.

`write(form)` -> the line of a said form (either spelling); `read(text, arm)` -> the form, or ValueError;
`read_at(text, i, arm)` -> (the form, where it ends), for a reader that goes on after it.

Wiring: STANDALONE (the reader's form `view_split` and its grammar use it; nothing in the serve path: CB_RULES off).
"""
from __future__ import annotations

import json
import re

from cubbyllm.speech.view_spec import FILTERS, NUM_OPS, STR_OPS

__wiring__ = "STANDALONE"

QUANT = ("n", "count", "any", "none", "all", "at")
VIEWS = tuple(FILTERS)
MAX_CONDS, MAX_DIGITS, MAX_STRING = 2, 4, 40
_NUM = re.compile(r"0|[1-9]\d*")
_STR = re.compile(r'"(?:[^"\\]|\\.)*"')


def _value(v) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, str) else str(v)


def _cond(c: dict) -> str:
    return ("not " if c.get("not") else "") + f"{c['col']} {c['op']} {_value(c['v'])}"


def write(form: dict) -> str:
    q, view = form["q"], form["view"]
    if q == "at":
        own = {"col": form["col"], "op": form["op"], "v": form["val"], "not": form.get("not")}
        return (f"at {view} {'last' if form['pos'] == -1 else form['pos']} {_cond(own)}"
                + (" absent pass" if form.get("absent") == "pass" else ""))
    head = ("not " if form.get("not") else "") + f"{q} {view}"
    if q in ("n", "count"):
        head += f" {form['op']} {form['v']}"
    return head if q == "n" else head + " where " + " and ".join(_cond(c) for c in form["where"])


class _Cur:
    """A place in the text; each take names the slot it could not fill."""

    def __init__(self, text: str, i: int = 0):
        self.text, self.i = text, i

    def lit(self, s: str) -> bool:
        if self.text.startswith(s, self.i):
            self.i += len(s)
            return True
        return False

    def need(self, s: str, what: str) -> None:
        if not self.lit(s):
            raise ValueError(f"{what} expected at {self.text[self.i:self.i + 20]!r}")

    def one(self, options, what: str) -> str:
        for o in sorted(options, key=len, reverse=True):       # a choice ends at a space or at the end of the text
            j = self.i + len(o)
            if self.text.startswith(o, self.i) and (j == len(self.text) or self.text[j] == " "):
                self.i = j
                return o
        raise ValueError(f"{what} not in its set at {self.text[self.i:self.i + 20]!r}")

    def number(self, what: str) -> int:
        m = _NUM.match(self.text, self.i)
        if not m or len(m.group()) > MAX_DIGITS:
            raise ValueError(f"{what} is not a number of at most {MAX_DIGITS} digits at {self.text[self.i:self.i + 20]!r}")
        self.i = m.end()
        return int(m.group())

    def string(self, what: str) -> str:
        m = _STR.match(self.text, self.i)
        try:
            v = json.loads(m.group()) if m else None
        except ValueError:
            v = None
        if not isinstance(v, str) or len(v) > MAX_STRING or _value(v) != m.group():   # one spelling a value
            raise ValueError(f"{what} is not a string of at most {MAX_STRING} characters at {self.text[self.i:self.i + 20]!r}")
        self.i = m.end()
        return v


def _read_cond(cur: _Cur, view: str, arm: str) -> dict:
    neg = arm == "w" and cur.lit("not ")
    col = cur.one(FILTERS[view], f"a column of {view}")
    num = FILTERS[view][col] == "num"
    cur.need(" ", "a space")
    op = cur.one(NUM_OPS if num else STR_OPS, "the comparison")
    cur.need(" ", "a space")
    c = {"col": col, "op": op, "v": cur.number("the value") if num else cur.string("the value")}
    return dict(c, **{"not": True}) if neg else c


def read_at(text: str, i: int = 0, arm: str = "m") -> tuple[dict, int]:
    cur = _Cur(text, i)
    neg = arm == "w" and cur.lit("not ")
    q = cur.one(QUANT, "the quantifier")
    if neg and q == "at":
        raise ValueError("a position's `not` goes before its column")
    cur.need(" ", "a space")
    view = cur.one(VIEWS, "the view")
    if q == "at":
        cur.need(" ", "a space")
        pos = -1 if cur.lit("last") else cur.number("the position")
        if pos == 0:
            raise ValueError("a position starts at 1")
        cur.need(" ", "a space")
        c = _read_cond(cur, view, arm)
        form = {"q": "at", "view": view, "pos": pos, "col": c["col"], "op": c["op"], "val": c["v"]}
        if c.get("not"):
            form["not"] = True
        if cur.lit(" absent pass"):
            form["absent"] = "pass"
        return form, cur.i
    form = {"q": q, "view": view}
    if q in ("n", "count"):
        cur.need(" ", "a space")
        op = cur.one(NUM_OPS, "the comparison")
        cur.need(" ", "a space")
        form.update(op=op, v=cur.number("the number"))
    if q != "n":
        cur.need(" where ", "`where`")
        form["where"] = [_read_cond(cur, view, arm)]
        while len(form["where"]) < MAX_CONDS and cur.lit(" and "):
            form["where"].append(_read_cond(cur, view, arm))
    if neg:
        form["not"] = True
    return form, cur.i


def read(text: str, arm: str = "m") -> dict:
    form, j = read_at(text, 0, arm)
    if j != len(text):
        raise ValueError(f"something after the description: {text[j:j + 20]!r}")
    return form


__all__ = ["MAX_CONDS", "MAX_DIGITS", "MAX_STRING", "QUANT", "VIEWS", "read", "read_at", "write"]
