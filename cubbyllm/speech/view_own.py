"""cubbyllm/speech/view_own.py -- what a line's own text lets its description say (order 0073, point 7's variant):
round 2's rule 9 (`view_words`), for a description that is written before its text.

The rule, on a line's text:
- **numbers:** a number compared is in the text as digits or as a number word (English or French), 0 and 1 always;
- **positions:** 1, one the text gives by an ordinal ("third", "troisième") or a number of the text; ``last`` always;
- **strings:** one of its column's own values (`view_words.FIXED`) or a string the text quotes ("..." or « ... »),
  as quoted or, where the view compares lowercased, in lower case; free when the column has no values of its own
  and the text quotes nothing;
- **views:** a view the text names ("words", "mots"); any view when it names none.

`allows(form, text)` says whether a description keeps the rule on its line's text (the host's reading, on
`view_words`' own tables). `Own(instruction, start)` is the same rule for the grammar, which writes the description
first: for the text from ``start`` to the next line break it gives every value with **where the text holds it**
(the place a line must reach for its text to have that value), so the grammar can let a description use a value
and then keep the line going until it is there.

Wiring: STANDALONE (`view_split.parse` and `view_machine` call it; nothing in the serve path: CB_RULES stays off).
"""
from __future__ import annotations

import re
import unicodedata

from cubbyllm.speech import rule_guard as RG
from cubbyllm.speech import view_words as VW
from cubbyllm.speech.view_spec import FILTERS

__wiring__ = "STANDALONE"

FIXED, NOCASE, MAX_N = VW.FIXED, VW.NOCASE, VW.MAX_N
QUOTED = re.compile(r'"("|[^"]*)"|« (»|[^»]*?) »')         # a value can be the quote mark itself
_RUN = re.compile(r"(?<![\d.,])\d+(?!\d)")


def quoted(text: str) -> list[tuple[str, int]]:
    """(a string the text quotes, where its closing quote ends), in order."""
    return [(m.group(1) if m.group(1) is not None else m.group(2), m.end()) for m in QUOTED.finditer(text)]


def allows(form: dict, text: str) -> bool:
    """Whether a description (either spelling) says only what its line's text says."""
    low, view = RG._fold(text.lower()), form["view"]
    nums, pos = VW.allowed(low)
    said = {s for s, _ in quoted(text)}

    def value(col: str, v) -> bool:
        if FILTERS[view][col] == "num":
            return v in nums
        ok = set(FIXED.get(col, ())) | said | ({s.lower() for s in said} if col in NOCASE else set())
        return v in ok if ok else True
    if view not in VW.named(low) or (form["q"] in ("n", "count") and form["v"] not in nums):
        return False
    if form["q"] == "at" and not ((form["pos"] == -1 or form["pos"] - 1 in pos) and value(form["col"], form["val"])):
        return False
    return all(value(c["col"], c["v"]) for c in form.get("where", []))


def fold(text: str) -> tuple[str, list[int]]:
    """The text folded as `rule_guard` folds it, and for each folded character where its source character ends."""
    out, ends = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKD", ch.lower().replace("’", "'")):
            if not unicodedata.combining(c):
                out.append(c)
                ends.append(i + 1)
    return "".join(out), ends


class Own:
    """The values the text from ``start`` to the next line break holds, each with the place where the text has it
    (``start`` for a value that needs no text: 0, 1, the first position, a column's own values)."""

    def __init__(self, instruction: str, start: int):
        stop = instruction.find("\n", start)
        stop = len(instruction) if stop < 0 else stop
        while stop > start and instruction[stop - 1].isspace():
            stop -= 1
        self.start, self.stop = start, stop
        text = instruction[start:stop]
        low, ends = fold(text)

        def first(found: list[int]) -> int | None:
            """The earliest of some places in the folded text, as a place in the instruction."""
            return start + ends[min(found) - 1] if found else None
        self.numbers = {0: start, 1: start}
        for m in _RUN.finditer(low):
            n = int(m.group())
            if str(n) == m.group() and 2 <= n <= MAX_N:
                self.numbers.setdefault(n, first([m.end()]))
        for n, words in RG.NUM_WORDS.items():
            at = first([m.end() for w in words for m in [re.search(rf"(?<![\w-]){re.escape(RG._fold(w))}(?![\w-])", low)] if m])
            if at is not None and 2 <= n <= MAX_N:
                self.numbers[n] = min(at, self.numbers.get(n, at))
        self.positions = {1: start}
        for n, words in VW.ORDINALS.items():
            at = first([low.find(w) + len(w) for w in words if w in low])
            if at is not None and n > 1:
                self.positions[n] = at
        for n, at in self.numbers.items():
            if n > 1:
                self.positions[n] = min(at, self.positions.get(n, at))
        self.views = {}
        for v, words in VW.VIEW_WORDS.items():
            at = first([low.find(w) + len(w) for w in words if w in low])
            if at is not None:
                self.views[v] = at
        self.view_cap = min(self.views.values(), default=None)      # a line that ends before it names no view
        self.quotes: dict = {}
        for s, end in quoted(text):
            self.quotes.setdefault(s, start + end)
        self.quote_cap = min(self.quotes.values(), default=None)    # a line that ends before it quotes nothing
        self.inside = {start + j for m in QUOTED.finditer(text) for j in range(m.start() + 1, m.end())}   # no line ends here

    def strings(self, col: str) -> dict:
        """{a string the column may be compared with: where the text holds it}: its own values, and the quoted ones."""
        got = {s: self.start for s in FIXED.get(col, ())}
        for s, at in self.quotes.items():
            for x in {s, s.lower()} if col in NOCASE else {s}:      # as quoted, or lowercased as the view compares
                got[x] = min(at, got.get(x, at))
        return got


__all__ = ["FIXED", "NOCASE", "Own", "allows", "fold", "quoted"]
