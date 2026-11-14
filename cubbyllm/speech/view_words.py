"""cubbyllm/speech/view_words.py -- rule 9 of the second reader's grammar: a part says only what its own words say
(order 0067, step 5).

The dev read with rules 1-8 showed most wrong parts naming numbers and views their text does not hold: "fewer than 48
words" read as ``n_paragraph <= 48``; "exactly one paragraph" as ``n_word <= 1``; "a number of words other than 5" as
``x3 < 3``. Every one of the generator's 12,923 parts keeps the rule below, so it costs no right answer:

- **numbers:** every number a part compares (``x OP N``, ``k OP N``, ``n_V OP N``) is in the part's own text, as digits
  or as a number word (English or French, `rule_guard`'s), 0 and 1 always allowed ("no", "a", "any" say them);
- **positions:** a position (``assign p = N``) is the first (0) or one the text gives by an ordinal ("third", "troisième")
  or a number;
- **views:** every view a part reads is named in its text ("words", "mots" for ``word``; "list", "élément" for
  ``item``...);
- **one part per check:** a part is written for each ``check N`` line of the split and for nothing else; the join
  starts once every check has its part.

A part's text is the reader's own split, the lines ``check N | ...`` (and ``check N condition | ...``) it copied from
the instruction, so the rule needs no gold and holds when Cubby reads a rule for real.

`texts(head)` -> {part number: its folded text}; `words_ok(prog, texts)` -> whether the program written so far keeps the
rule (a number still being written is judged as a prefix).

Wiring: STANDALONE (`view_types.TypedGrammar` calls it; nothing in the serve path).
"""
from __future__ import annotations

import re
from functools import lru_cache

from cubbyllm.speech import reply_views as RV
from cubbyllm.speech import rule_guard as RG

__wiring__ = "STANDALONE"

NAMES = sorted({f"n_{v}" for v in RV.COLUMNS} | {f"{v}_{c}" for v, cs in RV.COLUMNS.items() for c in cs})

VIEW_WORDS = {"word": ("word", "mot"), "sentence": ("sentence", "phrase"), "line": ("line", "ligne"),
              "paragraph": ("paragraph", "paragraphe"), "item": ("item", "list", "element", "liste", "bullet", "puce"),
              "char": ("letter", "lettre", "character", "caractere", "capital", "majuscule", "minuscule", "lowercase",
                       "uppercase", "digit", "chiffre", "punctuation", "ponctuation", "space", "espace"),
              "json": ("json", "field", "key", "champ", "cle")}
ORDINALS = {1: ("first", "premier", "premiere"), 2: ("second", "deuxieme", "seconde"), 3: ("third", "troisieme"),
            4: ("fourth", "quatrieme"), 5: ("fifth", "cinquieme"), 6: ("sixth", "sixieme"), 7: ("seventh", "septieme"),
            8: ("eighth", "huitieme"), 9: ("ninth", "neuvieme"), 10: ("tenth", "dixieme")}
MAX_N = 2000
_CHECK = re.compile(r"^check (\d+)(?: condition)? \| (.*)$")
_NUM = re.compile(r"(?:x\d+|k\d+|n_[a-z]+) (?:<=|>=|==|!=|<|>) (\d+)")
_POS = re.compile(r"assign p\d+ = (\d+)")
_FUNC = re.compile(r"    function part(\d+)\(([^)]*)")


def texts(head: str) -> dict:
    out: dict = {}
    for ln in head.split("\n"):
        m = _CHECK.match(ln)
        if m:
            out[m.group(1)] = (out.get(m.group(1), "") + " " + RG._fold(m.group(2).lower())).strip()
    return out


@lru_cache(maxsize=4096)
def allowed(low: str) -> tuple[frozenset, frozenset]:
    """(numbers a comparison may use, positions a part may take) for one part's folded text."""
    nums = frozenset({0, 1} | {n for n in range(2, MAX_N + 1) if str(n) in low and RG._has_number(low, n)}
                     | {n for n, ws in RG.NUM_WORDS.items() if n <= MAX_N and RG._has_number(low, n)})
    pos = frozenset({0} | {n - 1 for n, ws in ORDINALS.items() if any(w in low for w in ws)}
                    | {n - 1 for n in nums if n >= 1})
    return nums, pos


def _fits(digits: str, at_end: bool, ok: frozenset) -> bool:
    return any(str(a).startswith(digits) for a in ok) if at_end else int(digits) in ok


def _view(name: str) -> str:
    return name[2:] if name.startswith("n_") else name.split("_")[0]


def named(low: str) -> set:
    """The views a part's text names; every view when it names none (a check line the reader split oddly: the rule
    then leaves its views free rather than leave the reader no way on)."""
    got = {v for v, ws in VIEW_WORDS.items() if any(w in low for w in ws)}
    return got or set(VIEW_WORDS)


def words_ok(prog: str, part_texts: dict) -> bool:
    """The k-th part is the k-th check line's (in the order the split wrote them, whatever their numbers); with no
    check line the rule leaves the program free (the split decides that, not this rule)."""
    order = list(part_texts.values())
    if not order:
        return True
    part, lines = None, prog.split("\n")
    n_funcs = sum(1 for ln in lines[:-1] if _FUNC.match(ln))
    for ln in lines:
        if ln.startswith("    f") and not _FUNC.match(ln) and n_funcs >= len(order):
            return False                                     # every check has its part: no other function begins
        f = re.match(r"    function part(\d*)", ln)
        if f:
            if not any(str(j).startswith(f.group(1)) for j in range(1, len(order) + 1)):
                return False
            m = _FUNC.match(ln)
            if m:
                part = int(m.group(1))
                if not 1 <= part <= len(order):
                    return False
                views = named(order[part - 1])
                *whole, tail = m.group(2).split(",")             # a name is judged as it is typed, not once typed
                if any(_view(n.split(":")[0].strip()) not in views for n in whole if n.strip()):
                    return False
                t = tail.strip().split(":")[0]
                if t and not any(n.startswith(t) and _view(n) in views for n in NAMES):
                    return False
            continue
        if ln.startswith("    public function solve"):
            part = None
            continue
        if part is not None:
            nums, pos = allowed(order[part - 1])
            for rx, ok in ((_NUM, nums), (_POS, pos)):
                for m in rx.finditer(ln):
                    if not _fits(m.group(1), m.end() == len(ln), ok):
                        return False
        elif ln.startswith(("        c", "        r")) and n_funcs < len(order):
            return False                                     # a check without its part: the join waits
    return True


__all__ = ["ORDINALS", "VIEW_WORDS", "allowed", "texts", "words_ok"]
