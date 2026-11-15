"""Order 0067, the view reader's round 2 (Nick, 2026-11-13: "weight the key tokens"): which answer tokens carry a
program's meaning, and their loss weights.

The first reads showed the split mostly right and the program wrong where its meaning sits: a comparison reversed, a
number or a view the instruction does not give, a join's 0 for a 1. Those characters are a few percent of an answer,
so a mean over its tokens barely sees them. Here, in the program (after the split's ``---``), these are key:
- a comparison's operator (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``) and the number or quoted string it compares with;
- a position (``assign pK = N``);
- every view name (``n_word``, ``sentence_n_word``, ...);
- the join's test (``if (pK == 0)`` / ``== 1``).
`weights(tk, answer, w)` gives each token of ``tk.encode(" " + answer)`` (and the end token) weight ``w`` when it
overlaps a key character, else 1.0, in the order `talk_data.encode_records` makes them.

Order 0073's reader writes no program: its answer is the split alone, each check with its description. There the key
characters are each check line's join tag and description (`split_spans`): the quantifier, the view, the column,
the comparison, the value, a ``not``.

Wiring: STANDALONE (`validation/train_talk_torch.py --key-weight`; off by default).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

SEP = "\n---\n"
_KEY = [re.compile(r"[\s(](<=|>=|==|!=|<|>) "),                         # an operator inside a test
        re.compile(r"(?:<=|>=|==|!=|<|>) (\d+|\"(?:[^\"\\\n]|\\.)*\")"),   # what it compares with
        re.compile(r"assign p(\d+ = \d+)"),                            # a position
        re.compile(r"\b(n_[a-z]+|(?:char|word|sentence|line|paragraph|item|json)_[a-z_]+)\b")]   # a view name


def split_spans(answer: str) -> list[tuple[int, int]]:
    """Order 0073's form (`cubbyllm/speech/view_split.py`: the split alone, no program): on each ``check`` line the
    join tag and the description, from after ``check `` to the bar. These are the slots, where the answer's meaning
    sits. A check line whose description does not read gives no span."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cubbyllm.speech import view_desc as D
    from cubbyllm.speech import view_split as V
    out, at = [], 0
    for line in answer.split("\n"):
        if line.startswith("check "):
            i = 6
            join = next((j for j in V.JOINS if line.startswith(j + " ", i)), None)
            try:
                _, end = D.read_at(line, i + (len(join) + 1 if join else 0), "w")
                if line.startswith(V.BAR, end):
                    out.append((at + i, at + end))
            except ValueError:
                pass
        at += len(line) + 1
    return out


def key_spans(answer: str) -> list[tuple[int, int]]:
    """Character spans of the key characters in ``answer``: in a program after the split's separator (0067), or with
    no separator the slots of each check line (0073, `split_spans`)."""
    at = answer.find(SEP)
    if at < 0:
        return split_spans(answer)
    start = at + len(SEP)
    prog = answer[start:]
    return sorted((start + m.start(1), start + m.end(1)) for rx in _KEY for m in rx.finditer(prog))


def weights(tk, answer: str, w: float) -> list[float]:
    """One weight per token of ``tk.encode(" " + answer).ids``, plus 1.0 for the end token."""
    enc = tk.encode(" " + answer)
    spans = [(s + 1, e + 1) for s, e in key_spans(answer)]             # the leading space shifts every character
    out, j = [], 0
    for a, b in enc.offsets:
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        out.append(w if j < len(spans) and spans[j][0] < b and a < spans[j][1] else 1.0)
    return out + [1.0]


__all__ = ["key_spans", "split_spans", "weights"]
