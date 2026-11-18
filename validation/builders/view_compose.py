"""Order 0073, step 3: a long instruction's text from its parts and groups, with what each check says.

Order 0067's `build_long_rows.compose`, with three changes (the design is in the order's Report):
- **each connective before and after its partner:** "If C, A." or "A if C."; "A, unless B." or "Unless B, A.";
  "Si C, A." or "A si C."; "A, sauf si B." or "Sauf si B, A."; half each. "Either A, or B." has one order;
- **each check segment says how it was worded** (``how``: the imperative, a condition, goal first), so its said form
  can be read back (`view_said`);
- **the check that carries the connective is tagged** (``join``: ``[kind, "prev" | "next"]``, the neighbouring check
  it ties to). "Either" sits on the first of its two checks.

`compose(rng, parts, groups, lang, phrases, share)` -> (text, segments, the order the text states the parts in).
`renumber` numbers the parts in that order; `said_forms` gives each part's forms from its segment.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import view_long_pools as P  # noqa: E402
import view_said as S  # noqa: E402

__wiring__ = "STANDALONE"

IF = {"en": ("If ", " if "), "fr": ("Si ", " si ")}
UNLESS = {"en": ("Unless ", ", unless "), "fr": ("Sauf si ", ", sauf si ")}
EITHER = {"en": ("Either ", ", or "), "fr": ("Soit ", ", soit ")}


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:]


def _imp(rng, part: dict, lang: str, share: float) -> tuple[str, str]:
    """(the part as an instruction, how it was worded): goal first for a ``share`` of the parts that have one."""
    if rng.random() < share:
        text = S.say(part, lang, rng, "inverted")[0]
        if text is not None:
            return text, "inverted"
    return S.say(part, lang, rng, "imp")[0], "imp"


def _group(lay, word: list, g: tuple, parts: list, lang: str, share: float) -> tuple[list, list]:
    """One group's pieces [(text, segment)] and the parts in the order they are said."""
    def check(i, how, **kw):
        return dict({"kind": "check", "part": i, "how": how}, **kw)
    if g[0] == "one":
        text, how = _imp(word[g[1]], parts[g[1]], lang, share)
        return [(text, check(g[1], how))], [g[1]]
    if g[0] == "either":
        a, b = g[1], g[2]
        (ta, ha), (tb, hb) = _imp(word[a], parts[a], lang, share), _imp(word[b], parts[b], lang, share)
        return [(EITHER[lang][0] + _lower1(ta.rstrip(".")), check(a, ha, join=["either", "next"])),
                (EITHER[lang][1] + _lower1(tb), check(b, hb))], [a, b]
    a, c = (g[2], g[1]) if g[0] == "if" else (g[1], g[2])   # a: what is asked; c: the condition
    words = IF if g[0] == "if" else UNLESS
    cond = S.say(parts[c], lang, word[c], "cond")[0]
    text, how = _imp(word[a], parts[a], lang, share)
    if lay.random() < 0.5:                                 # the connective first: it ties to the check after it
        return [(words[lang][0] + cond + ", ", check(c, "cond", role="condition", join=[g[0], "next"])),
                (_lower1(text), check(a, how))], [c, a]
    return [(text.rstrip("."), check(a, how)),
            (words[lang][1] + cond + ".", check(c, "cond", role="condition", join=[g[0], "prev"]))], [a, c]


def compose(rng, parts: list, groups: list, lang: str, phrases: dict, share: float = 0.0,
            task: bool = True) -> tuple[str, list, list]:
    """`rng` gives one number; the layout draws from its own stream and each part's wording from its own, so two
    rows composed from the same state that differ in one part (a minimal pair) differ in that part's words only.
    ``task=False`` (order 0077: a real prompt is the task): no task sentence is drawn from the lists."""
    base = rng.getrandbits(48)
    rng, word = random.Random(base), [random.Random(base * 131 + i + 1) for i in range(len(parts))]
    pieces, order = [], []
    for g in groups:
        p, o = _group(rng, word, g, parts, lang, share)
        pieces.append(p)
        order += o
    others = []
    if task and rng.random() < 0.8:
        s, key = P.task_sentence(rng, lang)
        others.append(("task", [(s, {"kind": "task", "task": key})]))
    for _ in range(rng.choice([0, 1, 1, 2])):
        key, w = rng.choice(phrases[lang])
        others.append(("asked", [(w, {"kind": "asked", "asked": key})]))
    if rng.random() < 0.4:
        others.append(("process", [(rng.choice(P.PROCESS[lang]), {"kind": "process"})]))
    for _ in range(rng.choice([0, 1, 1, 2])):
        others.append(("none", [(rng.choice(P.CONTEXT[lang]), {"kind": "none"})]))
    seq = [("check", p) for p in pieces]
    for kind, p in others:                                # a task mostly first, the rest anywhere
        at = 0 if kind == "task" and rng.random() < 0.7 else rng.randrange(len(seq) + 1)
        seq.insert(at, (kind, p))
    text, segs = "", []
    for n, (_, p) in enumerate(seq):
        if n:
            text += rng.choice([" ", " ", "\n"])
        for t, seg in p:
            segs.append(dict(seg, start=len(text), end=len(text) + len(t)))
            text += t
    return text, segs, order


def renumber(parts: list, groups: list, segs: list, order: list) -> tuple[list, list, list]:
    """The parts numbered in the order the text states them."""
    new = {old: i for i, old in enumerate(order)}
    return ([parts[old] for old in order], [(g[0], *[new[x] for x in g[1:]]) for g in groups],
            [dict(s, part=new[s["part"]]) if "part" in s else s for s in segs])


def said_forms(parts: list, segs: list, lang: str) -> list[dict] | None:
    """Each part's ``{"how", "w", "m"}`` (arm W's form and arm M's), from the segment that says it; None when a
    wording has no description."""
    how = {s["part"]: s["how"] for s in segs if s["kind"] == "check"}
    out = []
    for i, part in enumerate(parts):
        w = S.said(part, lang, how[i])
        if w is None:
            return None
        out.append({"how": how[i], "w": w, "m": S.VS.fold(w)})
    return out


__all__ = ["compose", "renumber", "said_forms"]
