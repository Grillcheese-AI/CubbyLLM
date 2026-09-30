"""slots -- the emitter never copies a name or a number: the host places them (H-E15, 2026-09-29).

Wired: STANDALONE (a wrapper any host can put around any `Emitter`: `SlotEmitter(inner, index)`; the ask loop's
own path is untouched until a gate says to switch). Pinned by tests/reasoning/test_slots.py.

Panel round 3 (docs/research/2026-09-28-panel-round3-synthesis.md), converged on by five reviewers: a
VM-verified chain is not a correct answer when the model that wrote the program binds at chance -- a program
about the WRONG entity carries real facts, clears every hop and passes the value and claim checks. So no model
output carries a name or a number the host could have placed itself:

  the host    extracts the question's spans -- entities from a name index (longest match, left to right),
              numbers by position, referents the context graph resolved -- into a SlotTable: $E1, $E2, $N1, $R1
  the emitter sees the question ANNOTATED ("... of [E1: iridomyrmex bigi]?") and writes `bind frame, SEED, "$E1"`
  the host    checks the program (every slot it names exists; no entity literal that is not a slot; which spans
              went unused), FILLS the slots, and hands the filled program to the same disposer, walk and VM as
              before. Filling is a string substitution: deterministic, ledgered, and the copy the trunk cannot
              be trusted with.

What it measures for free: the extractor's miss rate (a question with no span the index knows), the wrong-slot
rate (the model named a slot the host had, but the wrong one -- a binding failure in another costume, the kill
of H-E15), and the copied-literal rate (the model wrote a name instead of a slot -- refused, never run).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..core.protocols import Wiring

__wiring__ = Wiring.STANDALONE

SLOT_RX = re.compile(r"\$(?P<kind>[ENRK])(?P<n>\d+)\b")      # K: a constant a world supplies (60 minutes per hour)
# a quoted literal bound into a frame: `bind frame, SEED, "..."` / `bind evt, AGENT, "..."`
BIND_LIT_RX = re.compile(r'bind\s+\w+\s*,\s*(?P<role>\w+)\s*,\s*"(?P<lit>(?:[^"\\]|\\.)*)"\s*;')
# a numeric literal assigned or added: `assign s0 = 48;` `add s1, 24;` `assign threshold = 29;`
NUM_LIT_RX = re.compile(r'(?P<op>assign|add|sub|mul|div)\s+(?P<reg>\w+)\s*(?:=|,)\s*(?P<num>-?\d+(?:\.\d+)?)\s*;')
NUM_IN_TEXT_RX = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?!\w)(?![.,]\d)")   # "29." at a sentence end is 29
# numbers the question states in WORDS (2026-09-30): 64% of v12e's arithmetic programs carried a constant no
# slot covered -- 30% of those a number word ("four people"), 26% a multiplicative ("twice", "half", "a dozen").
# They are the question's numbers as much as "48" is, so the host places them too. "half" is 2 because the
# harvest's convention is `div s0, 2`; a program that multiplies by 0.5 keeps its 0.5 (matching is by value).
_UNIT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEEN_WORDS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
               "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
               "ninety": 90}
MULTIPLE_WORDS = {"twice": 2, "double": 2, "doubled": 2, "half": 2, "triple": 3, "tripled": 3, "thrice": 3,
                  "dozen": 12, "dozens": 12}
NUMBER_WORDS = {**_UNIT_WORDS, **_TEEN_WORDS, **_TENS_WORDS, **MULTIPLE_WORDS}
NUM_WORD_RX = re.compile(
    r"(?<![\w-])(?:(?P<tens>" + "|".join(_TENS_WORDS) + r")(?:[- ](?P<unit>" + "|".join(_UNIT_WORDS) + r"))?"
    r"|(?P<word>" + "|".join(sorted({**_UNIT_WORDS, **_TEEN_WORDS, **MULTIPLE_WORDS}, key=len, reverse=True)) + r"))"
    r"(?![\w-])", re.I)


def num_value(text: str | None) -> float | None:
    """The number a literal or a span stands for: '3.50' and '3.5' are one number, '1,200' is 1200."""
    if text is None:
        return None
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def canon(x: float) -> str:
    """One spelling per number: 3.5, 1200, 0.25 (never 3.50 or 1.2e3)."""
    return ("%.10f" % x).rstrip("0").rstrip(".") if x != int(x) else str(int(x))


def number_words(question: str) -> list[tuple[str, float, int, int]]:
    """[(text, value, start, end)] of the numbers the question writes in words."""
    out = []
    for m in NUM_WORD_RX.finditer(question):
        if m.group("tens"):
            v = _TENS_WORDS[m.group("tens").lower()] + (_UNIT_WORDS[m.group("unit").lower()] if m.group("unit") else 0)
        else:
            v = NUMBER_WORDS[m.group("word").lower()]
        out.append((m.group(0), float(v), m.start(), m.end()))
    return out


ENTITY_ROLES = {"SEED", "AGENT", "OBJECT", "SUBJECT", "ENTITY", "TARGET", "PLACE", "PERSON", "EVENT"}
# the chain program's hop roles (`H1_CAPITAL`, `H2_INSTANCE`, ...) bind the walked objects: entities too
_HOP_ROLE_RX = re.compile(r"^H\d+_", re.I)


def is_entity_role(role: str) -> bool:
    """A role whose filler is a name the host places (a slot), not a relation word the emitter learns."""
    return role.upper() in ENTITY_ROLES or bool(_HOP_ROLE_RX.match(role))


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w']+", " ", s.lower()).split())


@dataclass(frozen=True)
class Span:
    id: str          # "$E1"
    text: str        # as it appears in the question
    start: int
    end: int
    kind: str        # E | N | R
    value: str | None = None   # a number written in words: the text is "four", the value "4" (what fill writes)

    @property
    def filled(self) -> str:
        return self.value if self.value is not None else self.text


@dataclass
class SlotTable:
    question: str
    spans: list = field(default_factory=list)

    def get(self, sid: str) -> Span | None:
        return next((s for s in self.spans if s.id == sid), None)

    def annotate(self) -> str:
        """The question with every span marked for the emitter: 'of [E1: iridomyrmex bigi]?'"""
        out, at = [], 0
        for s in sorted(self.spans, key=lambda s: s.start):
            if s.kind in "RK":                           # a referent / a world's constant is not in the text
                continue
            out.append(self.question[at:s.start]); out.append(f"[{s.id[1:]}: {s.text}]"); at = s.end
        out.append(self.question[at:])
        head = "".join(out)
        refs = [s for s in self.spans if s.kind in "RK"]
        if refs:
            head += "  " + "; ".join(f"[{s.id[1:]} = {s.text}]" for s in refs)
        return head

    def fill(self, program: str) -> str:
        def sub(m):
            s = self.get(m.group(0))
            return s.filled if s else m.group(0)
        return SLOT_RX.sub(sub, program)

    def check(self, program: str) -> "SlotVerdict":
        referenced = [m.group(0) for m in SLOT_RX.finditer(program)]
        missing = sorted({r for r in referenced if self.get(r) is None})
        copied = []
        for m in BIND_LIT_RX.finditer(program):
            lit = m.group("lit")
            if is_entity_role(m.group("role")) and not SLOT_RX.fullmatch(lit) and self._is_span_text(lit):
                copied.append(lit)                       # a name the host had as a slot, written out instead
        for m in NUM_LIT_RX.finditer(program):
            if self._is_span_text(m.group("num"), kind="N"):
                copied.append(m.group("num"))
        used = {r for r in referenced if self.get(r) is not None}
        unused = [s.id for s in self.spans if s.kind == "E" and s.id not in used]
        ok = not missing and not copied
        reason = ("unknown slot " + ", ".join(missing) if missing else
                  "copied instead of a slot: " + ", ".join(repr(c) for c in copied) if copied else "")
        return SlotVerdict(ok, reason, referenced, unused, copied)

    def _is_span_text(self, lit: str, kind: str = "E") -> bool:
        if kind == "N":                                  # by value: 3.5 copies "$3.50", 4 copies "four", 60 a $K
            v = num_value(lit)
            return v is not None and any(s.kind in "NK" and num_value(s.filled) == v for s in self.spans)
        n = _norm(lit)
        return any(s.kind == kind and _norm(s.text) == n for s in self.spans)


@dataclass
class SlotVerdict:
    ok: bool
    reason: str
    referenced: list
    unused_entities: list
    copied: list


class SpanIndex:
    """Entity spans by longest match over a name index (the history graph's names, a store's keys, the
    context graph's referents) -- normalized words, left to right, non-overlapping, longest first."""

    def __init__(self, names: Iterable[str], max_words: int = 8):
        self.by_len: dict[int, set] = {}
        self.max_words = max_words
        for n in names:
            w = tuple(_norm(n).split())
            if w and len(w) <= max_words:
                self.by_len.setdefault(len(w), set()).add(w)
        self.lengths = sorted(self.by_len, reverse=True)

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_len.values())

    def spans(self, question: str) -> list:
        """[(text, start, end)] of the question's entity mentions, in order."""
        toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w']+", question)]
        norm = [_norm(t) for t, _, _ in toks]
        out, i = [], 0
        while i < len(toks):
            hit = None
            for L in self.lengths:
                if i + L <= len(toks) and tuple(norm[i:i + L]) in self.by_len[L]:
                    hit = L; break
            if hit:
                s, e = toks[i][1], toks[i + hit - 1][2]
                out.append((question[s:e], s, e)); i += hit
            else:
                i += 1
        return out


def extract(question: str, index: SpanIndex | None = None, referents: dict | None = None,
            numbers: bool = True, words: bool = True, constants=None) -> SlotTable:
    """The slot table of a question: $E1.. its entity spans (from the index), $N1.. its numbers by position --
    digits, and with `words` the numbers it writes in words ("four", "twice", "a dozen"), one id per value --
    $R1.. the referents the context graph resolved ({'it': 'Siege of Vienna'}), and $K1.. the constants a world
    supplies for it (`constants`: [(text, value)], e.g. the arithmetic world's ("60 minutes per hour", 60)); a
    constant whose value the question already states is not repeated."""
    t = SlotTable(question)
    ents = index.spans(question) if index is not None else []
    # one id per distinct name (2026-09-29): a name mentioned twice -- in the question and in a fact
    # line -- is one entity and one slot, so the emitter has one right answer, not an arbitrary pick
    # among equal ids; every mention is annotated with it
    ids: dict[str, str] = {}
    for text, s, e in ents:
        sid = ids.setdefault(_norm(text), f"$E{len(ids) + 1}")
        t.spans.append(Span(sid, text, s, e, "E"))
    if numbers:
        taken = [(s.start, s.end) for s in t.spans]
        found = [(m.start(), m.end(), m.group(0).replace(",", ""), None) for m in NUM_IN_TEXT_RX.finditer(question)]
        if words:
            found += [(s, e, text, canon(v)) for text, v, s, e in number_words(question)]
        nids: dict[str, str] = {}
        for s, e, text, value in sorted(found, key=lambda f: f[0]):
            if any(a <= s < b for a, b in taken):
                continue
            v = num_value(value if value is not None else text)
            sid = nids.setdefault(canon(v) if v is not None else text, f"$N{len(nids) + 1}")
            t.spans.append(Span(sid, text, s, e, "N", value))
    for k, (word, target) in enumerate((referents or {}).items(), 1):
        t.spans.append(Span(f"$R{k}", str(target), -1, -1, "R"))
    stated = {num_value(s.filled) for s in t.spans if s.kind == "N"}
    kids: dict[str, str] = {}
    for text, value in constants or ():
        v = num_value(value)
        if v is None or v in stated or canon(v) in kids:
            continue
        kids[canon(v)] = f"$K{len(kids) + 1}"
        t.spans.append(Span(kids[canon(v)], str(text), -1, -1, "K", canon(v)))
    return t


def to_slots(prompt: str, program: str, index: SpanIndex | None = None) -> tuple[str, str, SlotTable, dict]:
    """Rewrite one training record into the slot form: the prompt annotated, the program's copied literals
    replaced by their slots. Returns (annotated prompt, slotted program, table, stats). The index may be
    None: then the program's own entity literals that occur in the prompt are the spans (the gold seed is
    known), which is how the existing harvest converts without a name index."""
    if index is None:
        lits = [m.group("lit") for m in BIND_LIT_RX.finditer(program) if is_entity_role(m.group("role"))]
        found = []
        for lit in dict.fromkeys(lits):
            m = re.search(re.escape(lit), prompt, re.I)
            if m:
                found.append(lit if prompt[m.start():m.end()] == lit else prompt[m.start():m.end()])
        index = SpanIndex(found)
    table = extract(prompt, index)
    out, replaced, kept = program, 0, 0
    for m in list(BIND_LIT_RX.finditer(program)):
        lit = m.group("lit")
        if not is_entity_role(m.group("role")):
            continue
        s = next((s for s in table.spans if s.kind == "E" and _norm(s.text) == _norm(lit)), None)
        if s is None:
            kept += 1
            continue
        out = out.replace(f'"{lit}"', f'"{s.id}"', 1); replaced += 1
    nums = 0
    for m in list(NUM_LIT_RX.finditer(program)):
        v = num_value(m.group("num"))
        s = next((s for s in table.spans if s.kind in "NK" and num_value(s.filled) == v), None)
        if s is not None:
            out = out.replace(m.group(0), m.group(0).replace(m.group("num"), s.id), 1); nums += 1
    return table.annotate(), out, table, {"entities_slotted": replaced, "entities_kept": kept, "numbers_slotted": nums,
                                         "spans": len([s for s in table.spans if s.kind == "E"])}


class SlotEmitter:
    """`Emitter` in, `Emitter` out: annotate -> emit -> check -> fill. A copied literal or an unknown slot is a
    refusal (the program never reaches the disposer); the verdict and the table ride on `.last` for the ledger."""

    def __init__(self, inner, index: SpanIndex | None = None, referents: Callable[[str], dict] | None = None):
        self.inner, self.index, self.referents = inner, index, referents
        self.last: dict = {}
        self.name = f"slots({getattr(inner, 'name', inner.__class__.__name__)})"

    def emit(self, prompt: str, max_new_tokens: int = 768, system: str | None = None, **kw) -> str:
        table = extract(prompt, self.index, self.referents(prompt) if self.referents else None)
        raw = self.inner.emit(table.annotate(), max_new_tokens=max_new_tokens, system=system, **kw) \
            if system is not None else self.inner.emit(table.annotate(), max_new_tokens=max_new_tokens, **kw)
        verdict = table.check(raw)
        self.last = {"table": table, "verdict": verdict, "raw": raw}
        if not verdict.ok:
            raise SlotRefused(verdict.reason)
        return table.fill(raw)


class SlotRefused(Exception):
    pass
