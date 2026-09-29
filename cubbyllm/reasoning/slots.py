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

SLOT_RX = re.compile(r"\$(?P<kind>[ENR])(?P<n>\d+)\b")
# a quoted literal bound into a frame: `bind frame, SEED, "..."` / `bind evt, AGENT, "..."`
BIND_LIT_RX = re.compile(r'bind\s+\w+\s*,\s*(?P<role>\w+)\s*,\s*"(?P<lit>(?:[^"\\]|\\.)*)"\s*;')
# a numeric literal assigned or added: `assign s0 = 48;` `add s1, 24;` `assign threshold = 29;`
NUM_LIT_RX = re.compile(r'(?P<op>assign|add|sub|mul|div)\s+(?P<reg>\w+)\s*(?:=|,)\s*(?P<num>-?\d+(?:\.\d+)?)\s*;')
NUM_IN_TEXT_RX = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.])")
ENTITY_ROLES = {"SEED", "AGENT", "OBJECT", "SUBJECT", "ENTITY", "TARGET", "PLACE", "PERSON", "EVENT"}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w']+", " ", s.lower()).split())


@dataclass(frozen=True)
class Span:
    id: str          # "$E1"
    text: str        # as it appears in the question
    start: int
    end: int
    kind: str        # E | N | R


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
            if s.kind == "R":                            # a referent is not in the text: it is what a pronoun means
                continue
            out.append(self.question[at:s.start]); out.append(f"[{s.id[1:]}: {s.text}]"); at = s.end
        out.append(self.question[at:])
        head = "".join(out)
        refs = [s for s in self.spans if s.kind == "R"]
        if refs:
            head += "  " + "; ".join(f"[{s.id[1:]} = {s.text}]" for s in refs)
        return head

    def fill(self, program: str) -> str:
        def sub(m):
            s = self.get(m.group(0))
            return s.text if s else m.group(0)
        return SLOT_RX.sub(sub, program)

    def check(self, program: str) -> "SlotVerdict":
        referenced = [m.group(0) for m in SLOT_RX.finditer(program)]
        missing = sorted({r for r in referenced if self.get(r) is None})
        copied = []
        for m in BIND_LIT_RX.finditer(program):
            lit = m.group("lit")
            if m.group("role").upper() in ENTITY_ROLES and not SLOT_RX.fullmatch(lit) and self._is_span_text(lit):
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
            numbers: bool = True) -> SlotTable:
    """The slot table of a question: $E1.. its entity spans (from the index), $N1.. its numbers by position,
    $R1.. the referents the context graph resolved ({'it': 'Siege of Vienna'})."""
    t = SlotTable(question)
    ents = index.spans(question) if index is not None else []
    for k, (text, s, e) in enumerate(ents, 1):
        t.spans.append(Span(f"$E{k}", text, s, e, "E"))
    if numbers:
        taken = [(s.start, s.end) for s in t.spans]
        k = 0
        for m in NUM_IN_TEXT_RX.finditer(question):
            if any(a <= m.start() < b for a, b in taken):
                continue
            k += 1
            t.spans.append(Span(f"$N{k}", m.group(0).replace(",", ""), m.start(), m.end(), "N"))
    for k, (word, target) in enumerate((referents or {}).items(), 1):
        t.spans.append(Span(f"$R{k}", str(target), -1, -1, "R"))
    return t


def to_slots(prompt: str, program: str, index: SpanIndex | None = None) -> tuple[str, str, SlotTable, dict]:
    """Rewrite one training record into the slot form: the prompt annotated, the program's copied literals
    replaced by their slots. Returns (annotated prompt, slotted program, table, stats). The index may be
    None: then the program's own entity literals that occur in the prompt are the spans (the gold seed is
    known), which is how the existing harvest converts without a name index."""
    if index is None:
        lits = [m.group("lit") for m in BIND_LIT_RX.finditer(program) if m.group("role").upper() in ENTITY_ROLES]
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
        if m.group("role").upper() not in ENTITY_ROLES:
            continue
        s = next((s for s in table.spans if s.kind == "E" and _norm(s.text) == _norm(lit)), None)
        if s is None:
            kept += 1
            continue
        out = out.replace(f'"{lit}"', f'"{s.id}"', 1); replaced += 1
    nums = 0
    for m in list(NUM_LIT_RX.finditer(program)):
        s = next((s for s in table.spans if s.kind == "N" and s.text == m.group("num")), None)
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
