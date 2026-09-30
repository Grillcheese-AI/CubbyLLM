"""arith_world -- arithmetic as a world: the emitter asks it for the constants and the solved plans it needs.

Wired: STANDALONE (2026-09-30; `emitter_data.slot_record(world=...)` and the curriculum read it; the serve path
does not yet).

The owner, 2026-09-30: "arithmetic is a world... it can go into the world and query it to get the right tools /
data to solve its problem. If it's not a world then it should be." What it holds -- knowledge and laws, never
state (the worlds rule) -- and what a query returns, symbols and scores only (the symbolic boundary):

  conversion facts   "60 minutes per hour", "16 ounces per pound", ... a closed list, each offered when every
                     unit it links appears in the question. The host places them as `$K` slots
                     (`slots.extract(constants=...)`, annotated `[K1 = 60 minutes per hour]`), so the emitter
                     never recalls a factor from its weights: 10% of v12e's leftover constants were these,
                     and a conversion the world supplied is exact and on the ledger.
  procedures         solved problems: the question's content words (numbers masked) and the SLOTTED program
                     the VM verified for it. `recall` returns (procedure id, similarity); `propose` rebinds a
                     remembered program to the new question's slots by position -- analogical transfer, a plan
                     candidate that goes through the slot check and the VM like any emitter program (memory
                     proposes; the VM disposes). A procedure is LATENT until attested (the VM hit a known
                     answer: a gold, or a user's confirmation) and only attested ones are recalled by default.
                     This is how a mistake, once solved, helps on the very next question -- before any retrain.

The codes are the hippocampus's (`hippocampus.encode`: SimHash over a bag of content words, 256 bits, Hamming
recall) so an arithmetic procedure and a chain episode are remembered the same way.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .hippocampus import N_BITS, content_words, encode, hamming
from .slots import NUM_IN_TEXT_RX, NUM_WORD_RX, SLOT_RX

__wiring__ = Wiring.STANDALONE

WORLD_ID = "arithmetic"


@dataclass(frozen=True)
class ConversionFact:
    value: float
    text: str                   # "60 minutes per hour"
    units: tuple[str, ...]      # every one must appear in the question


# unit word -> the pattern that finds it (plurals, abbreviations, the adjective forms)
UNIT_RX = {
    "second": r"seconds?|secs?", "minute": r"minutes?|mins?", "hour": r"hours?|hourly|hrs?", "day": r"days?|daily",
    "week": r"weeks?|weekly", "month": r"months?|monthly", "year": r"years?|yearly|annual(?:ly)?",
    "decade": r"decades?", "century": r"century|centuries", "cent": r"cents?", "dollar": r"dollars?|\$",
    "nickel": r"nickels?", "dime": r"dimes?", "quarter": r"quarters?", "percent": r"percent|%",
    "centimeter": r"centimet(?:er|re)s?|cm", "meter": r"met(?:er|re)s?", "kilometer": r"kilomet(?:er|re)s?|km",
    "gram": r"grams?", "kilogram": r"kilograms?|kilos?|kg", "milliliter": r"millilit(?:er|re)s?|ml",
    "liter": r"lit(?:er|re)s?", "ounce": r"ounces?|oz", "pound": r"pounds?|lbs?", "ton": r"tons?",
    "inch": r"inch(?:es)?", "foot": r"foot|feet", "yard": r"yards?", "mile": r"miles?", "quart": r"quarts?",
    "gallon": r"gallons?", "pint": r"pints?", "cup": r"cups?",
}
_UNIT_COMPILED = {u: re.compile(rf"(?<![\w]){p}(?![\w])" if p != r"dollars?|\$" else r"\bdollars?\b|\$", re.I)
                  for u, p in UNIT_RX.items()}

CONVERSIONS: tuple[ConversionFact, ...] = tuple(ConversionFact(float(v), t, u) for v, t, u in (
    (60, "60 seconds per minute", ("second", "minute")), (60, "60 minutes per hour", ("minute", "hour")),
    (3600, "3600 seconds per hour", ("second", "hour")), (24, "24 hours per day", ("hour", "day")),
    (7, "7 days per week", ("day", "week")), (52, "52 weeks per year", ("week", "year")),
    (12, "12 months per year", ("month", "year")), (365, "365 days per year", ("day", "year")),
    (30, "30 days per month", ("day", "month")), (4, "4 weeks per month", ("week", "month")),
    (10, "10 years per decade", ("year", "decade")), (100, "100 years per century", ("year", "century")),
    (100, "100 cents per dollar", ("cent", "dollar")), (5, "5 cents per nickel", ("cent", "nickel")),
    (10, "10 cents per dime", ("cent", "dime")), (25, "25 cents per quarter", ("cent", "quarter")),
    (100, "100 percent in the whole", ("percent",)),
    (100, "100 centimeters per meter", ("centimeter", "meter")), (1000, "1000 meters per kilometer", ("meter", "kilometer")),
    (1000, "1000 grams per kilogram", ("gram", "kilogram")), (1000, "1000 milliliters per liter", ("milliliter", "liter")),
    (16, "16 ounces per pound", ("ounce", "pound")), (2000, "2000 pounds per ton", ("pound", "ton")),
    (12, "12 inches per foot", ("inch", "foot")), (3, "3 feet per yard", ("foot", "yard")),
    (36, "36 inches per yard", ("inch", "yard")), (5280, "5280 feet per mile", ("foot", "mile")),
    (4, "4 quarts per gallon", ("quart", "gallon")), (2, "2 pints per quart", ("pint", "quart")),
    (8, "8 pints per gallon", ("pint", "gallon")), (16, "16 cups per gallon", ("cup", "gallon")),
    (4, "4 cups per quart", ("cup", "quart")), (8, "8 ounces per cup", ("ounce", "cup")),
    (32, "32 ounces per quart", ("ounce", "quart")), (128, "128 ounces per gallon", ("ounce", "gallon")),
))


def units_in(question: str) -> set[str]:
    return {u for u, rx in _UNIT_COMPILED.items() if rx.search(question)}


def conversions(question: str) -> list[ConversionFact]:
    """The conversion facts whose every unit the question mentions, in the table's order."""
    have = units_in(question)
    return [f for f in CONVERSIONS if all(u in have for u in f.units)]


_ANNOT_RX = re.compile(r"\[(?:[ENRK]\d+)(?::| =)[^\]]*\]")


def procedure_words(question: str) -> list[str]:
    """The words that say WHAT to compute, with every number and slot mark taken out: two questions that
    differ only in their numbers land on one code."""
    q = _ANNOT_RX.sub(" ", question)
    q = NUM_IN_TEXT_RX.sub(" ", q)
    q = NUM_WORD_RX.sub(" ", q)
    return content_words(q)


@dataclass
class Procedure:
    id: str
    question: str               # as it was asked (annotated or plain)
    program: str                # SLOTTED: $N / $K by position, never the numbers
    n_slots: int                # how many distinct $N it names
    attested: bool = False      # the VM hit a known answer (gold / user) -- latent until then
    source: str = ""            # "gold" | "user" | "curriculum" | ...
    code: int = 0
    uses: int = 0               # recalls that ended in a verified answer
    written_at: float = 0.0


@dataclass
class ArithmeticWorld:
    """One world, two kinds of member: the conversion facts (closed, exact) and the solved procedures."""
    procedures: list[Procedure] = field(default_factory=list)
    word_bits: object = None

    world_id: str = WORLD_ID

    # -- the data a question needs ------------------------------------------------
    def constants(self, question: str) -> list[tuple[str, float]]:
        """(text, value) for `slots.extract(constants=...)`: the host places them as $K slots."""
        return [(f.text, f.value) for f in conversions(question)]

    # -- the procedures -----------------------------------------------------------
    def remember(self, question: str, program: str, *, attested: bool, source: str) -> Procedure:
        """Keep a solved problem. `program` must be slotted (a number the question states is a $N)."""
        if "$N" not in program:
            raise ValueError("remember a SLOTTED program: numbers as $N, so it transfers to other numbers")
        code = encode(procedure_words(question), self.word_bits)
        n = len({m.group(0) for m in SLOT_RX.finditer(program) if m.group("kind") == "N"})
        pid = f"proc{len(self.procedures) + 1}"
        p = Procedure(pid, question, program, n, attested, source, code, 0, time.time())
        self.procedures.append(p)
        return p

    def attest(self, pid: str, source: str) -> None:
        p = self.get(pid)
        p.attested, p.source = True, source

    def get(self, pid: str) -> Procedure:
        return next(p for p in self.procedures if p.id == pid)

    def recall(self, question: str, k: int = 3, latent: bool = False, max_distance: int = N_BITS // 4
               ) -> list[tuple[str, float]]:
        """(procedure id, similarity 0..1) nearest first; attested only unless `latent`. Symbols and scores."""
        cue = encode(procedure_words(question), self.word_bits)
        if not cue:
            return []
        hits = []
        for p in self.procedures:
            if not (p.attested or latent) or not p.code:
                continue
            d = hamming(cue, p.code)
            if d <= max_distance:
                hits.append((p.id, 1.0 - d / N_BITS))
        return sorted(hits, key=lambda h: -h[1])[:k]

    def propose(self, question: str, n_slots: int, k: int = 3) -> list[tuple[str, str, float]]:
        """(procedure id, slotted program, similarity): remembered programs whose $N count fits the new
        question's, to be checked, filled and run exactly like an emitter's. The rebinding is by position."""
        out = []
        for pid, sim in self.recall(question, k=k * 3):
            p = self.get(pid)
            if p.n_slots <= n_slots:
                out.append((pid, p.program, sim))
            if len(out) == k:
                break
        return out

    def used(self, pid: str) -> None:
        self.get(pid).uses += 1

    # -- persistence ----------------------------------------------------------------
    def save(self, path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for q in self.procedures:
                f.write(json.dumps({**q.__dict__, "code": format(q.code, "x")}) + "\n")

    @classmethod
    def load(cls, path, word_bits=None) -> "ArithmeticWorld":
        w = cls(word_bits=word_bits)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            d["code"] = int(d["code"], 16)
            w.procedures.append(Procedure(**d))
        return w
