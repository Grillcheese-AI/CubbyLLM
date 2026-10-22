"""standin/book_reader.py -- the books' second reader (order 0013): the 450M with a reader LoRA, taught by
distant supervision (`standin/data/build_reader_sft.py`), reading one sentence for one subject.

The prompt and the answer, shared by the data builder, the adapter's eval and the bulk pass:

    Sentence: Napoleon was born in Ajaccio on 15 August 1769.
    Subject: Napoleon Bonaparte
    Facts:
 -> BORN_IN | Ajaccio
    BORN_ON | 15 August 1769

One line per fact the sentence states about the subject: a relation of the closed set (wikikg's codes, the ids
the wiki world's facts are made from) and the value copied from the sentence, or `NONE`. A line is kept only if
its relation is in the set and its value occurs in the sentence (`lfm_source.grounded`, the web reader's rule):
the reader may only point at text it was shown. A date's value becomes the store's form here (a full date ISO,
a year as written), so a book's fact and the wiki world's are one fact.

`frame_fact` brings frames' Wikidata labels ('place of birth') to the same codes, and `store_facts` writes any
(subject, code, value) as one sentence, whoever read it: frames' label where frames read the relation too, the
wiki world's noun otherwise.

Wired: STANDALONE (the bulk pass `validation/books_bulk_read.py` and the data builder; the loop reads its facts
from the store the bulk pass writes, never through the model).
"""
from __future__ import annotations

import re

import book_frames as bf
from cubbyllm.reasoning.planner import normalize

__wiring__ = "STANDALONE"

PROMPT = "Sentence: {sentence}\nSubject: {subject}\nFacts:"
NONE = "NONE"
# frames' labels -> wikikg's codes (the bench's map); a label with no code keeps its label
FRAME_CODE = {"date of birth": "BORN_ON", "place of birth": "BORN_IN", "date of death": "DIED_ON",
              "place of death": "DIED_IN", "founded by": "FOUNDED_BY", "location": "LOCATED_IN", "part of": "PART_OF",
              "place of burial": "BURIED_AT"}
_DATE = re.compile(bf.DATE)
_ISO = re.compile(r"-?\d{1,4}(?:-\d\d){0,2}")
DATE_CODES = {"BORN_ON", "DIED_ON", "inception", "point in time"}     # the codes whose wiki values are dates (>= 80%)


def prompt(sentence: str, subject: str) -> str:
    return PROMPT.format(sentence=" ".join(sentence.split()), subject=subject)


def answer(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f"{code} | {span}" for code, span in lines) or NONE


def is_date(value: str) -> bool:
    return bool(_ISO.fullmatch(value.strip()))


def date_value(span: str) -> str | None:
    """A date as the sentence writes it -> the store's form ('15 August 1769' -> 1769-08-15; '742' -> 742)."""
    m = _DATE.fullmatch(span.strip())
    return bf._date(m) if m else None


def parse(text: str, sentence: str, relations) -> list[tuple[str, str]]:
    """The model's lines -> [(code, span)]: the relation in the closed set, the span in the sentence; the rest
    dropped (an unknown relation is never mapped, a value the sentence lacks never kept)."""
    from lfm_source import grounded
    out = []
    for line in (text or "").splitlines():
        code, sep, span = line.strip().partition("|")
        code, span = code.strip(), span.strip()
        if not sep or code not in relations or not span or (code, span) in out:
            continue
        if span in sentence or grounded(span, sentence):
            out.append((code, span))
    return out


LABEL = {code: label for label, code in FRAME_CODE.items()}


def store_facts(subject: str, code: str, value: str) -> list[str]:
    """The store's sentences for (subject, code, value). A relation frames read too takes their Wikidata label, as
    `BookSource` writes it ('Ajaccio is the place of birth of Napoleon'): a bulk fact and a book fact learned live
    are one key, and Wikidata's statement corroborates both. Any other code takes the wiki world's noun, form and
    inverse reading (`wikikg.facts_for`); a label without a code reads 'value is the label of subject'."""
    import wikikg as wk
    if code in LABEL or not code.isupper():
        return [f"{value} is the {LABEL.get(code, code)} of {subject}"]
    return wk.facts_for(subject, code, value)


def frame_fact(triple) -> tuple[str, str]:
    """A frame's Triple -> (code or label, value)."""
    return FRAME_CODE.get(triple.rel, triple.rel), triple.obj


class BookReader:
    """`read(sentence, subject)` -> [(code, value)], values in the store's form. `generate(prompt) -> text` is the
    adapter's greedy decode (the 450M with the reader LoRA; on Colab, or a local trunk); reads are cached by text."""
    name = "book-reader"

    def __init__(self, generate, relations) -> None:
        self.generate, self.relations = generate, frozenset(relations)
        self.cache: dict[str, str] = {}
        self.calls = 0

    def read(self, sentence: str, subject: str) -> list[tuple[str, str]]:
        p = prompt(sentence, subject)
        if p not in self.cache:
            self.cache[p] = self.generate(p)
            self.calls += 1
        out = []
        for code, span in parse(self.cache[p], sentence, self.relations):
            v = date_value(span) if code in DATE_CODES else span
            if v:
                out.append((code, v))
        return out


def key(fact: str) -> str:
    return normalize(fact)
