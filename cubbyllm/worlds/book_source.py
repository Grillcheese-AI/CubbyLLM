"""standin/book_source.py -- the books as a `cubbyllm.reasoning.learn.Source` (order 0012): for the entity a
walk stalled on, the passages about it (`books.BookIndex.passages`, ranked by the cues of the relations the
walk needs), the facts frames read off their sentences (`book_frames.read`), and each fact's witnesses --
the book, its author, its era, the offset, the sentence.

Tiers (Nick's rule):
- one author states it: ATTRIBUTED (`attributed`; the loop stores it as such, asks the sources below it
  for a second statement, and speaks the answer hedged when none comes);
- two independent authors (different credited people, none shared) state it: CORROBORATED, plain; a book
  and Wikidata corroborate in the loop, when Wikidata restates the fact;
- two values for a relation that has one (a birth date, a founder): a CONTRADICTION -- both returned,
  both `held`, nothing spoken flat.
A book whose file credits no author (the events file, an anonymous folder) attributes, never corroborates.
The same-name guard (Cowork's review): a book's person counts for the subject only if the book fits the subject's
life as the world knows it (`life(entity)` -> (born, died) years): no birth or death the book states for that name
falls outside it -- the Mayflower's Samuel Fuller is not the filmmaker. The book's era is not a test: an era is the
book's subject, not its people's (the ancient-history books cite modern scholars; on the bench an era test set aside
Marie Tharp's and three others' true birth years, 2026-10-22).
`citations` keeps each fact's witnesses for the record (the audit view, order 0016); `provenance` their
names for the store. A fact's era is its books' (`citations[...]["era"]`).

Wired: WIRED (`ask.AskLoop(books=)`, `serve_api --books`).
"""
from __future__ import annotations

import re

from . import book_frames as bf
from .books import BookIndex, norm
from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

_FORMS_ONE = []


def _FORMS():
    if not _FORMS_ONE:
        from question_forms import Forms
        _FORMS_ONE.append(Forms())
    return _FORMS_ONE[0]


SPAN = 110                                                  # the longest life the guard allows


def year(v: str) -> int | None:
    m = re.match(r"(-?\d{1,4})", v or "")
    return None if m is None else -int(m.group(1)) if " BC" in v else int(m.group(1))


def window(life) -> tuple[int, int] | None:
    """(earliest birth, latest death) the world's dates allow, or None when it knows neither."""
    born, died = life or (None, None)
    if born is None and died is None:
        return None
    return (born if born is not None else died - SPAN, died if died is not None else born + SPAN)


def fits(life, era: str, reads) -> bool:
    """Does a book's person fit the subject's life: every birth or death year it states for the name lies inside it?
    (`era` is kept for the callers; it does not decide.)"""
    w = window(life)
    if w is None:
        return True
    return all(w[0] <= y <= w[1] for rel, v in reads if rel in ("date of birth", "date of death") for y in [year(v)] if y is not None)


FUNCTIONAL = {"date of birth", "place of birth", "date of death", "place of death", "place of burial", "inception",
              "founded by", "point in time", "reign"}


def credited(author: str) -> frozenset:
    """The people a file name credits: 'A, B & C', 'A and B', 'Edited by A' -> {a, b, c} (normalized)."""
    a = re.sub(r"\b(?:ed(?:ited)?\.?\s+by|eds?\.)\s*", "", author or "", flags=re.I)
    return frozenset(n for n in (norm(x) for x in re.split(r",|&|;|\band\b", a)) if n and len(n) > 2)


def independent(witnesses: list[dict]) -> bool:
    """Two witnesses whose credited authors are known and share nobody."""
    sets = list(dict.fromkeys(credited(w["author"]) for w in witnesses if credited(w["author"])))
    return any(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])


class BookSource:
    """`facts(entity, relations=None, via=None)` -> [Triple]: what the frames read about the entity in the
    `k` best passages; `provenance`, `held`, `attributed` and `citations` filled while it answers."""
    name = "books"

    def __init__(self, index: BookIndex | str | None = None, k: int = 20, life=None) -> None:
        self.index = index if isinstance(index, BookIndex) else BookIndex(index)
        self.life = life                                         # entity -> (born, died) years: the same-name guard
        self.unfit: dict[str, int] = {}                          # entity -> books the guard set aside
        self.k = int(k)
        self.provenance: dict[str, str] = {}
        self.held: set[str] = set()
        self.attributed: set[str] = set()
        self.citations: dict[str, list[dict]] = {}
        self.times: dict[str, dict] = {}                         # the dates a frame read beside a fact (the loop's `times`)
        self.last: dict = {}

    @staticmethod
    def labels(relations) -> list[str]:
        """The frames' labels a walk's wording asks for: the label itself, or the labels whose cue words or question
        forms hold the wording ("born" -> date and place of birth; the loop narrows by its when / where)."""
        out = []
        for r in relations or []:
            k = normalize(r)
            got = [k] if k in bf.CUES else [lab for lab, cues in bf.CUES.items() if k in cues] or                 [lab for lab in _FORMS().relations(k) if lab in bf.CUES]
            out += [g for g in got if g not in out]
        return out

    def facts(self, entity: str, relations=None, via=None) -> list[Triple]:
        want = self.labels(relations)
        cues = [c for r in (want or list(bf.CUES)) for c in bf.CUES[r]]
        reads: dict[tuple[str, str], dict] = {}
        passages = self.index.passages(entity, words=cues, k=self.k)
        by_book: dict[int, list] = {}
        for p in passages:
            for sent in bf.sentences(p.text):
                by_book.setdefault(p.book, []).extend((p, *x) for x in bf.read(sent, entity))
        life = self.life(entity) if self.life is not None else None
        self.unfit[entity] = 0
        for book, rows in by_book.items():
            if not rows:                                         # its passages read nothing
                continue
            if not fits(life, rows[0][0].era, [(t.rel, t.obj) for _p, t, _s, _w in rows]):
                self.unfit[entity] += 1                          # another person of the same name
                continue
            for p, t, sentence, when in rows:
                if want and t.rel not in want:
                    continue
                r = reads.setdefault((t.rel, normalize(t.obj)), {"triple": t, "w": [], "when": when})
                r["when"] = r["when"] or when
                if not any(w["book"] == p.book and w["sentence"] == sentence for w in r["w"]):
                    r["w"].append({"book": p.book, "author": p.author, "title": p.title, "era": p.era,
                                   "offset": p.offset, "sentence": sentence, "when": when or {}})
        for (rel, o), r in list(reads.items()):          # a year agrees with the one full date of that year: one value
            if re.fullmatch(r"\d{1,4}", r["triple"].obj):
                full = [k for k, v in reads.items() if k[0] == rel and re.fullmatch(rf"0*{r['triple'].obj}-\d\d-\d\d", v["triple"].obj)]
                if len(full) == 1:
                    reads[full[0]]["w"] += [w for w in r["w"] if w not in reads[full[0]]["w"]]
                    del reads[(rel, o)]
        values: dict[str, int] = {}
        for rel, _o in reads:
            values[rel] = values.get(rel, 0) + 1
        out = []
        for (rel, _o), r in reads.items():
            t = r["triple"]
            k = normalize(f"{t.obj} is the {t.rel} of {t.subj}")
            self.citations[k] = r["w"]
            if r["when"]:
                self.times[k] = r["when"]
            self.provenance[k] = "+".join(dict.fromkeys(
                "book:" + ("&".join(sorted(credited(w["author"]))) or f"#{w['book']}") for w in r["w"]))
            self.held.discard(k); self.attributed.discard(k)
            if rel in FUNCTIONAL and values[rel] > 1:
                self.held.add(k)                                   # the books disagree: kept, never said flat
            elif not independent(r["w"]):
                self.attributed.add(k)
            out.append(t)
        self.last = {"entity": entity, "how": f"{len(passages)} passages, {len(out)} facts", "needs": want or None,
                     "books_set_aside": self.unfit[entity]}
        return out
