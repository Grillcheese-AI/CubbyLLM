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
`citations` keeps each fact's witnesses for the record (the audit view, order 0016); `provenance` their
names for the store. A fact's era is its books' (`citations[...]["era"]`).

Wired: WIRED (`ask.AskLoop(books=)`, `serve_api --books`).
"""
from __future__ import annotations

import re

import book_frames as bf
from books import BookIndex, norm
from cubbyllm.reasoning.planner import Triple, normalize

__wiring__ = "WIRED"

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

    def __init__(self, index: BookIndex | str | None = None, k: int = 20) -> None:
        self.index = index if isinstance(index, BookIndex) else BookIndex(index)
        self.k = int(k)
        self.provenance: dict[str, str] = {}
        self.held: set[str] = set()
        self.attributed: set[str] = set()
        self.citations: dict[str, list[dict]] = {}
        self.times: dict[str, dict] = {}                         # the dates a frame read beside a fact (the loop's `times`)
        self.last: dict = {}

    def facts(self, entity: str, relations=None, via=None) -> list[Triple]:
        want = [r for r in (relations or []) if r in bf.CUES]
        cues = [c for r in (want or list(bf.CUES)) for c in bf.CUES[r]]
        reads: dict[tuple[str, str], dict] = {}
        passages = self.index.passages(entity, words=cues, k=self.k)
        for p in passages:
            for sent in bf.sentences(p.text):
                for t, sentence, when in bf.read(sent, entity):
                    if want and t.rel not in want:
                        continue
                    r = reads.setdefault((t.rel, normalize(t.obj)), {"triple": t, "w": [], "when": when})
                    r["when"] = r["when"] or when
                    if not any(w["book"] == p.book and w["sentence"] == sentence for w in r["w"]):
                        r["w"].append({"book": p.book, "author": p.author, "title": p.title, "era": p.era,
                                       "offset": p.offset, "sentence": sentence})
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
        self.last = {"entity": entity, "how": f"{len(passages)} passages, {len(out)} facts", "needs": want or None}
        return out
