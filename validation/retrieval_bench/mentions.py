"""mentions -- the gold items' names from Wikidata (by id, through the source's cache), and the mentions located in the
question for the sets that do not mark them (Mintaka's French, QALD-9-plus, SimpleQuestions-Wikidata, CronQuestions).

A mention is located as the longest of the gold item's own names (its label and aliases in the row's language, then
`mul`, then English) found in the question as whole words, folded (``common.fold``). This is the gold-mention setting:
R1 measures finding, not reading, so the words are taken where the gold item's names put them, and a mention no name
of its item spells is reported apart ("not located"), never guessed.

Wired: STANDALONE (order 0046's harness).
"""
from __future__ import annotations

import json
import os
import re

from retrieval_sets.common import data_root, entities, fold, names_of

__wiring__ = "STANDALONE"


class Gold:
    """The gold items' entity json, fetched once per item and kept off-repo (``gold_entities.jsonl``)."""

    def __init__(self, src) -> None:
        self.src, self.path = src, os.path.join(data_root(), "gold_entities.jsonl")
        self.ents: dict = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                q, e = json.loads(line)
                self.ents[q] = e

    def fetch(self, qids) -> None:
        need = [q for q in dict.fromkeys(qids) if q and q not in self.ents]
        got = entities(self.src, need) if need else {}
        with open(self.path, "a", encoding="utf-8") as f:
            for q in need:
                e = got.get(q) or {}
                slim = {"labels": e.get("labels") or {}, "aliases": e.get("aliases") or {},
                        "sitelinks": {k: {"title": v.get("title")} for k, v in (e.get("sitelinks") or {}).items()}}
                self.ents[q] = slim
                f.write(json.dumps([q, slim], ensure_ascii=False) + "\n")

    def names(self, qid: str, lang: str) -> list[str]:
        e = self.ents.get(qid) or {}
        out = names_of(e, lang)
        if lang != "en":
            out += [n for n in names_of(e, "en") if n not in out]
        return out

    def title(self, qid: str, site: str) -> str | None:
        return ((self.ents.get(qid) or {}).get("sitelinks") or {}).get(site, {}).get("title")


def locate(question: str, names: list[str]) -> tuple[str, list[int]] | None:
    """(the words as written, [start, end]) for the longest name found in the question as whole words, or None."""
    fq = fold(question)
    best = None
    for n in names:
        fn = fold(n)
        if not fn:
            continue
        m = re.search(r"(?<!\w)" + re.escape(fn) + r"(?!\w)", fq)
        if m and (best is None or len(fn) > len(best[0])):
            best = (fn, m.start(), m.end())
    if best is None:
        return None
    # map the folded span back onto the question: fold each prefix until it reaches the folded offsets
    start = _back(question, best[1])
    end = _back(question, best[2], start)
    return question[start:end].strip(" ,;:.?!'\"()"), [start, end]


def _back(question: str, folded_offset: int, lo: int = 0) -> int:
    for i in range(lo, len(question) + 1):
        if len(fold(question[:i] + "x")) - 1 >= folded_offset:
            return i
    return len(question)


def fill(rows: list[dict], gold: Gold) -> dict:
    """Mentions without words located in place; returns {"located", "not_located", "had_text"} counts."""
    gold.fetch(m["qid"] for r in rows for m in r["mentions"] if not m.get("text"))
    c = {"had_text": 0, "located": 0, "not_located": 0}
    for r in rows:
        for m in r["mentions"]:
            if m.get("text"):
                c["had_text"] += 1
                continue
            got = locate(r["question"], gold.names(m["qid"], r["lang"]))
            if got:
                m["text"], m["span"], m["located"] = got[0], got[1], True
                c["located"] += 1
            else:
                m["located"] = False
                c["not_located"] += 1
    return c


__all__ = ["Gold", "fill", "locate"]
