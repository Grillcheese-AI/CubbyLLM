"""validation/fact_store/audit_sets.py -- order 0055 step 0: the audit's 1,000 questions.

  simpleqa  400 of the 4,326 (SimpleQA's test set, as 0054 loads it); no ids: the asked item is found by name
  mintaka   300 of Mintaka's dev split (0046's rows; test is left alone), each with its French; the asked items and
            the answer items are the set's own ids
  mkqa      300 of MKQA's French questions whose answer a statement could hold (entity, date, number, number with
            unit, short phrase; long answers, yes/no and unanswerable left out, counted); no ids for the asked
            item: found by name in French. The answer's Q-id where MKQA gives one, its English text otherwise.
            MKQA's rows stay off the repo (0048): only counts reach a committed log.
Each draw is seeded and sorted by id, so a rerun draws the same questions. A row: {id, set, lang, question,
question_fr, asked (Q-ids or None), golds [{qids, text}], extra}.

Wired: STANDALONE (order 0055's audit).
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import random

__wiring__ = "STANDALONE"

SIZES = {"simpleqa": 400, "mintaka": 300, "mkqa": 300}
MKQA_TYPES = ("entity", "date", "number", "number_with_unit", "short_phrase")


def draw(rows: list[dict], n: int, name: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["id"])
    if len(rows) <= n:
        return rows
    return sorted(random.Random(f"0055:{name}").sample(rows, n), key=lambda r: r["id"])


def simpleqa() -> list[dict]:
    from system_eval.bench import load
    return [{"id": r["id"], "set": "simpleqa", "lang": "en", "question": r["question"], "question_fr": None,
             "asked": None, "golds": [{"qids": [], "text": g} for g in r["golds"]], "extra": {}}
            for r in load("simpleqa")]


def mintaka() -> list[dict]:
    from retrieval_sets.common import read_rows
    by = collections.defaultdict(dict)
    for r in read_rows("mintaka", split="dev"):
        by[r["id"].rsplit(":", 1)[0]][r["lang"]] = r
    out = []
    for base, langs in by.items():
        en = langs.get("en")
        if not en:
            continue
        golds = ([{"qids": list(en["answer_qids"]), "text": ""}] if en["answer_qids"] else []) + \
                [{"qids": [], "text": str(a)} for a in en["answers"]]
        out.append({"id": base, "set": "mintaka", "lang": "en", "question": en["question"],
                    "question_fr": (langs.get("fr") or {}).get("question"),
                    "asked": [m["qid"] for m in en["mentions"]] or None, "golds": golds,
                    "extra": {"complexity": en["extra"].get("complexity"), "category": en["extra"].get("category")}})
    return out


def mkqa(counts: dict | None = None) -> list[dict]:
    from retrieval_sets.common import raw_dir
    out, c = [], collections.Counter()
    for line in gzip.open(os.path.join(raw_dir("mkqa"), "mkqa.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(line)
        q, fr, en = (r.get("queries") or {}).get("fr"), (r.get("answers") or {}).get("fr") or [], \
            (r.get("answers") or {}).get("en") or []
        kinds = {a.get("type") for a in fr}
        if not q or not fr or not kinds <= set(MKQA_TYPES):
            c["left_out:" + ",".join(sorted(kinds - set(MKQA_TYPES)) or ["no question"])] += 1
            continue
        golds = []
        for i, a in enumerate(fr):
            e = en[i] if i < len(en) else a
            golds.append({"qids": [a["entity"]] if str(a.get("entity") or "").startswith("Q") else [],
                          "text": "" if a.get("entity") else str(e.get("text") or a.get("text") or "")})
        c["kept"] += 1
        out.append({"id": f"mkqa:{r['example_id']}", "set": "mkqa", "lang": "fr", "question": q, "question_fr": q,
                    "asked": None, "golds": golds, "extra": {"types": sorted(kinds)}})
    if counts is not None:
        counts.update(c)
    return out


def questions(counts: dict | None = None) -> list[dict]:
    rows = []
    for name, fn in (("simpleqa", simpleqa), ("mintaka", mintaka), ("mkqa", lambda: mkqa(counts))):
        rows += draw(fn(), SIZES[name], name)
    return rows


__all__ = ["MKQA_TYPES", "SIZES", "draw", "mintaka", "mkqa", "questions", "simpleqa"]
