"""build_el -- the entity-linking sets of order 0046 into the one schema (``common``): WebQSP_EL and GraphQ_EL (ELQ's
question entity linking data: spans and Wikidata ids), ShadowLink (Top, Shadow, Neutral, Tail: a snippet, a span, a
Wikipedia title mapped to its Wikidata id), AmbER (QA queries over names that several Wikidata items share; head and
tail items).

ShadowLink and AmbER have no split: they are cut in half by seed 0 (AmbER by its sets, so every query on one name
lands on one side).

Wired: STANDALONE (a data builder; ``build_sets.py`` runs it).
"""
from __future__ import annotations

import json
import os

from .common import halves, raw_dir, wikipedia_qids

__wiring__ = "STANDALONE"

EL = {"webqsp_el": "WebQSP_EL", "graphq_el": "graphquestions_EL"}


def el4qa(name: str) -> list[dict]:
    rows = []
    folder = os.path.join(raw_dir("el4qa"), "EL4QA_data", EL[name])
    for split in ("dev", "test"):
        for line in open(os.path.join(folder, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            ms = []
            for (s, e), q in zip(r.get("mentions") or [], r.get("wikidata_id") or []):
                if q:
                    ms.append({"qid": q, "text": r["text"][s:e], "span": [s, e]})
            rows.append({"id": f"{name}:{r['id']}", "set": name, "lang": "en", "question": r["text"], "mentions": ms,
                         "prop": None, "answers": [], "answer_qids": [], "split": split, "extra": {}})
    return rows


def shadowlink(src) -> list[dict]:
    raw, titles = [], []
    for sub in ("Top", "Shadow", "Neutral", "Tail"):                       # (Neutral carries no gold entity: its rows
        for i, r in enumerate(json.load(open(os.path.join(raw_dir("shadowlink"), f"{sub}.json"), encoding="utf-8"))):   # have none)
            raw.append((sub, i, r))
            titles.append(r.get("entity_name"))
    qid = wikipedia_qids(titles, "en")
    rows = []
    for sub, i, r in raw:
        s, n = (r.get("span") or [0, 0])[:2]
        text = (r.get("example") or "")[s:s + n]
        q = qid.get(r.get("entity_name"))
        if sub == "Neutral":
            continue                                    # neutral contexts name no entity: nothing to find or choose
        rows.append({"id": f"shadowlink:{sub}-{i}", "set": "shadowlink", "lang": "en", "question": r.get("example") or "",
                     "mentions": [{"qid": q, "text": text, "span": [s, s + n]}] if q else [],
                     "prop": None, "answers": [], "answer_qids": [], "split": None,
                     "extra": {"subset": sub, "title": r.get("entity_name"), "space": r.get("entity_space_name"),
                               "mapped": bool(q)}})
    side = halves([r["id"] for r in rows])
    for r in rows:
        r["split"] = side[r["id"]]
    return rows


def amber() -> list[dict]:
    rows, names = [], []
    for kind in ("human", "nonhuman"):
        for line in open(os.path.join(raw_dir("amber"), f"{kind}_qa_amber_sets.jsonl"), encoding="utf-8"):
            s = json.loads(line)
            names.append(s["name"])
            for q, item in (s.get("qids") or {}).items():
                for x in item.get("queries") or []:
                    rows.append({"id": f"amber:{x['id']}", "set": "amber", "lang": "en", "question": x["input"],
                                 "mentions": [{"qid": q, "text": s["name"], "span": None}],
                                 "prop": (x.get("meta") or {}).get("pid"),
                                 "answers": list((x.get("output") or {}).get("answer") or []), "answer_qids": [], "split": None,
                                 "extra": {"kind": kind, "name": s["name"], "head": bool(item.get("is_head")),
                                           "n_items": len(s.get("qids") or {})}})
    side = halves(sorted(set(names)))
    for r in rows:
        r["split"] = side[r["extra"]["name"]]
    return rows


__all__ = ["amber", "el4qa", "shadowlink"]
