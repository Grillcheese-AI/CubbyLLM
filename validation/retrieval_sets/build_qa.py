"""build_qa -- the question-answering sets of order 0046 into the one schema (``common``): Mintaka (EN and its FR
translation), QALD-9-plus (EN, FR; its Wikidata SPARQL), SimpleQuestions-Wikidata (the answerable files), CronQuestions
(its paraphrases, which spell the names; read through a restricted unpickler that builds nothing but plain containers).

A mention without its words in the question (Mintaka's French, QALD, SimpleQuestions, CronQuestions) is located later
by the gold item's own labels (``retrieval_bench.mentions``); here it carries its item only.

Wired: STANDALONE (a data builder; ``build_sets.py`` runs it).
"""
from __future__ import annotations

import json
import os
import pickle
import re

from .common import raw_dir

__wiring__ = "STANDALONE"

_Q = re.compile(r"(?:entity/|wd:)(Q\d+)")
_P = re.compile(r"(?:prop/direct/|wdt:|p:|ps:)(P\d+)")


def _qid(x) -> str | None:
    s = str(x or "")
    m = re.search(r"(Q\d+)$", s)
    return m.group(1) if m else None


def mintaka() -> list[dict]:
    rows = []
    for split in ("dev", "test"):
        for r in json.load(open(os.path.join(raw_dir("mintaka"), f"mintaka_{split}.json"), encoding="utf-8")):
            ents = [e for e in r.get("questionEntity") or [] if e.get("entityType") == "entity" and _qid(e.get("name"))]
            ans = (r.get("answer") or {}).get("answer") or []
            aq = [a["name"] for a in ans if isinstance(a, dict) and _qid(a.get("name"))]
            for lang in ("en", "fr"):
                q = r["question"] if lang == "en" else (r.get("translations") or {}).get("fr")
                if not q:
                    continue
                labels = [((a.get("label") or {}).get(lang) or (a.get("label") or {}).get("en")) if isinstance(a, dict) else str(a) for a in ans]
                rows.append({"id": f"mintaka:{r['id']}:{lang}", "set": "mintaka", "lang": lang, "question": q,
                             "mentions": [{"qid": e["name"], "text": e.get("mention") if lang == "en" else None,
                                           "span": e.get("span") if lang == "en" else None} for e in ents],
                             "prop": None, "answers": [x for x in labels if x], "answer_qids": aq, "split": split,
                             "extra": {"category": r.get("category"), "complexity": r.get("complexityType")}})
    return rows


def qald9plus() -> list[dict]:
    rows = []
    for native, split in (("train", "dev"), ("test", "test")):
        d = json.load(open(os.path.join(raw_dir("qald9plus"), f"qald_9_plus_{native}_wikidata.json"), encoding="utf-8"))
        for r in d["questions"]:
            sparql = (r.get("query") or {}).get("sparql") or ""
            qids = list(dict.fromkeys(_Q.findall(sparql)))
            props = list(dict.fromkeys(_P.findall(sparql)))
            aq, lits = [], []
            for a in r.get("answers") or []:
                for b in (a.get("results") or {}).get("bindings") or []:
                    for v in b.values():
                        q = _qid(v.get("value")) if v.get("type") == "uri" else None
                        (aq if q else lits).append(q or str(v.get("value")))
                if "boolean" in a:
                    lits.append(str(a["boolean"]).lower())
            for lang in ("en", "fr"):
                q = next((x["string"] for x in r.get("question") or [] if x.get("language") == lang and x.get("string")), None)
                if q:
                    rows.append({"id": f"qald9plus:{native}-{r['id']}:{lang}", "set": "qald9plus", "lang": lang, "question": q,
                                 "mentions": [{"qid": x, "text": None, "span": None} for x in qids],
                                 "prop": props[0] if len(props) == 1 else None, "answers": lits, "answer_qids": aq,
                                 "split": split, "extra": {"props": props}})
    return rows


def simplequestions_wd() -> list[dict]:
    rows = []
    for native, split in (("valid", "dev"), ("test", "test")):
        path = os.path.join(raw_dir("simplequestions_wd"), f"annotated_wd_data_{native}_answerable.txt")
        for i, line in enumerate(open(path, encoding="utf-8")):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            s, p, o, q = parts[:4]
            rows.append({"id": f"simplequestions_wd:{native}-{i}", "set": "simplequestions_wd", "lang": "en", "question": q,
                         "mentions": [{"qid": s, "text": None, "span": None}],
                         "prop": "P" + p[1:] if p[:1] in "PR" else None, "answers": [], "answer_qids": [o] if o.startswith("Q") else [],
                         "split": split, "extra": {"inverse": p.startswith("R")}})
    return rows


class _Plain(pickle.Unpickler):
    """Builds plain containers and scalars only; anything else in the stream is refused (a pickle can run code)."""
    OK = {"dict", "list", "set", "tuple", "str", "int", "float", "frozenset", "bool"}

    def find_class(self, module, name):
        if module == "builtins" and name in self.OK:
            return getattr(__import__("builtins"), name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def cronquestions() -> list[dict]:
    rows = []
    base = os.path.join(raw_dir("cronquestions"), "data", "wikidata_big", "questions")
    for native, split in (("valid", "dev"), ("test", "test")):
        with open(os.path.join(base, f"{native}.pickle"), "rb") as f:
            data = _Plain(f).load()
        for r in data:
            q = (r.get("paraphrases") or [None])[0] or r.get("question")
            ents = sorted(r.get("entities") or [])
            ans = sorted(r.get("answers") or [], key=str)
            rows.append({"id": f"cronquestions:{native}-{r.get('uniq_id')}", "set": "cronquestions", "lang": "en", "question": q,
                         "mentions": [{"qid": e, "text": None, "span": None} for e in ents if str(e).startswith("Q")],
                         "prop": next(iter(sorted(r.get("relations") or [])), None),
                         "answers": [str(a) for a in ans if not str(a).startswith("Q")],
                         "answer_qids": [a for a in ans if str(a).startswith("Q")], "split": split,
                         "extra": {"type": r.get("type"), "answer_type": r.get("answer_type"), "times": sorted(r.get("times") or [])}})
    return rows


__all__ = ["cronquestions", "mintaka", "qald9plus", "simplequestions_wd"]
