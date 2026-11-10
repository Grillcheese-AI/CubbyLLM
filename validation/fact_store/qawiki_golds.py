"""validation/fact_store/qawiki_golds.py -- order 0055 (Cowork's addendum on merged items): the entity ids QAWiki's
questions use, by role, and how the hot store resolves them.

QAWiki v2026.3 (the WikiKGQA 2026 challenge's dataset; Zenodo 22661413, CC0): ``wikikgqa.json`` (the questions, an
extended QALD schema) and the test set's ``*_solutions.json``. Each question's ids are taken from its SPARQL
(``wd:Q..``), its mentions' linked entities and its gold answers' bindings (``http://www.wikidata.org/entity/Q..``).

    python validation/fact_store/qawiki_golds.py --qawiki <dir with the unpacked files> --out <ids.txt>
    python validation/fact_store/qawiki_golds.py --qawiki <dir> --resolved <wdstore resolve's output> --out <json>

The first writes the ids, one a line (for ``wdstore resolve <store> <ids.txt>``); the second counts, by role, the
ids found in the store as they are, found only through a redirect, and not found: the addendum's "how many benchmark
gold ids resolve only through a redirect".

Wired: STANDALONE (order 0055's report; read once, offline).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

__wiring__ = "STANDALONE"

WD = re.compile(r"(?:wd:|http://www\.wikidata\.org/entity/)(Q[1-9][0-9]*)\b")


def questions(d: str):
    """(set, question) for every question of the training file and the test set's solutions."""
    for path in [os.path.join(d, "wikikgqa.json")] + sorted(glob.glob(os.path.join(d, "*_solutions.json"))):
        name = os.path.basename(path).replace(".json", "")
        with open(path, encoding="utf-8") as f:
            for q in json.load(f)["questions"]:
                yield name, q


def ids(q: dict) -> dict[str, set[str]]:
    """The question's entity ids by role: in its query, its mentions, its gold answers."""
    out = {"query": set(WD.findall((q.get("query") or {}).get("sparql") or "")),
           "mention": {m for x in q.get("mentions") or [] for m in WD.findall(x.get("entity") or "")}, "answer": set()}
    ans = q.get("answers") or []
    for a in ans if isinstance(ans, list) else [ans]:       # the training file lists result sets; the test file has one
        for b in ((a or {}).get("results") or {}).get("bindings") or []:
            for v in b.values():
                if isinstance(v, dict) and v.get("type") == "uri":
                    out["answer"].update(WD.findall(v.get("value") or ""))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qawiki", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolved", default=None, help="`wdstore resolve`'s output: id, then hot / cold / missing / -> target")
    a = ap.parse_args(argv)
    by_role, sets = collections.defaultdict(set), collections.Counter()
    for name, q in questions(a.qawiki):
        sets[name] += 1
        for role, xs in ids(q).items():
            by_role[role] |= xs
    every = sorted(set().union(*by_role.values()), key=lambda x: int(x[1:]))
    if not a.resolved:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(every) + "\n")
        print(json.dumps({"questions": dict(sets), "ids": {r: len(v) for r, v in by_role.items()}, "distinct": len(every)}))
        return
    status = {}
    with open(a.resolved, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                status[p[0]] = p[1] if not p[1].startswith("->") else "redirect" + ("_missing" if p[-1] == "missing" else "")
    res = {"questions": dict(sets), "by_role": {}}
    for role, xs in by_role.items():
        c = collections.Counter(status.get(x, "unresolved") for x in xs)
        res["by_role"][role] = {"ids": len(xs), **dict(sorted(c.items()))}
    res["through_a_redirect"] = sorted((x for x in every if status.get(x, "").startswith("redirect")), key=lambda x: int(x[1:]))
    res["missing"] = sorted((x for x in every if status.get(x) in ("missing", "redirect_missing")), key=lambda x: int(x[1:]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("through_a_redirect", "missing")}, indent=1))


if __name__ == "__main__":
    main()
