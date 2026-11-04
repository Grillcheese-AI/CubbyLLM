"""r3_relation -- order 0046, R3: which property does the question ask, given its item? Coverage first (the share of
each set an arm reads), then property recall@5 against the gold property (``r3_gold``).

Arms (``r3_arms``, ``r3_qfhrr``, ``r3_450m``):
  grammar            today's host reader (the everyday table) and the serving property table
  lexicon            the frozen exact lexicon over the words left once the names are out
  lexicon_content    the same without wordings made only of function words
  qfhrr@tau          the qFHRR codebook of every property, read only past a rank-normalized margin (on grilly2)
  450m               the 450M in the programs seat writes the program; its HOP1 wording through the property table
  <lexicon>+qfhrr@tau  the lexicon, and the codebook only where the lexicon reads nothing

Per set and language: rows; rows scored (a gold property known); per arm the rows read, hit, wrong, undecided, unread;
coverage = read / rows; recall@5 = hit / scored; wrong rate = wrong / scored.

Gate (the order): recall@5 at least 0.85 on SimpleQuestions-Wikidata and 0.80 on QALD-9-plus FR. The qFHRR arm is kept
only if it adds at least 3 points at the same wrong rate: on dev, ``tau`` is the smallest of ``--taus`` at which the
lexicon+qfhrr arm's wrong rate, pooled over every dev set, is no higher than the lexicon's alone; test is read once with it.

    python validation/retrieval_bench/r3_relation.py --split dev [--n 300] [--taus 0 0.1 0.25 0.5]
    python validation/retrieval_bench/r3_relation.py --split test --tau <dev's>
-> validation/logs/retrieval_R3_<split>_<date>.json (ids, properties and counts only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

GATE = {"simplequestions_wd/en": 0.85, "qald9plus/fr": 0.80}
LEX = ("lexicon", "lexicon_content")


def run(rows: list[dict], taus: list[float], use_450m: bool, use_qfhrr: bool, programs_path: str) -> dict:
    from retrieval_bench import r3_arms as A
    from retrieval_sets.lexicon import check
    lx, gram, info = check(), A.Grammar(), {}
    for r in rows:
        text = A.leftover(r["question"], r["spans"])
        r["out"] = {"grammar": gram(r["question"]), "lexicon": A.lexicon(text, r["lang"], lx),
                    "lexicon_content": A.lexicon(text, r["lang"], lx, content_only=True)}
    if use_qfhrr:
        from cubbyllm.worlds.fact_pool import V4_TABLE
        from retrieval_bench import r3_qfhrr as Q
        rep = Q.score(Q.job(rows, lx, str(V4_TABLE)))
        info["qfhrr_scorer"] = {k: v for k, v in rep.items() if k != "ranked"}
        for r in rows:
            ranked = rep["ranked"].get(r["id"]) or []
            for t in taus:
                q = Q.decide(ranked, t)
                r["out"][f"qfhrr@{t}"] = q
                for lex in LEX:
                    r["out"][f"{lex}+qfhrr@{t}"] = r["out"][lex] if r["out"][lex]["read"] else q
    if use_450m:
        from retrieval_bench import r3_450m as M
        got = M.run(rows, programs_path)
        info["seat"] = got["seat"]
        for r in rows:
            p = got["programs"].get(r["id"]) or {}
            rd = M.read(p.get("program"))
            r["out"]["450m"] = {**A.pids_of(rd["hop1"], gram.aliases), **rd, "wall_s": p.get("wall_s")}
    for r in rows:
        r["judge"] = {a: A.judge(o, r["golds"]) for a, o in r["out"].items()}
    return info


def summarize(rows: list[dict]) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['set']}/{r['lang']}"].append(r)
        groups["all"].append(r)
    out = {}
    for g, rs in groups.items():
        scored = [r for r in rs if r["why"] == "gold"]
        arms = sorted({a for r in rs for a in r["out"]})
        out[g] = {"rows": len(rs), "scored": len(scored), "why": dict(collections.Counter(r["why"] for r in rs)), "arms": {}}
        for a in arms:
            c = collections.Counter(r["judge"][a] for r in scored)
            read = sum(1 for r in rs if (r["out"].get(a) or {}).get("read"))
            out[g]["arms"][a] = {"coverage": round(read / len(rs), 4) if rs else 0.0, "read": read, **dict(c),
                                 "recall_at_5": round(c["hit"] / len(scored), 4) if scored else None,
                                 "wrong_rate": round(c["wrong"] / len(scored), 4) if scored else None}
        if g.startswith("simplequestions"):
            inv = [r for r in scored if (r.get("extra") or {}).get("inverse")]
            out[g]["inverse_rows"] = len(inv)
    return out


def pick_tau(summary: dict, taus: list[float], lex: str = "lexicon_content") -> float | None:
    base = (summary["all"]["arms"].get(lex) or {}).get("wrong_rate")
    for t in sorted(taus):
        got = (summary["all"]["arms"].get(f"{lex}+qfhrr@{t}") or {}).get("wrong_rate")
        if base is not None and got is not None and got <= base:
            return t
    return None


def gates(summary: dict) -> dict:
    out = {}
    for a in sorted({a for g in summary.values() for a in g["arms"]}):
        got = {g: (summary.get(g, {}).get("arms", {}).get(a) or {}).get("recall_at_5") for g in GATE}
        out[a] = {"recall": got, "passes": all(v is not None and v >= GATE[g] for g, v in got.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--tau", type=float, help="test: dev's tau")
    ap.add_argument("--no-450m", action="store_true")
    ap.add_argument("--no-qfhrr", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from retrieval_bench.r3_gold import prepare, rows as load
    from retrieval_sets.common import data_root, source
    t0 = time.time()
    taus = [a.tau] if a.split == "test" and a.tau is not None else a.taus
    rows = load(a.split, a.n)
    prep = prepare(rows, source())
    progs = os.path.join(data_root(), f"r3_programs_{a.split}_n{a.n}.jsonl")
    info = run(rows, taus, use_450m=not a.no_450m, use_qfhrr=not a.no_qfhrr, programs_path=progs)
    summary = summarize(rows)
    rep = {"what": "order 0046 R3: the relation, given the item", "date": _dt.date.today().isoformat(), "split": a.split,
           "n_per_set_lang": a.n, "taus": taus, "prepared": prep, **info,
           "tau_from_dev": ({lex: pick_tau(summary, taus, lex) for lex in LEX} if a.split == "dev" and not a.no_qfhrr else a.tau),
           "gates": gates(summary), "summary": summary, "wall_s": round(time.time() - t0),
           "rows": [{"id": r["id"], "set": r["set"], "lang": r["lang"], "golds": r["golds"], "why": r["why"],
                     "inverse": bool((r.get("extra") or {}).get("inverse")), "judge": r["judge"],
                     "out": {k: {x: o[x] for x in ("top", "relation", "hop1", "hops", "wordings", "margin", "profile") if x in o}
                             for k, o in r["out"].items()}} for r in rows],
           "args": vars(a)}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R3_{a.split}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({g: {"rows": s["rows"], "scored": s["scored"],
                          **{k: (v["coverage"], v["recall_at_5"], v["wrong_rate"]) for k, v in s["arms"].items()}}
                      for g, s in summary.items()}, indent=1))
    print("tau_from_dev", rep["tau_from_dev"], "->", path)


if __name__ == "__main__":
    main()
