"""r3o -- order 0046, R3o (Cowork, 2026-11-04): R3's arms, each choosing only among the properties the question's items
actually carry, outgoing and incoming (``r3o_props``). A property an item does not carry cannot verify, so it is never
handed over.

Rows, gold properties and the 450M's programs are R3's (``r3_gold``; R3's dev programs, read back, not regenerated).
The candidates of a row are the union over its items: those the question names (located), else all of them. Arms:
  lexicon_content   the frozen lexicon's wordings found in the words left once the names are out, function-word
                    wordings skipped; a property scores its longest wording's length; candidates only
  450m              the program's ``HOP1`` wording: the serving property table's properties for it (the whole
                    wording, scored above any part), and the frozen lexicon's wordings found inside it; candidates only
  grammar           today's everyday table, its relation through the serving table; candidates only
  qfhrr@tau         the qFHRR codebook scored against the candidates' codes only (on grilly2), read past the margin
  lexicon_content>450m  the lexicon, and the 450M only where the lexicon reads nothing (the exact instrument first)
The list is R3's (at most 5, a run of equal scores whole); judged as in R3.

Read on every scored row and on the rows whose gold property the items carry today ("held"); the rows where they do
not ("missing fact") are reported apart, with what each arm hands over there: a read there names a property the item
carries that is not the asked one, where a near property would become a wrong answer (R8's kill line reads it).

    python validation/retrieval_bench/r3o.py --split dev [--n 300]
-> validation/logs/retrieval_R3o_<split>_<date>.json

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

WHOLE = 99                                       # the serving table's properties for the 450M's whole wording
GATE = {"simplequestions_wd/en": 0.85, "qald9plus/fr": 0.80}


def within(found: list[tuple[str, int, list]], cands: set, extra: dict | None = None) -> dict:
    """R3's list from (wording, words, properties) matches, candidates only."""
    from retrieval_bench.r3_arms import top
    scores: dict[str, float] = dict(extra or {})
    for _w, n, ps in found:
        for p in ps:
            if p in cands:
                scores[p] = max(scores.get(p, 0), n)
    return {"read": bool(scores), **top(scores), "wordings": [w for w, _n, ps in found if set(ps) & cands]}


def arms_for(r: dict, cands: set, lx: dict, gram, program: str | None, arm: str = "450m") -> dict:
    from retrieval_bench import r3_450m as M
    from retrieval_bench import r3_arms as A
    text = A.leftover(r["question"], r["spans"])
    out = {"lexicon_content": within(A.matches(text, lx["by_wording"].get(r["lang"], {}), True), cands)}
    hop = M.read(program)["hop1"]
    if hop:
        wording = hop.replace("$E1", " ").replace("$E2", " ")
        whole = {p: WHOLE for p in gram.aliases.pids(wording) if p in cands}
        out[arm] = {**within(A.matches(wording, lx["by_wording"]["en"], True), cands, whole), "hop1": hop}
    else:
        out[arm] = {"read": False, "top": [], "undecided": False, "hop1": None}
    g = gram(r["question"])
    keep = [p for p in g.get("top") or () if p in cands]
    out["grammar"] = {"read": bool(keep), **A.top({p: 1.0 for p in keep})}
    out[f"lexicon_content>{arm}"] = out["lexicon_content"] if out["lexicon_content"]["read"] else out[arm]
    return out


def summarize(rows: list[dict]) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        if r["why"] == "gold":
            groups[f"{r['set']}/{r['lang']}"].append(r)
            groups["all"].append(r)
            if r.get("kind_asked"):
                groups[f"kind/{r['kind_asked']}"].append(r)
    out = {}
    for g, rs in groups.items():
        held = [r for r in rs if r["held"]]
        miss = [r for r in rs if not r["held"]]
        row = {"scored": len(rs), "held": len(held), "missing_fact": len(miss),
               "in_unknown": sum(1 for r in rs if r["in_unknown"]), "arms": {}}
        for a in sorted({a for r in rs for a in r["judge"]}):
            def part(sub):
                c = collections.Counter(r["judge"][a] for r in sub)
                n = len(sub) or 1
                return {"n": len(sub), **dict(c), "coverage": round(sum(1 for r in sub if r["out"][a].get("read")) / n, 4),
                        "recall_at_5": round(c["hit"] / n, 4), "wrong_rate": round(c["wrong"] / n, 4)}
            row["arms"][a] = {"all": part(rs), "held": part(held),
                              "missing_fact": {"n": len(miss), "handed_over": sum(1 for r in miss if r["out"][a].get("read"))}}
        out[g] = row
    return out


def main() -> None:
    from cubbyllm.worlds.fact_pool import V4_TABLE
    from retrieval_bench import r3_arms as A
    from retrieval_bench import r3_qfhrr as Q
    from retrieval_bench.r3_gold import prepare, rows as load
    from retrieval_bench.r3o_props import ItemProps, candidates
    from retrieval_sets.common import data_root, source
    from retrieval_sets.lexicon import check
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.0, 0.1, 0.25])
    ap.add_argument("--programs", default=None, help="R3's programs (default: its 300-row run of this split)")
    ap.add_argument("--arm", default="450m", help="the model arm's name (order 0048: 450m_wd, with its programs)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--kind", choices=("none", "one", "chain"), default="none",
                    help="R3k: the asked kind bounds the properties (one subclass step, or the whole chain)")
    a = ap.parse_args()
    t0, src = time.time(), source()
    rows = load(a.split, a.n)
    prep = prepare(rows, src)
    # the row's items: those whose names the question carries (located), else all of them (QALD's SPARQL also names
    # class constraints, Q5 "human", whose incoming properties would be every relation of a person)
    items = {r["id"]: [m["qid"] for m in r["mentions"] if m.get("span")] or [m["qid"] for m in r["mentions"]] for r in rows}
    props = ItemProps(src).get(sorted({q for v in items.values() for q in v}))
    progs = {}
    ppath = a.programs or os.path.join(data_root(), f"r3_programs_{a.split}_n300.jsonl")
    for line in open(ppath, encoding="utf-8"):
        d = json.loads(line)
        if "id" in d:
            progs[d["id"]] = d.get("program")
    lx, gram = check(), A.Grammar()
    kinds, kinfo = ({}, None)
    if a.kind != "none":
        from retrieval_bench.r3k_kind import bound
        kinds, kinfo = bound(rows, items, props, src, lx, a.kind)
    cands = {}
    for r in rows:
        c = candidates(items[r["id"]], props)
        if r["id"] in kinds:                             # R3k: the asked kind bounds them further
            r["held_before_kind"] = bool(set(r["golds"]) & c["all"])
            c["all"] = kinds[r["id"]]
        cands[r["id"]] = c["all"]
        r["held"], r["in_unknown"] = bool(set(r["golds"]) & c["all"]), c["in_unknown"]
        r["n_cands"] = len(c["all"])
        r["out"] = arms_for(r, c["all"], lx, gram, progs.get(r["id"]), a.arm)
    rep_q = Q.score(Q.job(rows, lx, str(V4_TABLE), cands))
    for r in rows:
        for t in a.taus:
            r["out"][f"qfhrr@{t}"] = Q.decide(rep_q["ranked"].get(r["id"]) or [], t)
        r["judge"] = {k: A.judge(o, r["golds"]) for k, o in r["out"].items()}
    summary = summarize(rows)
    gates = {arm: {part: {g: (summary.get(g, {}).get("arms", {}).get(arm, {}).get(part) or {}).get("recall_at_5") for g in GATE}
                   for part in ("all", "held")} for arm in summary["all"]["arms"]}
    for arm, v in gates.items():
        v["passes_all"] = all((x or 0) >= GATE[g] for g, x in v["all"].items())
        v["passes_held"] = all((x or 0) >= GATE[g] for g, x in v["held"].items())
    fetched = sorted({d["fetched"] for d in props.values()})
    rep = {"what": "order 0046 R3o: the relation among the item's own properties", "date": _dt.date.today().isoformat(),
           "split": a.split, "n_per_set_lang": a.n, "prepared": prep, "props_fetched": fetched,
           "props": {"items": len(props), "in_unavailable": sum(1 for d in props.values() if d["in"] is None)},
           "qfhrr_scorer": {k: v for k, v in rep_q.items() if k != "ranked"}, "gates": gates, "summary": summary,
           "kind": {"mode": a.kind, **(kinfo or {}),
                    "gold_dropped_by_kind": sum(1 for r in rows if r.get("held_before_kind") and not r.get("held")),
                    "margin_r3k": (__import__("retrieval_bench.r3k_kind", fromlist=["x"]).pick_margin(summary, a.taus)
                                   if a.kind != "none" else None)},
           "wall_s": round(time.time() - t0),
           "rows": [{"id": r["id"], "set": r["set"], "lang": r["lang"], "golds": r["golds"], "why": r["why"], "held": r.get("held"),
                     "n_cands": r.get("n_cands"), "judge": r.get("judge"), "kind_asked": r.get("kind_asked"),
                     "out": {k: {x: o[x] for x in ("top", "hop1", "wordings", "margin") if x in o} for k, o in (r.get("out") or {}).items()}}
                    for r in rows]}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R3o_{a.split}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for g, s in summary.items():
        print(g, s["scored"], "held", s["held"], {k: (v["all"]["recall_at_5"], v["held"]["recall_at_5"], v["all"]["wrong_rate"])
                                                    for k, v in s["arms"].items()})
    print(json.dumps(gates, indent=1), "->", path)


if __name__ == "__main__":
    main()
