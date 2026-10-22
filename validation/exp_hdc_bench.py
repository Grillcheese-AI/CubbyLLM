"""exp_hdc_bench -- order 0014's acceptance: the hdc world mounted behind the wiki world, asked through the ask loop
(no model, no network: the questions parse to their plan with the set's own templates, the source below the
stores answers nothing, the CubeLang VM verifies).

1. Lookup: p50 / p95 of a subject's facts (label, alias or Q-id) over `--lookups` random subjects; the process's
   memory with the world mounted.
2. 200 one-hop questions from `eval_ood_scaling_law.pq` (held-out domains), as the set writes them (a subject by one
   of its aliases), whose fact the world holds: spoken and correct (the answer is one of the world's objects for
   that subject and property), spoken and wrong (the kill line: 0), refused; said "as of 2019" counted apart.
3. 50 present-tense questions on time-sensitive properties: none may be said flat (an "as of" or a refusal).

    CUBBY_HDC_WORLD=<hdc_world.sqlite> python validation/exp_hdc_bench.py --src <hdc folder> [--n 200] [--ts 50]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.planner import normalize  # noqa: E402
from hdc_store import HdcWorld, Stores  # noqa: E402
from question_forms import Forms, TemplateEmitter, tensed  # noqa: E402


def set_forms(src: str, world: HdcWorld) -> Forms:
    """Every property's question as the set writes it (its template, the edited label in it), in three tenses."""
    rels = set(world.relations())
    forms: dict = {}
    for r in csv.DictReader(open(os.path.join(src, "relation_template_mapping.csv"), encoding="utf-8")):
        rel = r["relation"].strip()
        if normalize(rel) not in rels:
            continue
        t = r["template"].replace("{relation}", (r["edited_relation"] or rel).strip()).replace("{subject}", "{S}")
        by = forms.setdefault(rel, ("state", {}))[1]
        for tense, form in tensed(t).items():
            by.setdefault(tense, []).append(form)
    return Forms(forms)


class Offline:
    name = "offline"

    def facts(self, entity):
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--db", default=os.environ.get("CUBBY_HDC_WORLD"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ts", type=int, default=50)
    ap.add_argument("--lookups", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", f"exp_hdc_bench_{_dt.date.today().isoformat()}.json"))
    a = ap.parse_args()
    import psutil
    import pyarrow.parquet as pq
    import wikikg as wk
    from ask import AskLoop
    rng, t0 = random.Random(a.seed), time.time()
    hdc = HdcWorld(a.db)
    db = hdc.db
    subs = [s for (s,) in db.execute("SELECT skey FROM facts ORDER BY RANDOM() LIMIT ?", (a.lookups,))]
    lat = []
    for s in subs:
        t = time.perf_counter(); hdc._by_subj.get(s); lat.append(1000 * (time.perf_counter() - t))
    lat.sort()
    world = Stores(wk.wiki_world(), hdc)
    forms = set_forms(a.src, hdc)
    loop = AskLoop(TemplateEmitter(forms), world=world, source=Offline(), lexicon=True)
    loop.resolvers.append(forms)
    rss = psutil.Process().memory_info().rss
    ev = pq.read_table(os.path.join(a.src, "eval_ood_scaling_law.pq"),
                       columns=["triplet_subject", "triplet_relation", "triplet_object", "question_prompt"]).to_pylist()
    rng.shuffle(ev)
    rows, ts_rows, skipped = [], [], 0
    for r in ev:
        if len(rows) >= a.n:
            break
        gold = {normalize(o) for (o,) in db.execute("SELECT obj FROM facts WHERE sq = ? AND pid = ?",
                                                    (r["triplet_subject"], r["triplet_relation"]))}
        p = forms.parse(r["question_prompt"]) if gold else None
        if p is None:
            continue
        # one hop: the question names the row's own subject (an alias of it) and asks the row's own property; a
        # prompt that nests another hop ("the instance of the architect of X") is not this bench's
        q = db.execute("SELECT qid FROM names WHERE key = ?", (normalize(p[0]),)).fetchone()
        pid = db.execute("SELECT pid FROM rels WHERE rel = ?", (p[1],)).fetchone()
        if not q or q[0] != r["triplet_subject"] or not pid or pid[0] != r["triplet_relation"]:
            skipped += 1
            continue
        rec = loop.ask(r["question_prompt"])
        ans = rec.get("answer")
        rows.append({"question": r["question_prompt"], "verified": bool(rec["verified"]), "answer": ans, "tier": rec.get("tier"),
                     "reason": rec.get("reason"), "correct": (normalize(ans) in gold) if rec["verified"] and ans else None})
    ts = db.execute("SELECT subj, rel FROM facts WHERE ts = 1 ORDER BY RANDOM() LIMIT ?", (a.ts * 20,)).fetchall()
    for subj, rel in ts:
        if len(ts_rows) >= a.ts:
            break
        qs = forms.render(rel, subj, "present")
        if not qs:
            continue
        rec = loop.ask(qs[0])
        ts_rows.append({"question": qs[0], "verified": bool(rec["verified"]), "tier": rec.get("tier"), "reason": rec.get("reason"),
                        "flat": bool(rec["verified"] and not rec.get("tier"))})
    spoken = [x for x in rows if x["verified"]]
    rep = {"args": {k: v for k, v in vars(a).items() if k not in ("db", "src", "out")},
           "lookup_ms": {"p50": round(lat[len(lat) // 2], 2), "p95": round(lat[int(len(lat) * 0.95)], 2), "n": len(lat)},
           "rss_gb_mounted": round(rss / 1e9, 2), "facts": len(hdc),
           "one_hop": {"asked": len(rows), "not_one_hop_skipped": skipped, "spoken": len(spoken), "correct": sum(x["correct"] is True for x in spoken),
                       "wrong": sum(x["correct"] is False for x in spoken), "as_of": sum(x["tier"] == "as_of" for x in spoken),
                       "refused": dict(collections.Counter(x["reason"] for x in rows if not x["verified"]))},
           "time_sensitive": {"asked": len(ts_rows), "said_flat": sum(x["flat"] for x in ts_rows),
                              "as_of": sum(x["tier"] == "as_of" for x in ts_rows),
                              "refused": dict(collections.Counter(x["reason"] for x in ts_rows if not x["verified"]))},
           "wall_s": round(time.time() - t0, 1), "rows": rows, "time_sensitive_rows": ts_rows}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("rows")}, indent=1))


if __name__ == "__main__":
    main()
