"""Order 0075, steps 1, 3 and 4: the rule behind every "right" of 0054's stored records, the lenient matches to read,
and the strict matcher's counts beside the lenient ones.

The records (0054's current arms, under ``$CUBBY_DATA/system_eval``): C is the "after" (``run_0055_after``; NQ its
gather-seat rerun ``nq_final2``; CLUTRR the "before"'s, as 0054 reports it), C-VM is built from C's records
(``cvm_answer`` / ``cvm_verdict``), M is ``run_0054/m``. Each row's verdict is recomputed with `grade.verdict` (M with
its declines) and must equal the stored one; each "right" names its rule (`match_rules.rule`), and a "right" with no
rule is the kill. The read (step 3): for C every "right" whose rule is not ``equal`` and not CLUTRR's ``relation``;
for C-VM and M a sample of 200 per benchmark, seed 75 (all when fewer). Strict (step 4): the verdicts under
``strict=True``; a "right" the strict matcher drops goes to the read, a non-answer is declined.

  python validation/exp_grade_rules.py [--data $CUBBY_DATA/system_eval] [--out validation/logs/grade_rules_<date>.json]
-> the log (ids and counts), and ``to_read_0075.jsonl`` beside the records (question, golds, answer: for the read).

Wired: STANDALONE (order 0075).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

BENCHES = ("simpleqa", "nq", "webq", "clutrr", "aa")
SEED, SAMPLE = 75, 200
EXACT = ("equal", "relation")


def c_path(d: str, b: str) -> str:
    if b == "clutrr":
        return os.path.join(d, "run_0054", "c_clutrr.jsonl")
    if b == "nq":
        return os.path.join(d, "run_0055_after", "nq_final2", "c_gather_nq.jsonl")
    return os.path.join(d, "run_0055_after", f"c_{b}.jsonl")


def load(d: str) -> dict[tuple[str, str], list[dict]]:
    """The records, each row's ``golds`` the question's whole list (``run_0054/questions.jsonl``): a record keeps the
    first 8 only, and the run graded against them all (``golds_in_record`` keeps the record's count)."""
    rd = lambda p: [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    full = {q["id"]: q["golds"] for q in rd(os.path.join(d, "run_0054", "questions.jsonl"))}
    whole = lambda rows: [{**r, "golds": full.get(r["id"], r["golds"]), "golds_in_record": len(r["golds"])} for r in rows]
    out = {}
    for b in BENCHES:
        c = whole(rd(c_path(d, b)))
        out[("C", b)] = c
        if c and "cvm_verdict" in c[0]:
            out[("C-VM", b)] = [{**r, "verdict": r["cvm_verdict"], "answer": r["cvm_answer"],
                                 "spoken": r["cvm_verdict"] != "refused"} for r in c]
        out[("M", b)] = whole(rd(os.path.join(d, "run_0054", "m", f"m_{b}.jsonl")))
    return out


def main() -> None:
    from cubbyllm import local_paths
    from system_eval import grade
    from system_eval.match_rules import rule
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(local_paths.value("CUBBY_DATA") or ".", "system_eval"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = load(a.data)
    rules, mismatch, no_rule, strict_tab, reads = defaultdict(Counter), defaultdict(list), [], {}, []
    for (arm, b), rows in recs.items():
        lenient, strict = Counter(), Counter()
        lenient_matches = []
        for r in rows:
            spoken = bool(r.get("spoken")) and r.get("verdict") != "refused"
            v = grade.verdict(spoken, r.get("answer"), r["golds"], b, declines=(arm == "M"))
            if v != r["verdict"]:
                mismatch[f"{arm}/{b}"].append({"id": r["id"], "stored": r["verdict"], "recomputed": v})
            vs = grade.verdict(spoken, r.get("answer"), r["golds"], b, declines=(arm == "M"), strict=True)
            lenient[r["verdict"]] += 1
            strict[vs] += 1
            if r["verdict"] == "right":
                k = rule(r.get("answer"), r["golds"], b)
                rules[f"{arm}/{b}"][k or "none"] += 1
                if k is None:
                    no_rule.append({"arm": arm, "bench": b, "id": r["id"]})
                elif k not in EXACT:
                    lenient_matches.append((r, k, vs))
        strict_tab[f"{arm}/{b}"] = {"lenient": dict(lenient), "strict": dict(strict),
                                    "right_dropped_by_strict": sum(1 for _, _, vs in lenient_matches if vs != "right")}
        pick = lenient_matches
        if arm != "C" and len(pick) > SAMPLE:
            ids = sorted(random.Random(f"{SEED}:{arm}:{b}").sample(sorted(r["id"] for r, _, _ in pick), SAMPLE))
            pick = [x for x in pick if x[0]["id"] in ids]
        for r, k, vs in pick:
            reads.append({"arm": arm, "bench": b, "id": r["id"], "rule": k, "strict": vs, "question": r.get("question"),
                          "golds": r["golds"], "answer": r.get("answer"), "lenient_matches": len(lenient_matches)})
    reads.sort(key=lambda x: (x["arm"], x["bench"], x["id"]))
    with open(os.path.join(a.data, "to_read_0075.jsonl"), "w", encoding="utf-8") as f:
        for x in reads:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    longer = sum(1 for rows in recs.values() for r in rows if len(r["golds"]) > r["golds_in_record"])
    rep = {"golds": f"from run_0054/questions.jsonl: a record keeps the first 8; {longer} rows (over all arms) have more",
           "what": "order 0075 steps 1, 3, 4: the rule behind every 'right' of 0054's stored records (C, C-VM, M; five "
                   "benchmarks), the lenient matches to read, the strict matcher's counts",
           "date": _dt.date.today().isoformat(),
           "records": {b: os.path.relpath(c_path(a.data, b), a.data).replace(os.sep, "/") for b in BENCHES},
           "m_records": "run_0054/m/m_<bench>.jsonl", "c_vm": "from C's records (cvm_answer, cvm_verdict)",
           "rules": {k: dict(v) for k, v in sorted(rules.items())},
           "right_without_a_rule": no_rule, "kill_right_without_a_rule": len(no_rule) > 0,
           "verdicts_recomputed_differ": {k: {"n": len(v), "first": v[:10]} for k, v in sorted(mismatch.items())},
           "lenient_vs_strict": strict_tab,
           "to_read": dict(Counter(f"{x['arm']}/{x['bench']}" for x in reads)), "read_seed": SEED}
    out = a.out or os.path.join(HERE, "logs", f"grade_rules_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("rules", "kill_right_without_a_rule", "verdicts_recomputed_differ", "to_read")}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
