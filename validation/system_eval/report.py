"""validation/system_eval/report.py -- order 0054's tables, from the arms' records and the read; no second pass.

Arms, per benchmark:
  C       ``c_<bench>.jsonl`` (``run_c.py --seat replay``; CLUTRR ``clutrr_c.py``)
  C-VM    C's records: ``cvm_answer`` / ``cvm_verdict`` (the value C reached, spoken uncertified)
  C-none  ``c_none_<bench>.jsonl`` (the record pass: the stack with no model in the programs seat; CLUTRR: C)
  M       ``m_<bench>.jsonl`` (``gpu.py m``)
For each: right / wrong / refused (counts, %), the wrong rate among spoken answers with its 95% interval, the
estimate where unmatched answers were read by sample; the same over the questions never seen in development. For C:
spoken answers by proposer and by where their facts came from, refusals by kind, the VM's calls and time per check,
the time per question. The line (pre-registered): C's wrong at most 1% of its spoken answers on every benchmark, and
fewer than C-VM's. ``--to-read`` writes what a person reads (C's unmatched all; C-VM's and M's by a seeded sample of
200; on AA-Omniscience, 50 refusals for a missing fact, for whether Wikidata holds the fact).

    python validation/system_eval/report.py --dir <the run's folder> [--to-read] [--out report.json]

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

LINE = 0.01
AA_MISSING = 50


def _rows(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()] if os.path.exists(path) else []


def arms(d: str, bench: str) -> dict[str, list[dict]]:
    """{arm: rows}, each row with ``id``, ``verdict``, ``seen`` (C-VM built from C's records)."""
    c = _rows(os.path.join(d, f"c_{bench}.jsonl"))
    out = {"C": c,
           "C-VM": [{**r, "verdict": r["cvm_verdict"], "answer": r["cvm_answer"]} for r in c],
           "C-none": _rows(os.path.join(d, f"c_none_{bench}.jsonl")) if bench != "clutrr" else c,
           "M": _rows(os.path.join(d, f"m_{bench}.jsonl"))}
    return {k: v for k, v in out.items() if v}


def reads(d: str) -> dict[str, dict[str, str]]:
    """{arm: {id: right | wrong}} from ``reads.jsonl`` (a person's read)."""
    out: dict = collections.defaultdict(dict)
    for r in _rows(os.path.join(d, "reads.jsonl")):
        out[r["arm"]][r["id"]] = r["verdict"]
    return out


def c_detail(rows: list[dict]) -> dict:
    spoken = [r for r in rows if r["spoken"]]
    by_prop = collections.defaultdict(collections.Counter)
    for r in spoken:
        by_prop[r.get("proposer") or "none"][r["verdict"]] += 1
    calls = sum(r.get("vm_calls") or 0 for r in rows)
    return {"spoken_by_proposer": {k: dict(v) for k, v in sorted(by_prop.items())},
            "spoken_450m": sum(1 for r in spoken if r.get("proposer") == "450m"),
            "spoken_by_facts": dict(collections.Counter(r.get("facts_from") or "none" for r in spoken)),
            "refused_by_kind": dict(collections.Counter(r.get("refusal") or "none" for r in rows if not r["spoken"])
                                    .most_common()),
            "seat_called": sum(1 for r in rows if r.get("prompt")),
            "vm_calls": calls, "vm_ms_per_check": round(1000 * sum(r.get("vm_s") or 0 for r in rows) / max(1, calls), 1),
            "s_per_question": round(sum(r.get("wall_s") or 0 for r in rows) / max(1, len(rows)), 2)}


def table(d: str, benches) -> dict:
    from system_eval.grade import rates
    rd = reads(d)
    out = {}
    for b in benches:
        a = arms(d, b)
        if not a:
            continue
        t = {}
        for arm, rows in a.items():
            t[arm] = {"all": rates(rows, rd.get(arm)),
                      "unseen": rates([r for r in rows if not r.get("seen")], rd.get(arm))}
        if "C" in a:
            t["C"]["detail"] = c_detail(a["C"])
        out[b] = t
    return out


def line(tab: dict) -> dict:
    """The pre-registered line, per benchmark: C's wrong among spoken at most 1%, and fewer wrong than C-VM's."""
    res = {}
    for b, t in tab.items():
        if "C" not in t:
            continue
        c, v = t["C"]["all"], t.get("C-VM", {}).get("all")
        rate = c["wrong"] / max(1, c["spoken"])
        res[b] = {"c_wrong": c["wrong"], "c_spoken": c["spoken"], "c_wrong_of_spoken": round(rate, 4),
                  "under_1pct": rate <= LINE, "cvm_wrong": v["wrong"] if v else None,
                  "fewer_than_cvm": (c["wrong"] < v["wrong"]) if v else None, "unread": c["unread"]}
    res["holds"] = all(r["under_1pct"] and r["fewer_than_cvm"] and not r["unread"] for r in res.values())
    return res


def to_read(d: str, benches) -> dict:
    """Writes ``to_read.jsonl`` (one line a row to read, with what a reader needs) and returns the counts."""
    from system_eval.grade import to_read as pick
    n = collections.Counter()
    with open(os.path.join(d, "to_read.jsonl"), "w", encoding="utf-8") as f:
        for b in benches:
            for arm, rows in arms(d, b).items():
                if arm == "C-none":
                    continue
                by_id = {r["id"]: r for r in rows}
                for i in pick(rows, arm):
                    r = by_id[i]
                    f.write(json.dumps({"arm": arm, "id": i, "bench": b, "question": r["question"], "golds": r["golds"],
                                        "answer": r.get("answer"), "said": r.get("said"), "plan": r.get("plan"),
                                        "trace": r.get("trace"), "proposer": r.get("proposer"),
                                        "verdict": None, "why": None}, ensure_ascii=False) + "\n")
                    n[f"{arm}/{b}"] += 1
            if b == "aa":
                miss = sorted(r["id"] for r in arms(d, b).get("C", []) if r.get("refusal") == "fact_missing")
                pickd = sorted(random.Random("54:aa").sample(miss, min(AA_MISSING, len(miss))))
                by_id = {r["id"]: r for r in arms(d, b)["C"]} if miss else {}
                for i in pickd:
                    r = by_id[i]
                    f.write(json.dumps({"arm": "aa_missing", "id": i, "bench": b, "question": r["question"],
                                        "golds": r["golds"], "plan": r.get("plan"), "seed": r.get("seed"),
                                        "in_wikidata": None, "why": None}, ensure_ascii=False) + "\n")
                n["aa_missing"] = len(pickd)
    return dict(n)


def main(argv=None) -> None:
    from system_eval.bench import BENCHES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--to-read", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    tab = table(a.dir, BENCHES)
    res = {"tables": tab, "line": line(tab)}
    if a.to_read:
        res["to_read"] = to_read(a.dir, BENCHES)
    for b, t in tab.items():
        for arm, r in t.items():
            x = r["all"]
            print(f"{b:9s} {arm:7s} n {x['n']:5d}  right {x['right']:7.1f}  wrong {x['wrong']:7.1f}  refused "
                  f"{x['refused']:5d}  wrong/spoken {x['wrong_of_spoken']['rate']:.4f}{'  (est.)' if x['estimated'] else ''}")
    print(json.dumps(res["line"], indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
