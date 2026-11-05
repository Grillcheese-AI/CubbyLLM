"""validation/selfimprove/fastlearn_mark.py -- order 0053: the VM marks every cached candidate, at home (CPU, CubeLang).

Each candidate's program is the cached prefix + the word + '"' + the rest of the greedy program, marked as 0049 marks
wd (``mark.Marker``): "right" (the word names the gold property among those it names), "wrong" (it names one other),
"refused" (none, or several without the gold), "caught" (the VM cannot run it). Also kept: the properties the word
names (the learner's readout columns). The ceiling per family: the share of problems with at least one right
candidate among the K. Writes <dir>/fastlearn_marks.jsonl (aligned to the cache's lines) and prints the ceilings.

    python validation/selfimprove/fastlearn_mark.py --dir <the cache's folder> --records R --props P

Wired: STANDALONE (order 0053's harness).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"


def verdict(m: dict) -> str:
    if m["verdict"] == "right":
        return "right" if m["gold_right"] is not False else "wrong"
    return m["verdict"]


def named(pa, word: str) -> list[str]:
    from cubbyllm.sources.wikidata_aliases import _normalize
    return sorted({pa._pid_of.get(_normalize(lab)) for lab in pa.relations(word)} - {None})


def main(argv=None) -> None:
    from selfimprove.mark import Marker
    ap = argparse.ArgumentParser()
    for k in ("dir", "records", "props"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    recs = [json.loads(x) for x in open(a.records, encoding="utf-8") if x.strip()]
    mk = Marker(recs, a.props)
    cache = [json.loads(x) for x in open(os.path.join(a.dir, "fastlearn_cache.jsonl"), encoding="utf-8")]
    ceiling = collections.defaultdict(lambda: [0, 0])
    with open(os.path.join(a.dir, "fastlearn_marks.jsonl"), "w", encoding="utf-8") as f:
        for c in cache:
            out = {"id": c["id"], "verdicts": [], "props": []}
            if c.get("slot"):
                rec = {"task": "wd", "gold": c["gold"], "spans": c["spans"], "question": c["question"], "id": c["id"]}
                for w in c["words"]:
                    out["verdicts"].append(verdict(mk(rec, c["prefix"] + w + '"' + c["rest"])))
                    out["props"].append(named(mk.pa, w))
            k = c["family"] + ("/held" if c["held"] else "/stream")
            ceiling[k][0] += int("right" in out["verdicts"])
            ceiling[k][1] += 1
            f.write(json.dumps(out) + "\n")
    res = {k: {"problems": n, "with_a_right_candidate": r, "ceiling": round(r / max(1, n), 4)}
           for k, (r, n) in sorted(ceiling.items())}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(a.dir, "fastlearn_marks_meta.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
