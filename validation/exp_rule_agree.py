"""Order 0067, line 3: does a rule plan, run by the VM over the reply world, agree with the benchmark's own checker?

For each sampled rule (a row with an IFEval form), K replies built near the rule's edge (`validation/rule_replies.py`);
IFEval's checker labels each (`validation/ifeval_check.py`, in the checker's own interpreter); the plan is run by the
VM on each (`cubbyllm/speech/rule_plan.run`). Plans:
  gold  -- the row's own plan (what the builder drew): reads the plan family and the reply world against the checker,
           with no seat in the way. Line 3's agreement cannot be met by any reader where this misses;
  seat  -- the plans a reader wrote (``--plans`` and ``--arm``: `exp_rule_reader.py`'s ``--keep`` output), where
           written; a row the reader left "asked" counts as not written.
Reported: agreement; the rule passing a reply the checker fails (line 3's kill reads it on held-out kinds); the
checker's pass share (the replies' balance); all per kind.

  python validation/exp_rule_agree.py [--rows <rows.jsonl>] [--n 100] [--k 10] [--splits dev,new_wording] [--out <json>]
The checker's interpreter, folder and nltk data: $CUBBY_IFEVAL_PYTHON, $CUBBY_IFEVAL_CHECKER, $CUBBY_IFEVAL_NLTK.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import rule_replies  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402

__wiring__ = "STANDALONE"


def checker(pairs: list[dict]) -> dict:
    """{id: followed} from IFEval's checker, run in its own interpreter."""
    env = dict(os.environ, CUBBY_IFEVAL_CHECKER=local_paths.value("CUBBY_IFEVAL_CHECKER"),
               CUBBY_IFEVAL_NLTK=local_paths.value("CUBBY_IFEVAL_NLTK") or "")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "pairs.jsonl", Path(d) / "verdicts.jsonl"
        src.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        subprocess.run([local_paths.value("CUBBY_IFEVAL_PYTHON"), str(HERE / "ifeval_check.py"), str(src), str(dst)],
                       check=True, env=env)
        return {v["id"]: v["followed"] for v in map(json.loads, dst.read_text(encoding="utf-8").splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    ap.add_argument("--plans", default=None, help="a reader's plans, {row id: plan or null} (the seat arm)")
    ap.add_argument("--arm", default="few", help="which of the reader's arms (with --plans)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="dev,new_wording")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "rule_rows_0067" / "rows.jsonl")
    rows = [json.loads(line) for line in open(rows_p, encoding="utf-8")]
    pool = [r for r in rows if r["split"] in a.splits.split(",") and len(r.get("ifeval") or []) == 1 and "+" not in r["kind"]]
    seat = json.loads(Path(a.plans).read_text(encoding="utf-8"))[a.arm] if a.plans else None
    if seat is not None:                                 # only the rows the reader read
        pool = [r for r in pool if r["id"] in seat]
    rng = random.Random(a.seed)
    sample = rng.sample(pool, min(a.n, len(pool)))
    pairs, plans = [], {}
    for r in sample:
        plans[r["id"]] = r["plan"] if seat is None else seat.get(r["id"])
        iid, kw = r["ifeval"][0]
        for j, reply in enumerate(rule_replies.replies(iid, kw, r["lang"], rng, a.k)):
            pairs.append({"id": f"{r['id']}:{j}", "row": r["id"], "kind": r["kind"], "instruction_id": iid, "kwargs": kw,
                          "reply": reply})
    truth = checker(pairs)
    vm = cc.CubelangSession()
    run = lambda src, fn: vm.run(src, fn=fn)                                # noqa: E731
    tally, per_kind = Counter(), defaultdict(Counter)
    misses = []
    for p in pairs:
        plan = plans[p["row"]]
        if plan is None:
            tally["not written"] += 1
            continue
        got = rp.run(plan, p["reply"], run)
        want = truth[p["id"]]
        agree = got["held"] == want
        for c in (tally, per_kind[p["kind"]]):
            c.update({"pairs": 1, "agree": int(agree), "checker passes": int(want),
                      "rule passes, checker fails": int(bool(got["held"]) and not want), "asked": int(got["how"] == "asked")})
        if not agree and len(misses) < 25:
            misses.append({"kind": p["kind"], "kwargs": p["kwargs"], "reply": p["reply"][:160], "checker": want,
                           "rule": got["held"], "statements": got["statements"][:4]})
    n = max(1, tally["pairs"])
    out = {"order": "0067", "what": "rule plans run by the VM against IFEval's checker on replies built to pass and fail",
           "arm": "seat" if seat else "gold", "rows": len(sample), "replies_per_rule": a.k, "seed": a.seed, "splits": a.splits,
           "counts": dict(tally), "agreement": round(tally["agree"] / n, 4),
           "rule_passes_checker_fails": round(tally["rule passes, checker fails"] / n, 4),
           "checker_pass_share": round(tally["checker passes"] / n, 4),
           "per_kind": {k: dict(v, agreement=round(v["agree"] / max(1, v["pairs"]), 3)) for k, v in sorted(per_kind.items())},
           "misses": misses}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text[:4000])


if __name__ == "__main__":
    main()
