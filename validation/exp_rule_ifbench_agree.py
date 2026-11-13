"""Order 0067, the reader's fifth round: where the reader wrote a rule for an IFBench prompt, does the rule agree with
IFBench's own checker over replies?

Reads the plans `exp_rule_ifbench.py --keep` wrote. For each written rule, K replies: half aimed at the rule's own edge
(built from its leaves, the numbers moved a little either way, so the rule both holds and fails), half plain (texts of
several lengths, a list, capitals, quotes). IFBench's checker labels each reply (every instruction of the prompt
followed, `ifbench_check.py`); the VM judges it by the written rule. A false pass: the rule held, the checker failed.
Line 3 (held-out): agreement at least 95% where written; false passes at most 2% (kill above 5%).

  python validation/exp_rule_ifbench_agree.py --plans <plans.json> [--data <parquet>] [--k 10] [--out <json>]
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
import rule_replies as RR  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402

__wiring__ = "STANDALONE"


def _target(lf: dict, rng) -> int:
    v = int(lf["v"])
    v = {"<": v - 1, ">": v + 1, "!=": v + 1}.get(lf["op"], v)
    return max(0, RR.around(v, rng) if rng.random() < 0.6 else v)


def aimed(plan: dict, rng: random.Random) -> str:
    """A reply built from the rule's leaves, its numbers near each edge."""
    sents, paras, words, start, end, case, hl, title = 3, 1, None, "", "", None, 0, False
    extra = []
    for lf in rp.leaves(plan):
        m = lf["m"]
        if m == "sentence count":
            sents = max(1, _target(lf, rng))
        elif m == "paragraph count":
            paras = max(1, min(40, _target(lf, rng)))
        elif m == "word count":
            words = max(1, _target(lf, rng))
        elif m == "count of the word" and lf.get("arg"):
            extra += [f"{lf['arg']} is here." for _ in range(min(40, _target(lf, rng)))]
        elif m == "starts with" and lf.get("arg"):
            start = lf["arg"] if rng.random() < 0.8 else ""
        elif m == "ends with" and lf.get("arg"):
            end = lf["arg"] if rng.random() < 0.8 else ""
        elif m == "case":
            case = lf.get("v") if rng.random() < 0.8 else None
        elif m == "highlight count":
            hl = _target(lf, rng)
        elif m == "title count":
            title = rng.random() < 0.7
    body = [rng.choice(RR.SENT["en"]) for _ in range(min(sents, 60))] + extra
    if words is not None:
        flat = " ".join(body).split()
        flat = (flat + RR.words(max(0, words - len(flat)), rng).split())[:words] if words else flat
        body = [" ".join(flat)]
    body = [f"*{s.rstrip('.')}*" if i < hl else s for i, s in enumerate(body)]
    per = max(1, len(body) // paras)
    text = "\n\n".join(" ".join(body[i:i + per]) for i in range(0, len(body), per))
    text = (f"{start} " if start else "") + text + (f" {end}" if end else "")
    if title:
        text = "<<A short title>>\n" + text
    return text.lower() if case == "lower" else text.upper() if case == "upper" else text


def plain(rng: random.Random) -> str:
    t = RR.text("en", rng, rng.randint(1, 12))
    return rng.choice([t, t.upper(), f'"{t}"', "\n".join(f"* {s}" for s in t.split(". ")), t.replace(",", ""),
                       "\n\n".join([t, RR.text("en", rng, 3)])])


def checker(pairs: list[dict]) -> dict:
    env = dict(os.environ, CUBBY_IFBENCH_CHECKER=local_paths.value("CUBBY_IFBENCH_CHECKER"))
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "pairs.jsonl", Path(d) / "verdicts.jsonl"
        src.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        subprocess.run([local_paths.value("CUBBY_IFEVAL_PYTHON"), str(HERE / "ifbench_check.py"), str(src), str(dst)],
                       check=True, env=env)
        return {v["id"]: v["followed"] for v in map(json.loads, dst.read_text(encoding="utf-8").splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    data = Path(a.data or Path(local_paths.value("CUBBY_DATASETS") or ".") / "ifbench" / "ifbench_test.parquet")
    items = {str(r["key"]): r for r in pq.read_table(data).to_pylist()}
    plans = json.loads(Path(a.plans).read_text(encoding="utf-8"))
    rng, pairs = random.Random(0), []
    for key, got in plans.items():
        if got["how"] != "written":
            continue
        it = items[key]
        for j in range(a.k):
            reply = aimed(got["plan"], rng) if j < a.k // 2 else plain(rng)
            pairs.append({"id": f"{key}:{j}", "key": key, "prompt": it["prompt"], "instruction_ids": list(it["instruction_id_list"]),
                          "kwargs": list(it["kwargs"]), "reply": reply})
    truth = checker(pairs)
    vm = cc.CubelangSession()
    per, tot, misses = defaultdict(Counter), Counter(), []
    for p in pairs:
        plan = plans[p["key"]]["plan"]
        held = rp.run(plan, p["reply"], lambda src, fn: vm.run(src, fn=fn))["held"]
        want = truth[p["id"]]
        c = {"pairs": 1, "agree": int(held == want), "false passes": int(bool(held) and not want),
             "rule held": int(bool(held)), "checker passes": int(want)}
        tot.update(c)
        for k in p["instruction_ids"]:
            per[k].update(c)
        if bool(held) and not want and len(misses) < 25:
            misses.append({"kinds": p["instruction_ids"], "rule": plans[p["key"]]["surface"], "reply": p["reply"][:120]})
    n = max(1, tot["pairs"])
    out = {"order": "0067", "what": "IFBench, where the reader wrote a rule: the rule against IFBench's checker over replies",
           "plans": Path(a.plans).name, "written": len({p["key"] for p in pairs}), "replies_per_rule": a.k,
           "line": {"agreement": ">= 0.95", "false_passes": "<= 0.02 (kill > 0.05)"},
           "pairs": tot["pairs"], "agreement": round(tot["agree"] / n, 4), "false_passes": round(tot["false passes"] / n, 4),
           "rule_held_share": round(tot["rule held"] / n, 4), "checker_pass_share": round(tot["checker passes"] / n, 4),
           "false_passes_among_rule_held": round(tot["false passes"] / max(1, tot["rule held"]), 4),
           "per_kind": {k: dict(v) for k, v in sorted(per.items())}, "false_pass_examples": misses,
           "vm": vm_identity.identity()}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("written", "pairs", "agreement", "false_passes", "rule_held_share",
                                          "checker_pass_share", "false_passes_among_rule_held")}, indent=1))


if __name__ == "__main__":
    main()
