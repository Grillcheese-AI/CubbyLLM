"""Order 0067, the addendum "say the rule back": its lines, measured.

1. **Every rule applied was shown to whoever set it** (kill below 100%): 1,000 simulated sessions (rules set at every
   level, shown to their setter or to someone else or not yet, corrected, confirmed, withdrawn); at every turn, every
   rule in force is checked for a show to its setter at or before that turn.
2. **The sample to judge by hand** (reported, written down before the next round's results are seen): the reader's
   wrong rules from the held-out reads (round 4 on IFEval's held-out kinds, `rule_heldout_2026-11-13_v4.json`; round 6
   on IFBench, `_rule_ifbench_2026-11-13_v6_plans.json`), a rule counted wrong when the benchmark's checker disagrees
   with it on any of its replies (the replies rebuilt exactly as those reads built them); 100 drawn by seed (half each),
   each with its id and the line Cubby would say back (`rule_say.said_back`); the instructions' text stays off-repo
   (``$CUBBY_DATASETS/rule_say_0067``): no benchmark prompt in a committed file.

  python validation/exp_rule_say.py [--out validation/logs/rule_say_<date>.json] [--sample <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / "builders")]
import exp_rule_agree as AG  # noqa: E402
import exp_rule_ifbench_agree as IA  # noqa: E402
import rule_replies  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402
from cubbyllm.speech import rule_say  # noqa: E402
from cubbyllm.speech.rule_book import LEVELS, RuleBook  # noqa: E402

__wiring__ = "STANDALONE"
PLANS = [{"m": "word count", "op": "<", "v": 50}, {"m": "case", "op": "==", "v": "lower"},
         {"m": "language", "op": "==", "v": "fr"}, {"m": "word", "op": "lacks", "v": "river"}]


def simulate(n: int = 1000) -> dict:
    rng, applied, shown_ok = random.Random(0), 0, 0
    for _ in range(n):
        b, people = RuleBook(confirm=("company",)), ["operator", "admin", "user", "someone else"]
        for turn in range(15):
            act = rng.random()
            if act < 0.4:
                r = b.set(rng.choice(PLANS), rng.choice(LEVELS), rng.choice(people[:3]), at=turn)
                if rng.random() < 0.8:
                    b.show([r.id], r.set_by if rng.random() < 0.85 else "someone else", turn + rng.choice([0, 0, 1]))
            elif act < 0.55 and b.rules:
                r = rng.choice(b.rules)
                if r.withdrawn_at is None:
                    b.correct(r.id, rng.choice(PLANS), r.set_by, turn)
            elif act < 0.65 and b.rules:
                b.confirm([rng.choice(b.rules).id], turn)
            elif act < 0.75 and b.rules:
                r = rng.choice(b.rules)
                if r.withdrawn_at is None:
                    b.withdraw(r.id, turn + 1)
            for r in b.in_force(turn):
                applied += 1
                shown_ok += int(r.shown_to == r.set_by and r.shown_at is not None and r.shown_at <= turn)
    return {"sessions": n, "rule_turns_applied": applied, "shown_to_their_setter_first": shown_ok,
            "share": round(shown_ok / max(1, applied), 6)}


def wrong_ifeval(vm) -> list[dict]:
    log = json.loads((HERE / "logs" / "rule_heldout_2026-11-13_v4.json").read_text(encoding="utf-8"))
    rng, pairs, reads = random.Random(0), [], [r for r in log["reads"]]
    for it in reads:                                       # the same draw as exp_rule_heldout.py
        if it["how"] == "written":
            for j, reply in enumerate(rule_replies.replies(it["kind"], it["kwargs"] or {}, "en", rng, log["replies_per_rule"])):
                pairs.append({"id": f"{it['id']}:{j}", "item": it["id"], "instruction_id": it["kind"],
                              "kwargs": it["kwargs"] or {}, "reply": reply})
    truth, bad = AG.checker(pairs), set()
    by = {it["id"]: it for it in reads}
    for p in pairs:
        if rp.run(by[p["item"]]["plan"], p["reply"], lambda s, fn: vm.run(s, fn=fn))["held"] != truth[p["id"]]:
            bad.add(p["item"])
    return [{"source": "IFEval held-out (round 4)", "id": f"ifeval_heldout_v4:{i}", "kind": by[i]["kind"],
             "instruction": by[i]["instruction"], "plan": by[i]["plan"]} for i in sorted(bad)]


def wrong_ifbench(vm) -> list[dict]:
    import pyarrow.parquet as pq
    items = {str(r["key"]): r for r in pq.read_table(Path(local_paths.value("CUBBY_DATASETS")) / "ifbench" / "ifbench_test.parquet").to_pylist()}
    plans = json.loads((HERE / "logs" / "_rule_ifbench_2026-11-13_v6_plans.json").read_text(encoding="utf-8"))
    rng, pairs = random.Random(0), []
    for key, got in plans.items():                         # the same draw as exp_rule_ifbench_agree.py
        if got["how"] != "written":
            continue
        it = items[key]
        for j in range(10):
            reply = IA.aimed(got["plan"], rng) if j < 5 else IA.plain(rng)
            pairs.append({"id": f"{key}:{j}", "key": key, "prompt": it["prompt"], "instruction_ids": list(it["instruction_id_list"]),
                          "kwargs": list(it["kwargs"]), "reply": reply})
    truth, bad = IA.checker(pairs), set()
    for p in pairs:
        if rp.run(plans[p["key"]]["plan"], p["reply"], lambda s, fn: vm.run(s, fn=fn))["held"] != truth[p["id"]]:
            bad.add(p["key"])
    return [{"source": "IFBench (round 6)", "id": f"ifbench:{k}", "kind": ",".join(plans[k]["kinds"]),
             "instruction": items[k]["prompt"], "plan": plans[k]["plan"]} for k in sorted(bad)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", default=None)
    a = ap.parse_args()
    vm = cc.CubelangSession()
    sim = simulate()
    a_wrong, b_wrong = wrong_ifeval(vm), wrong_ifbench(vm)
    rng = random.Random(0)
    sample = rng.sample(a_wrong, min(50, len(a_wrong))) + rng.sample(b_wrong, min(100 - min(50, len(a_wrong)), len(b_wrong)))
    rows = [dict(s, said_back=rule_say.said_back("set", [s["plan"]]), shows_the_error=None) for s in sample]
    out = {"order": "0067", "what": "say the rule back: its lines", "line_1_shown_before_applied": sim,
           "wrong_rules": {"ifeval_heldout_round4": len(a_wrong), "ifbench_round6": len(b_wrong)},
           "sample": len(rows), "vm": vm_identity.identity()}
    if a.sample:                                          # the benchmarks' text stays off-repo; the log keeps ids
        off = Path(local_paths.value("CUBBY_DATASETS") or ".") / "rule_say_0067"
        off.mkdir(parents=True, exist_ok=True)
        (off / Path(a.sample).name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        Path(a.sample).write_text("".join(json.dumps({k: v for k, v in r.items() if k != "instruction"}, ensure_ascii=False)
                                          + "\n" for r in rows), encoding="utf-8")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
