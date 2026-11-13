"""Order 0067: the host guard (`cubbyllm/speech/rule_guard.py`) on the plans round 6's reader wrote (Cowork,
2026-11-13: "build it and measure it, no model call").

The guard's lexicon was tuned on the train and dev rows' own plans (0 of 15,903 asked); the reads here are the held-out
splits, once:
- **seen kinds** (round 6's real-prompt tests; the second reads: natural prompts, rule-free tasks, unmeasurable
  phrases, v4b's new wordings): of the plans the reader wrote that are right today, how many the guard would turn into
  "asked" (line: at most 1 point), and how many wrong ones it catches;
- **IFBench** (a diagnostic, spent): round 6's replies rebuilt exactly as `exp_rule_ifbench_agree.py` built them (the
  same seed), labelled by IFBench's checker and the VM; of the false passes, how many each of the guard's three parts
  catches, and the guard as a whole.

  python validation/exp_rule_guard.py --tag v6 [--k 10] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import exp_rule_ifbench_agree as IA  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import rule_guard as G  # noqa: E402
from cubbyllm.speech import rule_plan as rp  # noqa: E402
from cubbyllm.speech import rule_reader as rr  # noqa: E402

__wiring__ = "STANDALONE"

LOGS = HERE / "logs"
DS = Path(local_paths.value("CUBBY_DATASETS") or ".")
READS = [  # (rows dir, plans file, splits)
    ("rule_rows_0067_v4_asked_real", "_rule_reader_2026-11-13_{t}_real_plans.json", ("real_rule_test", "real_none_test", "real_asked_test")),
    ("rule_rows_0067_v4_asked_real", "_rule_reader_2026-11-13_{t}_regress_plans.json", ("natural_test", "none_test", "asked_test")),
    ("rule_rows_0067_v4b", "_rule_reader_2026-11-13_{t}_v4b_new_plans.json", ("new_wording",)),
]


def seen(tag: str) -> dict:
    out = {}
    for rows_dir, plans_f, splits in READS:
        rows = {}
        for line in open(DS / rows_dir / "rows.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["split"] in splits:
                rows[r["id"]] = r
        plans = json.loads((LOGS / plans_f.format(t=tag)).read_text(encoding="utf-8"))["zero"]
        for split in splits:
            t, misses = Counter(), []
            for rid, plan in plans.items():
                r = rows.get(rid)
                if r is None or r["split"] != split or not plan:
                    continue
                right = r["plan"] is not None and rr.same(plan, r["plan"])
                g = G.guard(r["instruction"], plan)
                t.update({"written": 1, "right": int(right), "right, guard asks": int(right and bool(g)),
                          "wrong": int(not right), "wrong, guard asks": int(not right and bool(g))})
                if g and right and len(misses) < 8:
                    misses.append({"instruction": r["instruction"][-160:], "plan": rr.surface(plan), "guard": g["why"]})
            t = dict(t)
            t["right_asked_share"] = round(t.get("right, guard asks", 0) / max(1, t.get("right", 0)), 4)
            out[split] = {"counts": t, "right_plans_the_guard_asks": misses}
    return out


def ifbench(tag: str, k: int) -> dict:
    import pyarrow.parquet as pq
    items = {str(r["key"]): r for r in pq.read_table(DS / "ifbench" / "ifbench_test.parquet").to_pylist()}
    plans = json.loads((LOGS / f"_rule_ifbench_2026-11-13_{tag}_plans.json").read_text(encoding="utf-8"))
    rng, pairs = random.Random(0), []                    # the same draw as exp_rule_ifbench_agree.py
    for key, got in plans.items():
        if got["how"] != "written":
            continue
        it = items[key]
        for j in range(k):
            reply = IA.aimed(got["plan"], rng) if j < k // 2 else IA.plain(rng)
            pairs.append({"id": f"{key}:{j}", "key": key, "prompt": it["prompt"], "instruction_ids": list(it["instruction_id_list"]),
                          "kwargs": list(it["kwargs"]), "reply": reply})
    truth = IA.checker(pairs)
    vm = cc.CubelangSession()
    fails = {key: {p["part"] for p in G.parts(items[key]["prompt"], got["plan"])}
             for key, got in plans.items() if got["how"] == "written"}
    t = Counter()
    for p in pairs:
        held = rp.run(plans[p["key"]]["plan"], p["reply"], lambda src, fn: vm.run(src, fn=fn))["held"]
        fp = bool(held) and not truth[p["id"]]
        f = fails[p["key"]]
        t.update({"pairs": 1, "false passes": int(fp), "agree": int(held == truth[p["id"]])})
        for part in (1, 2, 3):
            t[f"false passes caught by part {part}"] += int(fp and part in f)
        t["false passes caught by the guard"] += int(fp and bool(f))
        t["right verdicts the guard would ask"] += int(held == truth[p["id"]] and bool(held) and bool(f))
    written = len(fails)
    return {"written_plans": written, "plans_the_guard_asks": sum(1 for f in fails.values() if f),
            "plans_by_part": {p: sum(1 for f in fails.values() if p in f) for p in (1, 2, 3)}, "pairs": dict(t),
            "false_passes_left": t["false passes"] - t["false passes caught by the guard"],
            "false_pass_share_with_the_guard": round((t["false passes"] - t["false passes caught by the guard"]) / max(1, t["pairs"]), 4),
            "vm": vm_identity.identity()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"order": "0067", "what": "the host guard on round 6's plans: seen kinds (once) and IFBench (a diagnostic)",
           "line": "seen kinds: the guard asks at most 1 point of the plans that are right today",
           "tuned_on": "train and dev rows' own plans (0 of 15,903 asked)", "seen": seen(a.tag), "ifbench": ifbench(a.tag, a.k)}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for s, v in out["seen"].items():
        print(f"  {s:16s} {v['counts']}")
    print(json.dumps({k: v for k, v in out["ifbench"].items() if k != "vm"}, indent=1))


if __name__ == "__main__":
    main()
