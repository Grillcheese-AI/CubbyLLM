"""Order 0072, build step 5: every verb's programs on their cases, the VM against the Python evaluator, and the gate.

For each verb and kind: 20 cases (`validation/builders/verb_cases*.py`), half broken one named way; a case whose
evaluator result is not what its breakage says is drawn again (counted). Each program passes the static guard
(`cubbyllm/speech/verb_guard.py`) before it runs; the VM's verdict is compared with the evaluator's
(`verb_contracts.py`). Then the time of each list program at 50 and 250 items, and the gate's fixed set through the
fake host (`verb_fake_host.py`).
Lines (0072): 0 unexplained disagreements (a disagreement from comparing a missing value is the kill); the gate:
ask-first without a yes 100% "ask", with a yes "do", no tool "refuse", "done" only after the read-back agrees;
p95 at most 50 ms a program at 50 items.

  python validation/exp_verb_programs.py [--n 20] [--seed 0] [--out validation/logs/verb_programs_<date>.json]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / "builders")]
import verb_cases as VC  # noqa: E402
import verb_cases_act as VA  # noqa: E402
import verb_contracts as EV  # noqa: E402
import verb_fake_host as FH  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import verb_guard, verb_programs  # noqa: E402

__wiring__ = "STANDALONE"

SIZE_CAP = {"add": 249, "delete": 249, "merge": 244}
LIST_VARIANTS = ("filter", "sort", "move", "replace", "add", "delete", "set", "copy", "merge", "apply", "same",
                 "aggregate", "mapped", "required", "equal")


def intended(case) -> bool:
    got = EV.evaluate(case["kind"], case["variant"], case["data"])
    if case["variant"] == "gate":
        return (got == 2) == (case["breakage"] is None)
    return got == (0 if case["breakage"] else 1)


def run(vm, case):
    src = verb_programs.program(case["kind"], case["variant"], case["data"])
    why = verb_guard.check(src)
    if why:
        return None, why, 0.0
    t0 = time.perf_counter()
    try:
        res = vm.run(src, fn="solve").get("result")
    except Exception as e:                                # a VM error is a named disagreement, not a stop
        return f"error: {str(e)[:120]}", None, time.perf_counter() - t0
    return (None if res is None else int(float(res))), None, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng, vm = random.Random(a.seed), cc.CubelangSession()
    per, disagree, refused, redraws = defaultdict(Counter), [], [], 0
    pairs = sorted(VC.VARIANT_OF) + [(v, "act") for v in VA.ACT_VERBS]
    for verb, kind in pairs:
        made = VA.cases_act(verb, rng, a.n) if kind == "act" else VC.cases(verb, kind, rng, a.n)
        for case in made:
            for _ in range(20):
                if intended(case):
                    break
                redraws += 1
                case["data"] = VC.build(verb, case["variant"], rng, case["lang"], case["breakage"]) if kind != "act" \
                    else next(c for c in VA.cases_act(verb, rng, a.n) if c["variant"] == case["variant"]
                              and c["breakage"] == case["breakage"])["data"]
            got, why, _ = run(vm, case)
            for _ in range(20):                            # a case the guard refuses (a shared VM key) is drawn again
                if not why or kind == "act":
                    break
                refused.append({"verb": verb, "kind": kind, "variant": case["variant"], "why": why[:3], "redrawn": True})
                case["data"] = VC.build(verb, case["variant"], rng, case["lang"], case["breakage"])
                if intended(case):
                    got, why, _ = run(vm, case)
            want = EV.evaluate(kind, case["variant"], case["data"])
            key = f"{verb}/{kind}"
            per[key].update({"cases": 1, "broken": int(bool(case["breakage"]))})
            if why:
                per[key]["refused by the guard"] += 1
                refused.append({"verb": verb, "kind": kind, "variant": case["variant"], "why": why[:3]})
                continue
            per[key]["agree"] += int(got == want)
            if got != want:
                disagree.append({"verb": verb, "kind": kind, "variant": case["variant"], "breakage": case["breakage"],
                                 "lang": case["lang"], "vm": got, "evaluator": want, "data": case["data"]})
    times = {}
    for variant in LIST_VARIANTS:
        verb = next(v for (v, k), var in VC.VARIANT_OF.items() if var == variant)
        kind = next(k for (v, k), var in VC.VARIANT_OF.items() if var == variant)
        for size in (50, 250):
            n = min(size, SIZE_CAP.get(variant, size))
            ts = []
            for _ in range(10):
                case = {"kind": kind, "variant": variant, "data": VC.build(verb, variant, rng, "en", None, size=n)}
                got, why, dt = run(vm, case)
                if not why:
                    ts.append(dt * 1000)
            if ts:
                times[f"{variant}@{size}"] = {"items": n, "median_ms": round(statistics.median(ts), 2),
                                              "p95_ms": round(sorted(ts)[max(0, int(len(ts) * 0.95) - 1)], 2)}
    gate, outcomes = VA.gate_set(), Counter()
    host = FH.FakeHost()
    gate_rows = []
    for g in gate:
        before = len(host.calls)
        got = FH.run_act(g, vm, host)
        reached = len(host.calls) > before
        outcomes[(g["scenario"], g["class"], got == g["want"])] += 1
        gate_rows.append({**{k: g[k] for k in ("verb", "lang", "scenario", "class", "want")}, "got": got, "host_reached": reached})
    ask_no_yes = [r for r in gate_rows if r["class"] == "ask first" and r["scenario"] in ("no yes", "a yes from an earlier turn only")]
    p95_50 = max((t["p95_ms"] for k, t in times.items() if k.endswith("@50")), default=None)
    out = {"order": "0072", "what": "the verb programs on their cases: the VM against the evaluator; the gate; time",
           "pairs": len(pairs), "cases": sum(c["cases"] for c in per.values()),
           "agree": sum(c["agree"] for c in per.values()), "disagreements": len(disagree),
           "refused_by_the_guard": len(refused), "label_redraws": redraws,
           "gate": {"instructions": len(gate_rows), "right": sum(r["got"] == r["want"] for r in gate_rows),
                    "ask_first_without_a_yes_this_turn": len(ask_no_yes),
                    "of_which_asked": sum(r["got"] == "ask" for r in ask_no_yes),
                    "host_reached_without_a_yes": sum(r["host_reached"] for r in ask_no_yes),
                    "wrong": [r for r in gate_rows if r["got"] != r["want"]][:20]},
           "time": times, "p95_ms_at_50_worst": p95_50, "per_pair": {k: dict(v) for k, v in sorted(per.items())},
           "disagreement_cases": disagree[:40], "refusals": refused[:20], "vm": vm_identity.identity()}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pairs", "cases", "agree", "disagreements", "refused_by_the_guard",
                                          "label_redraws", "p95_ms_at_50_worst")}, indent=1))
    print(json.dumps({k: v for k, v in out["gate"].items() if k != "wrong"}, indent=1))
    for d in disagree[:10]:
        print("DIS", d["verb"], d["kind"], d["variant"], d["breakage"], "vm", d["vm"], "eval", d["evaluator"])


if __name__ == "__main__":
    main()
