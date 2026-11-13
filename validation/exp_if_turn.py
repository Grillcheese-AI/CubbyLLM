"""Order 0067, line 4: `exp_if_follow`'s prompts through the turn under rules (step 4), against step 0's baseline (51.8%).

The same prompts (`exp_if_follow.prompts`, seed 0) and the same check (`exp_if_follow.check`). Each prompt's rules are
given as plans (`plan_of`: exp_if_follow's ten kinds as plans over the reply world, the shapes
`tests/speech/test_rule_plan.py` pins against the same check), filed in a rule book, and the turn runs as
`rule_turn.TurnRules` does in the chat: steering through grilly2's hook (a mask, an opening as the prefix), then up to
``--candidates`` replies (the first greedy, as the baseline; the rest sampled at ``--temperature``), each run by the VM
against every rule; the first that passes is said; when none passes, the reply is Cubby's "can't" line (a refusal).

Scored per rule check, as the baseline: followed (said, and the check passes), refused (Cubby's line said), broken (a
reply said whose check fails: line 1's kill, which the VM's check before speaking should hold at 0).
The rules here are the prompts' own, known to the host; reading them from the words is step 3's reader (line 3).

  python validation/exp_if_turn.py --export <450M export> --tokenizer <tokenizer.json> --adapter <talk adapter> \
      [--n 300] [--candidates 4] [--temperature 0.7] [--no-steer] [--tag _turn]
Writes validation/logs/if_turn_<date><tag>.json (the summary) and _if_turn_<date><tag>_rows.json (scratch: replies).
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
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import exp_if_follow as IF  # noqa: E402

from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.speech import rule_turn as rt  # noqa: E402
from cubbyllm.speech.rule_book import RuleBook  # noqa: E402

__wiring__ = "STANDALONE"


def plan_of(rule: dict) -> dict:
    """exp_if_follow's kinds as plans (the same shapes as tests/speech/test_rule_plan.py)."""
    k = rule["kind"]
    if k == "words":
        return {"m": "word count", "op": "<=" if "max" in rule else ">=", "v": rule.get("max", rule.get("min"))}
    if k == "sentences":
        if "exact" in rule:
            return {"m": "sentence count", "op": "==", "v": rule["exact"]}
        return {"and": [{"m": "sentence count", "op": ">", "v": 0}, {"m": "sentence count", "op": "<=", "v": rule["max"]}]}
    if k in ("forbid", "require"):
        return {"m": "word", "op": "lacks" if k == "forbid" else "has", "v": rule["word"]}
    if k == "language":
        return {"m": "language", "op": "==", "v": rule["lang"]}
    if k == "list":
        return {"and": [{"m": "list item count", "op": ">=", "v": 2}] + ([{"m": "list item count", "op": "==", "v": rule["n"]}]
                                                                       if "n" in rule else [])}
    if k == "json":
        return {"and": [{"m": "json", "op": "==", "v": "yes"}] + [{"m": "json key", "op": "has", "v": key} for key in rule["keys"]]}
    if k == "no_markdown":
        return {"m": "markdown", "op": "==", "v": "no"}
    if k == "case":
        return {"m": "case", "op": "==", "v": rule["case"]}
    if k == "first":
        alts = [a.strip() for a in rule["text"].split("|")]
        return {"m": "opening", "op": "has", "v": alts[0]} if len(alts) == 1 else \
            {"or": [{"m": "opening", "op": "has", "v": a} for a in alts]}
    raise KeyError(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "standin", "data", "out", "chat_sft.jsonl"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--no-steer", action="store_true")
    ap.add_argument("--tag", default="_turn")
    a = ap.parse_args()
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    from cubbyllm.emit.talk import Cubby450mTalk
    from cubbyllm.ops import grilly_seats as ops

    rows = IF.prompts(a.data, a.n, a.seed)
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"talk": a.adapter})
    seat = Cubby450mTalk(trunk)
    vm = cc.CubelangSession()
    run = lambda src, fn: vm.run(src, fn=fn)              # noqa: E731
    tk, eos = trunk.tokenizer, trunk.eos
    drawn, t0 = collections.Counter(), time.time()
    for i, r in enumerate(rows):
        book = RuleBook()
        for rule in r["rules"]:
            book.set(plan_of(rule), "user", "the prompt", at=0, text=rule["kind"])
        tr = rt.TurnRules(book, 0, None if a.no_steer else seat, run)
        st = tr.steering
        hook = {}
        if st["suppress"]:
            proc = ops.suppress_processor(st["suppress"])
            hook = {"logits_processor": [proc]} if proc is not None else {}
        prefix = st["prefix"] or ""
        ids = tk.encode(r["prompt"] + ((" " + prefix) if prefix else "")).ids[-2048:]
        said, tried = None, []
        for k in range(a.candidates):
            temp = 0.0 if k == 0 else a.temperature
            got = trunk.generate("talk", [ids], a.max_new, temperature=temp, seed=k, **hook)[0]
            if eos in got:
                got = got[:got.index(eos)]
            reply = ((prefix + " ") if prefix else "") + tk.decode(got).split("\nQuestion:")[0].strip()
            tried.append(reply)
            if tr.why_not(reply, steered=bool(hook)) is None:
                said = reply
                drawn[k + 1] += 1
                break
        if said is None:
            drawn["refused"] += 1
        r.update(reply=said if said is not None else rt.CANT.get(r.get("lang") or "en", rt.CANT["en"]),
                 said="model" if said is not None else "cant", tried=len(tried), record=tr.record(said or rt.CANT["en"]))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}  {dict(drawn)}  {time.time() - t0:.0f}s", flush=True)
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        for rule in r["rules"]:
            ok = IF.check(rule, r["reply"])
            c = per[rule["kind"]]
            c["n"] += 1
            c["followed"] += int(r["said"] == "model" and ok)
            c["refused"] += int(r["said"] == "cant")
            c["broken"] += int(r["said"] == "model" and not ok)
    tot = collections.Counter()
    for c in per.values():
        tot.update(c)
    base = 0.518
    summary = {"order": "0067", "what": "exp_if_follow through the turn under rules (line 4)", "prompts": len(rows),
               "checks": tot["n"], "followed": tot["followed"], "refused": tot["refused"], "broken": tot["broken"],
               "followed_rate": round(tot["followed"] / max(1, tot["n"]), 4), "baseline_rate": base,
               "line": round(base + 0.15, 4), "candidates": a.candidates, "temperature": a.temperature,
               "steer": not a.no_steer, "said_at_candidate": {str(k): v for k, v in sorted(drawn.items(), key=str)},
               "by_kind": {k: dict(v, followed_rate=round(v["followed"] / max(1, v["n"]), 4)) for k, v in sorted(per.items())},
               "seconds": round(time.time() - t0), "adapter": os.path.basename(os.path.normpath(a.adapter)), "seed": a.seed}
    day = _dt.date.today().isoformat()
    logs = os.path.join(HERE, "logs")
    json.dump(summary, open(os.path.join(logs, f"if_turn_{day}{a.tag}.json"), "w", encoding="utf-8"), indent=1)
    json.dump(rows, open(os.path.join(logs, f"_if_turn_{day}{a.tag}_rows.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("checks", "followed", "refused", "broken", "followed_rate", "line",
                                              "said_at_candidate")}, indent=1))
    for k, v in summary["by_kind"].items():
        print(f"  {k:12s} followed {v['followed']:4d}/{v['n']:<4d} refused {v['refused']:4d} broken {v['broken']}")


if __name__ == "__main__":
    main()
