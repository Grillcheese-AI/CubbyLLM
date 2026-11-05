"""validation/selfimprove/t3.py -- order 0049's T3: mistakes as training signal, on the planted gaps, at equal compute.

  round 1   each planted gap's first problems (2/3 of its stream problems, up to --t3-problems) practised by the base
            adapter: 8 sampled attempts each, the VM marking each (the mistake memory)
  arms      A: the right programs only; B: + the gap's own negatives; C: + the other planted gaps' negatives too. Each
            from the base adapter, the same steps and the same rows a step (B and C: half of each batch's rows are
            wrong programs, unlikelihood from their first divergence; a problem with no right attempt keeps its
            negatives in the memory, unused)
  measures  held-out right rate (the gap's held problems); transfer (a sibling gap's held problems, not trained on:
            wd/en -> wd/fr, wd/fr -> wd/en, level k -> k + 1); right answers lost (rows A gets right that the arm gets
            wrong, over every held row read); round 2: the gap's other problems practised afresh by each arm's adapter
            (repeated identical wrong programs, attempts until right)
  gate      B or C beats A by >= 5 points held-out, or needs >= 30% fewer attempts until right, with right answers lost
            <= 0.5% of rows; kill: lost > 0.5%, or no gain on any measure
Writes <out>/t3.json.

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import json
import math
import os

from cubbyllm.selfimprove.mistakes import MistakeMemory

from selfimprove.loop import Loop, key_of, ok

__wiring__ = "STANDALONE"


def siblings(keys: dict) -> dict:
    """{planted family: its sibling family}, from the planted families' names (wd/en <-> wd/fr, steps=k -> k + 1)."""
    out = {}
    for fam in keys:
        if fam.startswith("wd/"):
            out[fam] = "wd/fr" if fam == "wd/en" else "wd/en"
        elif fam.startswith("arithmetic/steps="):
            out[fam] = f"arithmetic/steps={int(fam.split('=')[1]) + 1}"
    return out


def practise(lp: Loop, mm: MistakeMemory, rows: list[dict], key: str, weights) -> None:
    keep = lp.mm
    lp.mm = mm
    lp.practice(rows, key, weights)
    lp.mm = keep


def run(lp: Loop, stream: list[dict], held: list[dict], planted: list[str], out: str, a) -> dict:
    fams = {f: [r for r in stream if r["family"] == f][:a.t3_problems] for f in planted}
    keys = {f: key_of(rows[0]) for f, rows in fams.items() if rows}
    held_of = lambda f: [r for r in held if r["family"] == f]                                    # noqa: E731
    mm1 = MistakeMemory(8)
    split = {f: (rows[:2 * len(rows) // 3], rows[2 * len(rows) // 3:]) for f, rows in fams.items()}
    for f, (r1, _) in split.items():
        practise(lp, mm1, r1, keys[f], lp.t.base)
    by_id = {r["id"]: r for r in stream}
    sib = siblings(keys)
    battery = [r for r in held if r["family"] not in keys]
    res = {"round1": {f: mm1.stats(keys[f]) for f in keys}, "gaps": {}}
    for f, key in keys.items():
        right = [(by_id[p]["prompt"], prog) for p, prog in mm1.right_set(key)]
        own = [(by_id[p]["prompt"], w, r) for p, w, r in mm1.pairs(key)]
        other = [(by_id[p]["prompt"], w, r) for p, w, r in mm1.pairs(exclude=key)]
        if len(right) < a.min_right:
            res["gaps"][f] = {"why": "too few right programs in round 1", "right": len(right)}
            continue
        steps = max(a.min_steps, min(a.max_steps, math.ceil(a.epochs * len(right) / a.batch)))
        rows_read = held_of(f) + held_of(sib.get(f, "")) + battery
        arms, marks = {}, {}
        for arm, negs in (("A", []), ("B", own), ("C", own + other)):
            lp.t.use(lp.t.base)
            train = lp.t.train(right, negs, steps=steps, batch=a.batch, lr=a.lr, ul=a.ul)
            w = lp.t.state()
            m = lp.read(rows_read, w)
            marks[arm] = [ok(x) for x in m]
            n_g, n_s = len(held_of(f)), len(held_of(sib.get(f, "")))
            mm2 = MistakeMemory(8)
            practise(lp, mm2, split[f][1], key, w)
            arms[arm] = {"train": train, "negatives_given": len(negs),
                         "held": _share(marks[arm][:n_g]), "transfer": _share(marks[arm][n_g:n_g + n_s]),
                         "battery": _share(marks[arm][n_g + n_s:]), "round2": mm2.stats(key)}
            lp.t.save(w, os.path.join(out, "t3", f"{f.replace('/', '_')}_{arm}"), {"gap": key, "arm": arm})
        for arm in ("B", "C"):
            lost = sum(x and not y for x, y in zip(marks["A"], marks[arm]))
            arms[arm]["right_lost"] = {"rows": lost, "share": round(lost / max(1, len(rows_read)), 4)}
            ar, br = arms["A"]["round2"]["attempts_until_right"], arms[arm]["round2"]["attempts_until_right"]
            arms[arm]["vs_A"] = {"held_points": None if arms[arm]["held"] is None else
                                 round(100 * (arms[arm]["held"] - arms["A"]["held"]), 1),
                                 "fewer_attempts": None if not ar or br is None else round(1 - br / ar, 3)}
            v = arms[arm]["vs_A"]
            arms[arm]["gate"] = bool(arms[arm]["right_lost"]["share"] <= 0.005 and (
                (v["held_points"] or 0) >= 5 or (v["fewer_attempts"] or 0) >= 0.30))
        res["gaps"][f] = {"key": key, "sibling": sib.get(f), "steps": steps, "right": len(right),
                          "own_pairs": len(own), "other_pairs": len(other), "rows_read": len(rows_read), "arms": arms}
        print(json.dumps({f: {k: {x: arms[k].get(x) for x in ("held", "transfer", "vs_A", "right_lost")}
                              for k in arms}}), flush=True)
    json.dump(res, open(os.path.join(out, "t3.json"), "w", encoding="utf-8"), indent=1)
    return res


def _share(xs: list[bool]) -> float | None:
    return None if not xs else round(sum(xs) / len(xs), 4)


__all__ = ["run", "siblings"]
