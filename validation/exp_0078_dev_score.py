"""exp_0078_dev_score - order 0078's dev read scored (Cowork's ruling, 2026-11-20), from the rows
`exp_0078_dev_read.py` wrote, with no second pass:

- **the speak rule (deciding):** for k in {8, 16} (k = 8 is the first 8 tries), m chosen by `speak_vote.choose_m` (the
  smallest m keeping dev's wrong at or under 1% of the questions); right / wrong / refused at that (k, m), and the
  table for every m. The arm's dev right is the better k's right at its own m.
- **one greedy try (reported, not deciding):** right / wrong / refused, the gold within 0.01 (the speak rule's).

`drop`: a json file with a "drop" list (or a bare list) of ids to leave out (S0's footnote).

**The split (Cowork, 2026-11-24; the hypothesis file's held-out rule):** `held=(80, 200)` chooses k and m on dev 0:80
and 200:329 (209 problems, by the read's `i`: dev's order in `exp_0078_dev_read.load`), then reads dev 80:200 (120
problems, all unseen templates) once at the frozen k and m: right / wrong / refused, the top-1 (the most common value
of the k tries, right), and per m the spoken share and the spoken-wrong share, for the pre-registered next step (a path
verifier if top-1 < 0.300, or spoken-wrong > 1% at every m that speaks on 20% or more). `held=None`: all rows, as before.

**The answer-kind rules (0078's addendum, part 1; off unless `rules` is given):** `gold_check` counts, per rule, the
GSM8K train golds it would refuse (`cubbyllm.reasoning.answer_kind`); a rule refusing more than 7 (0.1%) is dropped.
With `rules` and `questions` (id -> question), each try breaking a kept rule does not vote (`answer_kind.kept`), and k,
m and the held read are chosen and read as above. `rules_read` gives both reads, without and with the rules, and the
order's verdict on dev 80:200 (kept: no fewer right, no more wrong).

Wiring: STANDALONE (order 0078).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.reasoning.speak_vote import choose_m  # noqa: E402

__wiring__ = "STANDALONE"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(path: str, drop: str = "") -> list[dict]:
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    if drop:
        d = json.load(open(drop, encoding="utf-8"))
        ids = set(d["drop"] if isinstance(d, dict) else d)
        rows = [r for r in rows if r["id"] not in ids]
    return rows


def greedy(rows: list[dict]) -> dict:
    c = {"right": 0, "wrong": 0, "refused": 0}
    for r in rows:
        v, gold = num(r["greedy"]["answer"]), float(r["gold"])
        c["refused" if v is None else "right" if abs(v - gold) <= 0.01 else "wrong"] += 1
    return c


HELD = (80, 200)


def held_out_ids(paths: list[str], lo: int = HELD[0], hi: int = HELD[1]) -> set:
    """The ids of dev lo:hi in the dev read's order (arithmetic, val, whole questions, file order): the school's gate
    leaves them out (`train_school_torch.py --gate-skip`)."""
    ids = []
    for p in paths:
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("task") == "arithmetic" and r.get("split", "val") == "val" and "#" not in r["id"]:
                ids.append(r["id"])
    return set(ids[lo:hi])


def top1(values: list, gold: float) -> bool:
    from cubbyllm.reasoning.speak_vote import vote
    v = vote(values, 1)
    if v["answer"] is None and v["why"] == "tie":       # a tie for first: the top-1 is not one value
        return False
    return v["answer"] is not None and abs(v["answer"] - gold) <= 0.01


def values(r: dict, rules=None, questions=None) -> list:
    """A row's tries' VM values; with ``rules``, a value breaking one is None (it does not vote)."""
    vs = [num(t["answer"]) for t in r["tries"]]
    if rules:
        from cubbyllm.reasoning.answer_kind import kept
        vs = kept(vs, questions[r["id"]], rules)
    return vs


GOLD_LINE = 7          # a rule refusing more than 7 of GSM8K train's 7,473 golds (0.1%) is dropped (the order)


def gold_check(train_path: str) -> dict:
    """{"golds": n, "refused by": {rule: count}, "kept": [rules within the line], "dropped": [...]}: each rule on its
    own, on GSM8K train's golds (the number after `####`)."""
    from cubbyllm.reasoning.answer_kind import RULES, broken
    rows = [json.loads(ln) for ln in open(train_path, encoding="utf-8")]
    by = {r: 0 for r in RULES}
    for row in rows:
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        for rule in RULES:
            by[rule] += broken(gold, row["question"], (rule,)) is not None
    return {"golds": len(rows), "refused by": by, "line": GOLD_LINE,
            "kept": [r for r in RULES if by[r] <= GOLD_LINE], "dropped": [r for r in RULES if by[r] > GOLD_LINE]}


def held_read(rows: list[dict], k: int, m: int, rules=None, questions=None) -> dict:
    """Dev lo:hi at the frozen k and m, read once; and every m's spoken / spoken-wrong share (the next step's line)."""
    dev = [(values(r, rules, questions)[:k], float(r["gold"])) for r in rows]
    ch = choose_m(dev, k)
    n = max(1, len(dev))
    by_m = {mm: {"spoken share": round((c["right"] + c["wrong"]) / n, 4), "spoken-wrong share": c["wrong share"]}
            for mm, c in ch["by m"].items()}
    speaks = [mm for mm, c in by_m.items() if c["spoken share"] >= 0.20]
    t1 = sum(top1(v, g) for v, g in dev) / n
    return {"questions": len(dev), "k": k, "m": m, "at m": ch["by m"].get(m), "top-1": round(t1, 4),
            "by m": by_m, "next step (path verifier) pre-registered": t1 < 0.300 or all(
                by_m[mm]["spoken-wrong share"] > 0.01 for mm in speaks)}


def score(path: str, drop: str = "", held: "tuple | None" = None, rules=None, questions=None) -> dict:
    rows = load_rows(path, drop)
    out_rows = []
    if held:
        out_rows = [r for r in rows if held[0] <= r["i"] < held[1]]
        rows = [r for r in rows if not held[0] <= r["i"] < held[1]]
    dev = [(values(r, rules, questions), float(r["gold"])) for r in rows]
    by_k = {}
    for k in (8, 16):
        ch = choose_m(dev, k)
        at = ch["by m"].get(ch["m"]) if ch["m"] else None
        by_k[k] = {"m": ch["m"], "at m": at, "by m": ch["by m"]}
    best = max((k for k in by_k if by_k[k]["at m"]), key=lambda k: (by_k[k]["at m"]["right"], -k), default=None)
    secs = [r["seconds"]["greedy"] + r["seconds"]["tries"] for r in rows + out_rows]
    res = {"rows": os.path.basename(path), "dropped": os.path.basename(drop) if drop else None,
           "chosen on": f"dev without {held[0]}:{held[1]}" if held else "all rows", "questions": len(rows),
            "speak rule": {"k": best, "m": by_k[best]["m"] if best else None,
                           "right": by_k[best]["at m"]["right"] if best else 0, "by k": by_k},
            "greedy": greedy(rows),
            "any try right (the oracle)": sum(any(v is not None and abs(v - g) <= 0.01 for v in vs) for vs, g in dev),
            "seconds a question": {"mean": round(sum(secs) / max(1, len(secs)), 1),
                                   "median": sorted(secs)[len(secs) // 2] if secs else None}}
    if held and best:
        res["held read"] = {**held_read(out_rows, best, by_k[best]["m"], rules, questions), "greedy": greedy(out_rows),
                            "dev": f"{held[0]}:{held[1]}"}
    if rules:
        res["rules"] = list(rules)
        res["tries out by rule"] = tries_out(rows + out_rows, rules, questions)
    return res


def tries_out(rows: list[dict], rules, questions) -> dict:
    """How many tries each rule takes out of the vote (the first rule a value breaks), and their golds' share right."""
    from cubbyllm.reasoning.answer_kind import broken
    out = {r: {"tries": 0, "right": 0} for r in rules}
    for r in rows:
        g = float(r["gold"])
        for v in (num(t["answer"]) for t in r["tries"]):
            b = broken(v, questions[r["id"]], rules) if v is not None else None
            if b:
                out[b]["tries"] += 1
                out[b]["right"] += abs(v - g) <= 0.01
    return out


def rules_read(path: str, train_path: str, questions: dict, drop: str = "", held=HELD) -> dict:
    """The gold check, then the read without and with the kept rules; the verdict on dev lo:hi (the order's line)."""
    gc = gold_check(train_path)
    off = score(path, drop, held)
    on = score(path, drop, held, gc["kept"], questions) if gc["kept"] else None
    res = {"gold check": gc, "without the rules": off, "with the rules": on}
    a, b = (off.get("held read") or {}).get("at m"), ((on or {}).get("held read") or {}).get("at m")
    if a and b:
        res["verdict on the held read"] = {
            "without": {k: a[k] for k in ("right", "wrong", "refused")},
            "with": {k: b[k] for k in ("right", "wrong", "refused")},
            "kept": b["right"] >= a["right"] and b["wrong"] <= a["wrong"]}
    return res


__all__ = ["GOLD_LINE", "HELD", "gold_check", "greedy", "held_out_ids", "held_read", "load_rows", "rules_read", "score",
           "top1", "tries_out", "values"]
