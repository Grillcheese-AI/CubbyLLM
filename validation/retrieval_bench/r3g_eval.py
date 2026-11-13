"""r3g_eval -- order 0070, C: each arm behind each guard on a row variant (present: the gold carried; hidden: the gold
taken out; natural: R3o's 38 rows the items do not carry), the counts per set and language, the curves on guard-dev and
the acceptance lines.

A row variant's record, per arm: the arm's list among the variant's candidates, G1's two numbers, G2's decision at each
share (None: unavailable), G3's list. A guard and its thresholds turn the record into the list handed over.
  hidden, natural   a handover is a list read (R3o's "handed_over"); a refusal is right there
  present           judged as R3: hit, wrong, undecided, unread; a refusal counts apart ("refused")

Wired: STANDALONE (order 0070's bench).
"""
from __future__ import annotations

import collections
import time

from retrieval_bench import r3g_guards as G
from retrieval_bench.r3g_arms import MODELS, QF

__wiring__ = "STANDALONE"

GUARDS = ("none", "G1", "G2", "G1+G2", "G3")


def guards_for(arm: str) -> tuple:
    return GUARDS if arm in QF or arm in MODELS else tuple(g for g in GUARDS if g != "G3")


def record(arms, arm: str, c: dict, cands: set, classes: list, table: dict) -> dict:
    """What every guard needs for one arm on one row variant, every threshold at once."""
    from retrieval_bench.r3g_classes import expected
    base = arms.out(arm, c, cands)
    g1 = arms.qbests(c, cands) if arm in QF else G.bests(arms.scores(arm, c, None), cands)
    g2 = {}
    for s in G.SHARES:
        exp = expected(classes, table, s, G.N_MIN)
        if exp is None:
            g2[s] = None
            continue
        absent = exp - cands
        g2[s] = G.g2(arms.scores(arm, c, cands | absent), absent) if absent else False
    rec = {"base": base, "g1": g1, "g2": g2}
    if arm in QF or arm in MODELS:
        rec["g3"] = G.g3(base, arms.named(c, cands))
    return rec


def handed(rec: dict, guard: str, t: dict) -> dict:
    """The list the arm hands over behind `guard` at thresholds `t`."""
    if guard == "none":
        return rec["base"]
    if guard == "G3":
        return rec["g3"]
    refuse = False
    if "G1" in guard:
        refuse |= G.g1(*rec["g1"], t["m"])
    if "G2" in guard:
        refuse |= bool(rec["g2"][t["s"]])
    return G.apply(rec["base"], refuse)


def outcome(out: dict, golds) -> str:
    from retrieval_bench.r3_arms import judge
    return "refused" if out.get("refused") else judge(out, golds)


def tally(items: list[tuple[dict, list, str]]) -> dict:
    """(handed list, golds, variant) for one group -> its counts."""
    c = collections.Counter(outcome(o, g) for o, g, _v in items)
    n = len(items)
    return {"n": n, **dict(c), "read": sum(1 for o, _g, _v in items if o.get("read")),
            "recall_at_5": round(c["hit"] / n, 4) if n else None, "wrong_rate": round(c["wrong"] / n, 4) if n else None}


def table_of(rows: list[dict], recs: dict, arm: str, guard: str, t: dict, variant: str) -> dict:
    """Counts per set/language and overall for one arm, guard and thresholds on one variant."""
    groups = collections.defaultdict(list)
    for r in rows:
        o = handed(recs[(r["id"], variant)][arm], guard, t)
        for k in ("all", f"{r['set']}/{r['lang']}"):
            groups[k].append((o, r["golds"], variant))
    return {k: tally(v) for k, v in sorted(groups.items())}


SQ = "simplequestions_wd/en"


def curve(rows: list[dict], recs: dict, arm: str, guard: str) -> list[dict]:
    """guard-dev: per threshold, hidden handovers, the present loss, false refusals and wrongs refused."""
    base = table_of(rows, recs, arm, "none", {}, "present")
    pts = []
    for t in G.grid(guard):
        hid = table_of(rows, recs, arm, guard, t, "hidden")
        pre = table_of(rows, recs, arm, guard, t, "present")
        false_ref = sum(1 for r in rows if outcome(recs[(r["id"], "present")][arm]["base"], r["golds"]) == "hit"
                        and handed(recs[(r["id"], "present")][arm], guard, t).get("refused"))
        wrong_ref = sum(1 for r in rows if outcome(recs[(r["id"], "present")][arm]["base"], r["golds"]) == "wrong"
                        and handed(recs[(r["id"], "present")][arm], guard, t).get("refused"))
        had_sq = (base.get(SQ, {}).get("recall_at_5") or 0) >= G.SQ_LINE
        pts.append({"t": t, "handovers": hid["all"]["read"], "hidden_n": hid["all"]["n"],
                    "loss": round(base["all"]["recall_at_5"] - pre["all"]["recall_at_5"], 4),
                    "present_recall_at_5": pre["all"]["recall_at_5"], "present_wrong_rate": pre["all"]["wrong_rate"],
                    "false_refusals": false_ref, "wrongs_refused": wrong_ref,
                    "sq_recall_at_5": pre.get(SQ, {}).get("recall_at_5"),
                    "sq_ok": (not had_sq) or (pre.get(SQ, {}).get("recall_at_5") or 0) >= G.SQ_LINE})
    return pts


def verdict(test_hidden: dict, natural: dict, present: dict, present_base: dict) -> dict:
    """The pre-registered lines for one arm and guard."""
    hid_rate = test_hidden["all"]["read"] / test_hidden["all"]["n"]
    nat = natural["all"]["read"]
    loss = present_base["all"]["recall_at_5"] - present["all"]["recall_at_5"]
    sq0, sq = present_base.get(SQ, {}).get("recall_at_5") or 0, present.get(SQ, {}).get("recall_at_5") or 0
    had = sq0 >= G.SQ_LINE
    passes = {"test_handovers_at_most_10pct": hid_rate <= 0.10, "natural_at_most_4": nat <= 4,
              "present_loss_at_most_2pts": loss <= G.LOSS + 1e-12, "sq_at_least_085_where_it_was": (not had) or sq >= G.SQ_LINE}
    kills = {"test_handovers_over_25pct": hid_rate > 0.25, "natural_over_16": nat > 16,
             "sq_under_085_where_it_was_above": had and sq < G.SQ_LINE}
    return {"guard_test_handover_rate": round(hid_rate, 4), "natural_handovers": nat, "present_loss": round(loss, 4),
            "sq_recall_at_5": sq, "lines": passes, "kills": kills, "passes": all(passes.values()), "killed": any(kills.values())}


def timings(arms, rows: list[dict], ctxs: dict, cands: dict, classes: dict, table: dict, t_of: dict) -> dict:
    """Per guard and arm, the time of the guard's own work on each present row (its extra scoring, the class lookup,
    the decision), in ms: p50, p95, max."""
    from retrieval_bench.r3g_classes import expected
    out = {}
    for arm, per in t_of.items():
        for guard, t in per.items():
            if guard == "none":
                continue
            ms = []
            for r in rows:
                c, cd = ctxs[r["id"]], cands[r["id"]]
                base = arms.out(arm, c, cd)                  # the arm's own list: not the guard's work
                t0 = time.perf_counter()
                if guard == "G3":
                    G.g3(base, arms.named(c, cd))
                if "G1" in guard:
                    G.g1(*(arms.qbests(c, cd) if arm in QF else G.bests(arms.scores(arm, c, None), cd)), t["m"])
                if "G2" in guard:
                    exp = expected(classes[r["id"]], table, t["s"], G.N_MIN)
                    if exp:
                        G.g2(arms.scores(arm, c, cd | (exp - cd)), exp - cd)
                ms.append((time.perf_counter() - t0) * 1000)
            ms.sort()
            out[f"{arm}|{guard}"] = {"n": len(ms), "p50_ms": round(ms[len(ms) // 2], 3),
                                     "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 3), "max_ms": round(ms[-1], 3)}
    return out


__all__ = ["GUARDS", "curve", "guards_for", "handed", "outcome", "record", "table_of", "tally", "timings", "verdict"]
