"""r3g -- order 0070: the missing-fact guard, read on the hidden-fact set (``r3g_set``), in front of R3o's relation arms.

  --part dev    guard-dev only: per arm and guard, the curve (hidden handovers against the present rows' loss, false
                refusals and wrongs refused) over the thresholds' grid, and the thresholds set by the pre-registered rule
                (``r3g_guards.choose``) -> ``r3g_thresholds.json`` beside this file, with their data and date
  --part test   the thresholds frozen: guard-test's hidden rows read once (a second read is refused), the present rows
                (all 870, and by half), R3o's 38 natural rows, the acceptance lines per arm and guard, every count per
                set and language; the share of rows G2 finds "unavailable"; each guard's time per question
  --part time   the guards' time per question only (present rows; no outcome is read), for a quiet machine

Every number carries the caveat: a hidden fact is not distributed like a missing one.

    python validation/retrieval_bench/r3g.py --part dev|test|time
-> validation/logs/retrieval_R3g_<part>_<date>.json

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

LOGS = os.path.join(os.path.dirname(HERE), "logs")
THRESHOLDS = os.path.join(HERE, "r3g_thresholds.json")
R3O_LOGS = {"retrieval_R3o_dev_2026-11-04.json": ("lexicon_content", "450m", "qfhrr@0.0", "qfhrr@0.25"),
            "retrieval_R3o_dev_2026-11-06_450m_wd.json": ("450m_wd",)}


def load() -> dict:
    """R3o's dev rows rebuilt (rows, candidates, readings), checked against R3o's logs; the split; G2's table."""
    from retrieval_bench import r3_arms as A
    from retrieval_bench import r3g_classes as C
    from retrieval_bench import r3g_set as S
    from retrieval_bench.r3_gold import prepare, rows as load_rows
    from retrieval_bench.r3g_arms import Arms, check
    from retrieval_bench.r3o_props import ItemProps, candidates
    from retrieval_sets.common import source
    from retrieval_sets.lexicon import check as lexicon
    t0, src = time.time(), source()
    rows = load_rows("dev", 300)
    prepare(rows, src)
    rows = [r for r in rows if r["why"] == "gold"]
    items = {r["id"]: [m["qid"] for m in r["mentions"] if m.get("span")] or [m["qid"] for m in r["mentions"]] for r in rows}
    props = ItemProps(src).get(sorted({q for v in items.values() for q in v}))
    cands = {r["id"]: candidates(items[r["id"]], props)["all"] for r in rows}
    for r in rows:
        r["held"] = bool(set(r["golds"]) & cands[r["id"]])
    arms = Arms(lexicon(), A.Grammar())
    ctxs = {r["id"]: arms.context(r) for r in rows}
    logs = {}
    for f, names in R3O_LOGS.items():
        lg = {x["id"]: x for x in json.load(open(os.path.join(LOGS, f), encoding="utf-8"))["rows"]}
        logs.update({a: lg for a in names})
    diff = check(arms, rows, cands, ctxs, logs)
    if any(d["differ"] for d in diff.values()):
        raise SystemExit(f"the arms do not reproduce R3o: {diff}")
    split = S.load_split()
    held = [r for r in rows if r["held"]]
    if sorted(r["id"] for r in held) != sorted(split["part_of"]):
        raise SystemExit("the rebuilt held rows are not the split's rows")
    inv = S.inverses(src, sorted({g for r in held for g in r["golds"]}, key=lambda p: int(p[1:])))
    cl = C.load()
    classes = {r["id"]: sorted({c for q in items[r["id"]] for c in cl["class_of"].get(q, [])}) for r in rows}
    hidden = {r["id"]: S.hide(cands[r["id"]], r["golds"], inv) for r in held}
    return {"rows": rows, "held": held, "natural": [r for r in rows if not r["held"]], "cands": cands, "hidden": hidden,
            "arms": arms, "ctxs": ctxs, "split": split, "inverse": {g: v for g, v in inv.items() if v},
            "classes": classes, "table": cl["table"], "reproduced": diff, "load_s": round(time.time() - t0, 1),
            "class_file": {k: cl[k] for k in ("made", "store_snapshot", "classes", "classes_not_in_store")}}


def records(d: dict, rows: list[dict], variants: tuple) -> dict:
    from retrieval_bench.r3g_arms import ARMS
    from retrieval_bench.r3g_eval import record
    recs = {}
    for r in rows:
        for v in variants:
            cd = d["hidden"][r["id"]] if v == "hidden" else d["cands"][r["id"]]
            recs[(r["id"], v)] = {a: record(d["arms"], a, d["ctxs"][r["id"]], cd, d["classes"][r["id"]], d["table"]) for a in ARMS}
    return recs


def unavailable(d: dict, rows: list[dict]) -> dict:
    from retrieval_bench.r3g_classes import expected
    from retrieval_bench.r3g_guards import N_MIN
    n = sum(1 for r in rows if expected(d["classes"][r["id"]], d["table"], 0.5, N_MIN) is None)
    return {"rows": len(rows), "unavailable": n, "share": round(n / len(rows), 4) if rows else None}


def head(d: dict, part: str) -> dict:
    from retrieval_bench.r3g_set import CAVEAT
    return {"what": f"order 0070: the missing-fact guard, part {part}", "date": _dt.date.today().isoformat(),
            "caveat": CAVEAT, "split_sha256": d["split"]["sha256"], "reproduced_r3o": d["reproduced"],
            "class_file": d["class_file"], "inverse_properties": d["inverse"], "load_s": d["load_s"],
            "threads": os.environ.get("OMP_NUM_THREADS")}


def part_dev(d: dict) -> dict:
    from retrieval_bench.r3g_arms import ARMS
    from retrieval_bench.r3g_eval import curve, guards_for, table_of
    from retrieval_bench.r3g_guards import choose
    rows = [r for r in d["held"] if d["split"]["part_of"][r["id"]] == "guard-dev"]
    recs = records(d, rows, ("present", "hidden"))
    curves, chosen = {}, {}
    for a in ARMS:
        for g in guards_for(a):
            if g in ("none", "G3"):
                continue
            pts = curve(rows, recs, a, g)
            curves[f"{a}|{g}"] = pts
            chosen.setdefault(a, {})[g] = choose(pts)
    for a in ARMS:
        for g in ("none", "G3") if "G3" in guards_for(a) else ("none",):
            chosen.setdefault(a, {})[g] = {"t": {}, **{k: v for k, v in curve(rows, recs, a, g)[0].items() if k != "t"},
                                           "flag": None}
    thr = {"what": "order 0070: the guards' thresholds, set on guard-dev only by the pre-registered rule (r3g_guards)",
           "date": _dt.date.today().isoformat(), "data": f"guard-dev, {len(rows)} rows of split {d['split']['sha256']}",
           "arms": {a: {g: {"t": v["t"], "flag": v["flag"], "guard_dev": {k: x for k, x in v.items() if k not in ("t", "flag")}}
                        for g, v in per.items()} for a, per in chosen.items()}}
    if os.path.exists(THRESHOLDS):
        raise SystemExit(f"{THRESHOLDS} exists: the thresholds are set once")
    json.dump(thr, open(THRESHOLDS, "w", encoding="utf-8"), indent=1)
    base = {a: table_of(rows, recs, a, "none", {}, "hidden") for a in ARMS}
    return {**head(d, "dev"), "rows": len(rows), "g2_unavailable": unavailable(d, rows), "hidden_without_guard": base,
            "thresholds": thr["arms"], "curves": curves}


def part_test(d: dict) -> dict:
    from retrieval_bench.r3g_arms import ARMS
    from retrieval_bench.r3g_eval import guards_for, handed, outcome, table_of, timings, verdict
    if glob.glob(os.path.join(LOGS, "retrieval_R3g_test_*.json")):
        raise SystemExit("guard-test has been read: each held-out read happens once")
    thr = json.load(open(THRESHOLDS, encoding="utf-8"))
    if d["split"]["sha256"] not in thr["data"]:
        raise SystemExit("the thresholds were set on another split")
    test = [r for r in d["held"] if d["split"]["part_of"][r["id"]] == "guard-test"]
    recs = records(d, d["held"], ("present", "hidden"))
    recs.update(records(d, d["natural"], ("present",)))
    out, verdicts = {}, {}
    halves = {p: [r for r in d["held"] if d["split"]["part_of"][r["id"]] == p] for p in ("guard-dev", "guard-test")}
    for a in ARMS:
        pb = table_of(d["held"], recs, a, "none", {}, "present")
        for g in guards_for(a):
            t = thr["arms"][a][g]["t"]
            res = {"t": t, "flag": thr["arms"][a][g]["flag"],
                   "guard_test_hidden": table_of(test, recs, a, g, t, "hidden"),
                   "present": table_of(d["held"], recs, a, g, t, "present"),
                   "present_by_half": {p: table_of(rs, recs, a, g, t, "present")["all"] for p, rs in halves.items()},
                   "natural": table_of(d["natural"], recs, a, g, t, "present"),
                   "natural_handed_over": {r["id"]: handed(recs[(r["id"], "present")][a], g, t).get("top")
                                           for r in d["natural"] if handed(recs[(r["id"], "present")][a], g, t).get("read")},
                   "present_false_refusals": sum(1 for r in d["held"] if outcome(recs[(r["id"], "present")][a]["base"], r["golds"])
                                                 == "hit" and handed(recs[(r["id"], "present")][a], g, t).get("refused"))}
            out[f"{a}|{g}"] = res
            verdicts[f"{a}|{g}"] = verdict(res["guard_test_hidden"], res["natural"], res["present"], pb)
    t_of = {a: {g: thr["arms"][a][g]["t"] for g in guards_for(a)} for a in ARMS}
    return {**head(d, "test"), "rows": {"guard_test": len(test), "present": len(d["held"]), "natural": len(d["natural"])},
            "g2_unavailable": {"guard_test": unavailable(d, test), "present": unavailable(d, d["held"]),
                               "natural": unavailable(d, d["natural"])},
            "verdicts": verdicts, "reads": out,
            "latency": timings(d["arms"], d["held"], d["ctxs"], d["cands"], d["classes"], d["table"], t_of)}


def part_time(d: dict) -> dict:
    from retrieval_bench.r3g_arms import ARMS
    from retrieval_bench.r3g_eval import guards_for, timings
    thr = json.load(open(THRESHOLDS, encoding="utf-8"))
    t_of = {a: {g: thr["arms"][a][g]["t"] for g in guards_for(a)} for a in ARMS}
    return {**head(d, "time"), "latency": timings(d["arms"], d["held"], d["ctxs"], d["cands"], d["classes"], d["table"], t_of)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("dev", "test", "time"), required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    d = load()
    rep = {"dev": part_dev, "test": part_test, "time": part_time}[a.part](d)
    rep["wall_s"] = round(time.time() - t0, 1)
    path = os.path.join(LOGS, f"retrieval_R3g_{a.part}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(rep.get("verdicts") or rep.get("thresholds") or rep.get("latency"), indent=1)[:6000], "->", path)


if __name__ == "__main__":
    main()
