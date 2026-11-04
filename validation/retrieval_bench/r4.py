"""r4 -- order 0046, R4: several values and time, on CronQuestions and the scoreboard's 44 + 2 rows (dev only here).

  cron     CronQuestions dev, `simple_entity` (one entity, a relation, a year; an item asked), a sample of --n (seed
           0): the item's statements of the relation, outgoing when it carries any, else incoming (who held it).
           `simple_time` (when a pair held) beside it, outside the line: the statement joining the pair, its years.
  board    the live scoreboard's rows refused for several values (44 `ambiguous_hop`) or time (2 `time_mismatch`),
           from ``everyday_webq_on_live_tense_0047_2026-10-31.json``: the seed resolved as today's source does
           (``WikidataSource.resolve``, with the plan's relation), the relation by its wording's property ids.
           Spoken values are matched to the WebQuestions gold by their EN label or alias. A value that matches no
           gold is read by hand (the order: every changed row read by hand), once per (question, value), and the
           reading is kept in ``r4_board_read.json`` beside this file with its reason: "holds" (true today and an
           answer to what was asked) or "wrong" (not true, or not what was asked). A row is right when every value
           spoken is gold or read "holds", wrong when any is read "wrong", and "to read" until every value is read.
Lines (0046): 0 wrong; at least 20 of the 46 rows spoken; at least 60% of CronQuestions single-entity dev spoken.
--rules: Cowork's four rules after dev (``r4_rules``: open ends placed only if nothing says they ended, at most 10
values, the class asked, gold through merges), read once on CronQuestions' test (``--split test``); the board re-read.

    python validation/retrieval_bench/r4.py --n 500 [--split test --rules --class-exempt serving|none --tag _x]
-> validation/logs/retrieval_R4_<split>_<date><tag>.json (ids, values and counts; no question text beyond the board's)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

ARMS = ("order", "interval", "now")
BOARD = "everyday_webq_on_live_tense_0047_2026-10-31.json"
READ = os.path.join(HERE, "r4_board_read.json")


def readings() -> dict:
    """{question: {value id: {"verdict": "holds" | "wrong", "why": ...}}}, the hand reading."""
    return json.load(open(READ, encoding="utf-8")) if os.path.exists(READ) else {}


def arms_on(vals, single, year, question, rules: dict | None = None, asked: str = "", plan_rel=None) -> dict:
    """The three arms; with `rules` ({"st", "src", "exempt", "cache"}): rule 1 where a year applies, then 3, then 2."""
    from retrieval_bench import r4_rules as R
    from retrieval_bench import r4_values as V
    if not rules:
        return {"order": V.order(vals, single), "interval": V.interval(vals, single, year),
                "now": V.now(vals, single, year, question)}
    lv = V.live(vals)
    ends = rules["st"].ends(R.ended_ids(lv, asked)) if lv else {}
    at_y, drop_y = R.drop_ended(lv, year, asked, ends, single)
    y_now = _dt.date.today().year if (year is None and V.present(question)) else year
    at_now, drop_now = R.drop_ended(lv, y_now, asked, ends, single)
    out = {"order": V.order(lv, single), "interval": V.interval(at_y, single, year),
           "now": V.now(at_now, single, year, question)}
    if year is not None:
        out["interval"]["ended"] = drop_y
    if y_now is not None:
        out["now"]["ended"] = drop_now
    for a, res in out.items():
        res = R.class_filter(rules["src"], question, res, plan_rel, rules["exempt"], rules["cache"])
        out[a] = {**R.bound(res), **{k: res[k] for k in ("ended", "class", "class_dropped") if k in res}}
    return out


def values_for(st, qid: str, pid: str) -> tuple[list | None, str, str]:
    """(values, direction, fetched): the item's own statements if it carries the property, else those naming it."""
    from retrieval_bench.r4_values import live
    o = st.out(qid, pid)
    if live(o["values"]):
        return o["values"], "out", o["fetched"]
    i = st.inc(qid, pid)
    return i["values"], ("in" if i["values"] is not None else "in unavailable"), i["fetched"]


def pid_labels(pid: str) -> set:
    if not _PA:
        from cubbyllm.sources.wikidata_aliases import PropertyAliases
        _PA.append(PropertyAliases())
    return {lab for lab, p in _PA[0]._pid_of.items() if p == pid}


_PA: list = []


def cron_rows(st, n: int, split: str = "dev", rules: dict | None = None) -> list[dict]:
    from retrieval_bench import r4_rules as R
    from retrieval_bench import r4_values as V
    from retrieval_sets.common import read_rows
    rows = [r for r in read_rows("cronquestions", split=split, lang="en") if r["extra"]["type"] == "simple_entity"]
    sample = random.Random(0).sample(rows, min(n, len(rows)))
    merges = R.merged(rules["src"], {g for r in sample for g in r["answer_qids"]}) if rules else {}
    out = []
    for r in sample:
        q, pid, year = r["mentions"][0]["qid"], r["prop"], V.asked_year(r["question"], r["extra"]["times"])
        vals, how, fetched = values_for(st, q, pid)
        arms = arms_on(vals, st.single(pid), year, r["question"], rules, q, pid_labels(pid) if rules else None)
        gold = R.gold_through_merges(set(r["answer_qids"]), merges)
        out.append({"id": r["id"], "pid": pid, "year": year, "direction": how, "fetched": fetched,
                    "n_values": len(V.live(vals)), "gold": sorted(gold),
                    "merged": sorted(g for g in r["answer_qids"] if g in merges),
                    "arms": {a: {**o, "judge": "asked" if o.get("ask") else V.judge(o["spoken"], gold)}
                             for a, o in arms.items()}})
        if len(out) % 100 == 0:
            print(f"cron: {len(out)}/{n}", flush=True)
    return out


def cron_time_rows(st, n: int, split: str = "dev") -> list[dict]:
    """`simple_time`: the statement(s) joining the pair; its years (a point, or start to end), right if within gold."""
    from retrieval_bench import r4_values as V
    from retrieval_sets.common import read_rows
    rows = [r for r in read_rows("cronquestions", split=split, lang="en") if r["extra"]["type"] == "simple_time"]
    out = []
    for r in random.Random(0).sample(rows, min(n, len(rows))):
        a, b = (m["qid"] for m in r["mentions"])
        pid = r["prop"]
        joined = [v for x, y in ((a, b), (b, a)) for v in V.live(st.out(x, pid)["values"]) if v["v"] == y]
        years: set[str] = set()
        for v in joined:
            if v.get("point"):
                years.add(str(v["point"]["y"]))
            elif v.get("start") and v.get("end"):
                years |= {str(y) for y in range(v["start"]["y"], v["end"]["y"] + 1)}
            elif v.get("start"):
                years.add(str(v["start"]["y"]))
        spoken = sorted(years) or None
        out.append({"id": r["id"], "pid": pid, "statements": len(joined), "spoken": spoken, "gold": r["answers"],
                    "judge": V.judge(spoken, set(r["answers"]))})
    return out


def board_rows(st, src, rules: dict | None = None) -> list[dict]:
    from cubbyllm.sources.wikidata_aliases import PropertyAliases
    from retrieval_bench import r4_values as V
    from retrieval_sets.common import entities, names_of
    from cubbyllm.sources.wikidata import _normalize
    log = json.load(open(os.path.join(os.path.dirname(HERE), "logs", BOARD), encoding="utf-8"))
    rows = [r for r in log["rows"] if r.get("reason") in ("ambiguous_hop", "time_mismatch")]
    pa, out, hand = PropertyAliases(), [], readings()
    src.aliases = pa                   # as the stack mounts it: the plan's relation decides among several exact hits
    for r in rows:
        res = src.resolve(r["seed"], r["plan"])
        rec = {"question": r["question"], "reason": r["reason"], "golds": r["golds"], "qid": res[0] if res else None}
        if not res:
            out.append({**rec, "why": "seed not resolved"})
            continue
        pids = [p for w in r["plan"] for p in pa.pids(w)]
        carried = [p for p in dict.fromkeys(pids) if V.live(st.out(res[0], p)["values"])]
        if len(carried) != 1:
            out.append({**rec, "pids": pids, "why": "no property carried" if not carried else "several properties"})
            continue
        pid = carried[0]
        o = st.out(res[0], pid)
        arms = arms_on(o["values"], st.single(pid), V.asked_year(r["question"]), r["question"], rules, res[0], set(r["plan"]))
        ents = entities(src, sorted({x for a in arms.values() for x in a["spoken"] or ()}), props="labels|aliases", langs="en")
        gold = {_normalize(g) for g in r["golds"]}
        rec.update(pid=pid, fetched=o["fetched"], n_values=len(V.live(o["values"])), arms={})
        for a, res_a in arms.items():
            sp = res_a["spoken"] or []
            names = {x: names_of(ents.get(x) or {}, "en") for x in sp}
            matched = {x for x in sp if gold & {_normalize(n) for n in names[x]}}
            read = {x: (hand.get(r["question"]) or {}).get(x, {}).get("verdict") for x in set(sp) - matched}
            judge = "asked" if res_a.get("ask") else "refused" if not sp else ("wrong" if "wrong" in read.values() else
                                              "right" if all(v == "holds" for v in read.values()) else "to read")
            rec["arms"][a] = {**res_a, "labels": {x: (names[x] or [x])[0] for x in sp}, "unmatched": sorted(set(sp) - matched),
                              "judge": judge}
        out.append(rec)
    return out


def tally(rows: list[dict], key=lambda r, a: r.get("arms", {}).get(a, {}).get("judge", "refused")) -> dict:
    return {a: dict(collections.Counter(key(r, a) for r in rows)) for a in ARMS}


def effects(rows: list[dict]) -> dict:
    """Per arm: rows hitting the bound (and their share), rows with values dropped by rule 1 or rule 3, golds merged."""
    out = {}
    for a in ARMS:
        arm = [r["arms"][a] for r in rows]
        out[a] = {"bound": sum(1 for x in arm if x.get("ask")),
                  "bound_share": round(sum(1 for x in arm if x.get("ask")) / max(1, len(rows)), 4),
                  "rule1_rows": sum(1 for x in arm if x.get("ended")),
                  "rule1_values": sum(len(x.get("ended") or ()) for x in arm),
                  "class_rows": sum(1 for x in arm if x.get("class_dropped"))}
    out["golds_merged"] = sum(1 for r in rows if r.get("merged"))
    return out


def main() -> None:
    from retrieval_bench.r4_statements import Statements
    from retrieval_sets.common import source
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-time", type=int, default=300)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--rules", action="store_true", help="Cowork's four rules after dev (r4_rules)")
    ap.add_argument("--class-exempt", choices=("serving", "none"), default="serving",
                    help="rule 3: keep the serving path's exemption (the class word read as the relation asked) or not")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    src = source()
    st = Statements(src)
    rules = {"st": st, "src": src, "exempt": a.class_exempt == "serving", "cache": {}} if a.rules else None
    board = board_rows(st, src, rules)
    cron = cron_rows(st, a.n, a.split, rules)
    ctime = cron_time_rows(st, a.n_time, a.split)
    spoken = lambda t: sum(v for k, v in t.items() if k not in ("refused", "asked"))                    # noqa: E731
    tb, tc = tally(board), tally(cron)
    rep = {"what": "order 0046 R4: several values and time", "date": _dt.date.today().isoformat(), "split": a.split,
           "rules": ({"on": True, "class_exempt": a.class_exempt} if a.rules else {"on": False}),
           "board": {"rows": len(board), "tally": tb, "spoken": {x: spoken(tb[x]) for x in ARMS}},
           "cron_simple_entity": {"rows": len(cron), "tally": tc,
                                  "spoken_share": {x: round(spoken(tc[x]) / max(1, len(cron)), 4) for x in ARMS},
                                  "direction": dict(collections.Counter(r["direction"] for r in cron)),
                                  "rule_effects": effects(cron) if a.rules else None},
           "cron_simple_time": {"rows": len(ctime), "tally": dict(collections.Counter(r["judge"] for r in ctime))},
           "lines": "0 wrong; >= 20 of 46 board rows spoken; >= 60% of CronQuestions simple_entity spoken",
           "board_rows": board, "cron_rows": cron, "cron_time_rows": ctime}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R4_{a.split}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("board", "cron_simple_entity", "cron_simple_time")}, indent=1), "->", path)


if __name__ == "__main__":
    main()
