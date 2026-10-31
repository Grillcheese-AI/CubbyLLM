"""r2_choosing -- order 0046, R2: which of a name's items does the question mean? On R1's candidates (``r2_items``),
each arm decides (``r2_arms``, ``r2_vm``), never by fame or order.

Arms: ``tiers`` (today's: the name, the relation, the label over an alias), ``bm25`` (+ the question's own words past a
rank-normalized margin ``tau``), ``b1`` (VM branches: the relation's values verified per candidate; only where the
relation is known -- the fame set's). Not run here, and why: the usage table between kinds (no anchor counts: the
local Wikipedia text kept no links; R1b's redirects give names, not counts), and the qFHRR context (it needs the
grilly2 encoder (`../grilly2`, on the local card) and one prototype per item; the next pass).

Sets: the fame set (bare names: no single pick allowed; qualified: that item), ShadowLink (Top, Shadow, Tail), Mintaka
and WebQSP_EL (each mention's gold item). ``tau`` is set on dev: the smallest of ``--taus`` with 0 wrong picks on the
dev rows of the three entity-linking sets; test is read once, with that ``tau``.

Gate (the order): fame bare 0 single picks; qualified 0 wrong and >= 70% resolved; ShadowLink: Shadow's accuracy among
spoken picks within 5 points of Top's; Mintaka and WebQSP_EL test: no more wrong picks than tiers (and <= 1 in 50),
with more decided. Kill: any bare-name single pick, or a Shadow-Top gap over 10 points.

    python validation/retrieval_bench/r2_choosing.py --split dev [--n 300] [--taus 0.25 0.5 0.75 0.9]
    python validation/retrieval_bench/r2_choosing.py --split test --tau <dev's>
-> validation/logs/retrieval_R2_<split>_<date>.json (ids and outcomes only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

EL = ("shadowlink", "mintaka", "webqsp_el")


def fame_rows() -> list[dict]:
    path = os.path.join(os.path.dirname(HERE), "retrieval_sets", "fame_set.jsonl")
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append({"id": r["id"], "set": "fame", "lang": r["lang"], "question": r["question"], "mention": r["name"],
                    "pids": [r["relation"]], "gold": r.get("gold_item"), "kind": r["form"]})
    return out


def el_rows(split: str, n: int) -> list[dict]:
    from retrieval_sets.common import read_rows, sample
    out = []
    for name in EL:
        for lang in ("en", "fr"):
            rows = read_rows(name, split=split, lang=lang)
            for r in sample(rows, n) if rows else []:
                for k, m in enumerate(r["mentions"]):
                    if m.get("qid") and m.get("text"):
                        out.append({"id": f"{r['id']}#{k}", "set": name, "lang": lang, "question": r["question"],
                                    "mention": m["text"], "pids": [], "gold": m["qid"],
                                    "kind": (r.get("extra") or {}).get("subset") or "row"})
    return out


def judge(row: dict, out: dict) -> str:
    """right / wrong / undecided / refused / single (a bare name picked)."""
    o = out["outcome"]
    if row["kind"] == "bare":
        return "single" if o == "pick" else "held"
    if o == "pick":
        return "right" if out["item"] == row["gold"] else "wrong"
    if o in ("each", "once", "several", "ask") and row["gold"] in (out.get("items") or ()):
        return "undecided"
    return "refused" if o == "refuse" else "undecided"


def run(rows: list[dict], taus: list[float], use_vm: bool) -> list[dict]:
    from retrieval_bench import r2_arms as A
    from retrieval_bench import r2_vm as V
    from retrieval_bench.name_index import NameIndex
    from retrieval_bench.r2_items import Profiles, candidates
    from retrieval_sets.common import source
    src, ix = source(), NameIndex()
    prof = Profiles(src)
    seen: dict[tuple, dict] = {}
    for i, r in enumerate(rows):
        key = (r["mention"], r["lang"])
        if key not in seen:
            seen[key] = candidates(src, ix, r["mention"], r["lang"])
        r["cands"] = seen[key]["all"]
        r["n_api"], r["n_local"] = len(seen[key]["api"]), len(seen[key]["local"])
        prof.get(r["cands"] + ([r["gold"]] if r["gold"] else []))
        if i % 200 == 0:
            print(f"{i}/{len(rows)} rows, {len(prof.p)} profiles", flush=True)
    P = prof.p
    session, kpath = None, None
    if use_vm:
        from cubbyllm.bridges.cubelang_client import CubelangSession
        pids = sorted({p for r in rows for p in r["pids"]})
        kpath = os.path.join(tempfile.mkdtemp(), "r2_segment.jsonl")
        V.write_segment(kpath, {q: P[q] for r in rows if r["pids"] for q in r["cands"] if q in P}, pids)
        session = CubelangSession()
    try:
        for r in rows:
            arms = {"tiers": A.tiers(r["mention"], r["cands"], P, r["pids"])}
            for t in taus:
                arms[f"bm25@{t}"] = A.bm25(r["mention"], r["question"], r["lang"], r["cands"], P, r["pids"], tau=t)
            if session is not None and r["pids"]:
                named = A.bearing(r["mention"], r["cands"], P)                 # the VM checks the relation itself
                arms["b1"] = V.b1(session, kpath, named, r["pids"][0]) if named else {"outcome": "refuse", "why": "no candidate bears the name"}
            r["arms"] = {a: {"outcome": o["outcome"], "item": o.get("item"), "judge": judge(r, o),
                             **{k: o[k] for k in ("k", "wall_ms") if k in o}} for a, o in arms.items()}
            r["gold_in_cands"] = bool(r["gold"]) and r["gold"] in r["cands"]
    finally:
        if session is not None:
            session.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['set']}/{r['kind']}" if r["set"] in ("fame", "shadowlink") else r["set"]].append(r)
        groups[f"{r['set']}/{r['lang']}"].append(r)
    for g, rs in sorted(groups.items()):
        arms = sorted({a for r in rs for a in r["arms"]})
        out[g] = {"rows": len(rs), "gold_in_candidates": sum(r["gold_in_cands"] for r in rs),
                  **{a: dict(collections.Counter(r["arms"][a]["judge"] for r in rs if a in r["arms"])) for a in arms}}
    return out


def pick_tau(summary: dict, taus: list[float]) -> float | None:
    """The smallest tau with 0 wrong picks on the entity-linking sets' dev rows."""
    for t in sorted(taus):
        if all(summary.get(s, {}).get(f"bm25@{t}", {}).get("wrong", 0) == 0 for s in ("mintaka", "webqsp_el", "shadowlink/Top",
                                                                                         "shadowlink/Shadow", "shadowlink/Tail")):
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--tau", type=float, help="test: dev's tau")
    ap.add_argument("--no-vm", action="store_true")
    ap.add_argument("--fame-n", type=int, default=0, help="a smoke: the first N fame rows (0: all)")
    ap.add_argument("--no-fame", action="store_true")
    ap.add_argument("--no-el", action="store_true", help="the fame set only")
    ap.add_argument("--tag", default="", help="appended to the log's name (a partial rerun keeps the full run's log)")
    a = ap.parse_args()
    t0 = time.time()
    taus = [a.tau] if a.split == "test" and a.tau is not None else a.taus
    fame = [] if a.no_fame else fame_rows()
    rows = (fame[:a.fame_n] if a.fame_n else fame) + ([] if a.no_el else el_rows(a.split, a.n))
    run(rows, taus, use_vm=not a.no_vm)
    summary = summarize(rows)
    rep = {"what": "order 0046 R2: choosing among a name's items", "date": _dt.date.today().isoformat(), "split": a.split,
           "n_per_set_lang": a.n, "taus": taus, "tau_from_dev": pick_tau(summary, taus) if a.split == "dev" else a.tau,
           "not_run": {"usage": "no anchor counts here (the local Wikipedia text kept no links)", "qfhrr": "next pass, on grilly2"},
           "summary": summary, "wall_s": round(time.time() - t0),
           "rows": [{k: r[k] for k in ("id", "set", "lang", "kind", "gold", "n_api", "n_local", "gold_in_cands", "arms")} for r in rows]}
    from retrieval_bench.r2_gates import budget, gates
    rep["gates"] = gates(summary)
    rep["b1_budget"] = budget(rows)
    rep["args"] = {"no_fame": a.no_fame, "no_el": a.no_el, "no_vm": a.no_vm, "fame_n": a.fame_n, "tag": a.tag}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R2_{a.split}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": summary, "tau_from_dev": rep["tau_from_dev"]}, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
