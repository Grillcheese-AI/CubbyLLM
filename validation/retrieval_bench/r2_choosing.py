"""r2_choosing -- order 0046, R2: which of a name's items does the question mean? On R1's candidates (``r2_items``),
each arm decides (``r2_arms``, ``r2_vm``), never by fame or order.

Arms: ``tiers`` (today's: the name, the relation, the label over an alias), ``bm25`` (+ the question's own words past a
rank-normalized margin ``tau``), ``qfhrr`` (the same words as qFHRR codes, one prototype per item, scored on grilly2:
``r2_qfhrr``), ``b1`` (VM branches: the relation's values verified per candidate; only where the relation is known --
the fame set's). Not run here, and why: the usage table between kinds (no anchor counts: the local Wikipedia text
kept no links; R1b's redirects give names, not counts).

Sets: the fame set (bare names: no single pick allowed; qualified: that item), ShadowLink (Top, Shadow, Tail), Mintaka
and WebQSP_EL (each mention's gold item). ``tau`` is set on dev: the smallest of ``--taus`` with 0 wrong picks on the
dev rows of the three entity-linking sets; test is read once, with that ``tau``.

Gate (the order): fame bare 0 single picks; qualified 0 wrong and >= 70% resolved; ShadowLink: Shadow's accuracy among
spoken picks within 5 points of Top's; Mintaka and WebQSP_EL test: no more wrong picks than tiers (and <= 1 in 50),
with more decided. Kill: any bare-name single pick, or a Shadow-Top gap over 10 points.

    python validation/retrieval_bench/r2_choosing.py --split dev [--n 300] [--taus 0.25 0.5 0.75 0.9]
    python validation/retrieval_bench/r2_choosing.py --split test --tau <dev's BM25> --tau-qfhrr <dev's qFHRR>
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


def fame_rows(name: str = "fame_set") -> list[dict]:
    """A frozen fame set's rows (``fame_set`` is seed 0, dev; ``fame_set_seed1`` is read once), its hash checked."""
    from retrieval_sets.lexicon import frozen, sha256
    path = os.path.join(os.path.dirname(HERE), "retrieval_sets", f"{name}.jsonl")
    if sha256(path) != frozen()[name]["sha256"]:
        raise SystemExit(f"{name}.jsonl moved since it was frozen")
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append({"id": r["id"], "set": "fame", "lang": r["lang"], "question": r["question"], "mention": r["name"],
                    "pids": [r["relation"]], "gold": r.get("gold_item"), "kind": r["form"], "items": r.get("items") or [],
                    "relation_words": template_words(r)})
    return out


def template_words(r: dict) -> list[str]:
    """The fame template's own words (B1q's span accounting: the relation reading's words), name and value left out."""
    from retrieval_bench.r2_arms import fold
    from retrieval_sets.fame_set import ASK, qualified_text
    n, v, i = "zqnamez", "zqvaluez", 0 if r["lang"] == "en" else 1
    if r["form"] == "qualified":
        t = qualified_text(n, r["relation"], r["discriminator"]["relation"], v, v)[i]
    else:
        t = ASK[r["relation"]][i].replace("{n}", n)
    return [w for w in fold(t or "").split() if w not in (n, v)]


def el_rows(split: str, n: int, only: dict | None = None) -> list[dict]:
    """The entity-linking rows: `n` a set and language by seed 0, or (`only`, order 0069's fresh read) exactly those ids."""
    from retrieval_sets.common import read_rows, sample
    out = []
    for name in EL:
        for lang in ("en", "fr"):
            rows = read_rows(name, split=split, lang=lang)
            if only is not None:
                keep = set(only.get(name) or ()) if lang == "en" else set()
                picked = sorted((r for r in rows if r["id"] in keep), key=lambda r: r["id"])
            else:
                picked = sample(rows, n) if rows else []
            for r in picked:
                for k, m in enumerate(r["mentions"]):
                    if m.get("qid") and m.get("text"):
                        out.append({"id": f"{r['id']}#{k}", "set": name, "lang": lang, "question": r["question"],
                                    "mention": m["text"], "pids": [], "gold": m["qid"],
                                    "kind": (r.get("extra") or {}).get("subset") or "row",
                                    "row_items": [x["qid"] for x in r["mentions"] if x.get("qid")],
                                    "answer_qids": r.get("answer_qids") or []})
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


def run(rows: list[dict], taus: list[float], use_vm: bool, qfhrr_taus: list[float] = (), b1_qfhrr_taus: list[float] = ()) -> dict:
    """Every arm on every row (``r["arms"]``); returns what the qFHRR scorer reported (its card, its numpy check)."""
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
    if session is not None:                              # B1q's qualifier facts and Mintaka's relation, given (2026-11-04)
        from retrieval_bench.r2_b1q_prep import prepare
        prepare(rows, P, kpath)
    try:
        for r in rows:
            arms = {"tiers": A.tiers(r["mention"], r["cands"], P, r["pids"])}
            for t in taus:
                arms[f"bm25@{t}"] = A.bm25(r["mention"], r["question"], r["lang"], r["cands"], P, r["pids"], tau=t)
            if session is not None and r["pids"]:
                named = A.bearing(r["mention"], r["cands"], P)                 # the VM checks the relation itself
                arms["b1"] = V.b1(session, kpath, named, r["pids"][0]) if named else {"outcome": "refuse", "why": "no candidate bears the name"}
            if session is not None:
                from retrieval_bench.r2_b1q_prep import b_arms
                arms.update(b_arms(session, kpath, r, P))
            if (arms.get("b1") or {}).get("items"):             # the branches that verify, for the comparison arm
                r["b1_items"] = list(arms["b1"]["items"])
            r["arms"] = {a: {"outcome": o["outcome"], "item": o.get("item"), "judge": judge(r, o),
                             **{k: o[k] for k in ("k", "wall_ms", "qualifier") if k in o}} for a, o in arms.items()}
            r["gold_in_cands"] = bool(r["gold"]) and r["gold"] in r["cands"]
    finally:
        if session is not None:
            session.close()
    info = {}
    if b1_qfhrr_taus and use_vm:
        from retrieval_bench.r2_b1q_prep import b1_qfhrr
        info["b1_qfhrr_scorer"] = b1_qfhrr(rows, P, list(b1_qfhrr_taus))
    if qfhrr_taus:
        from cubbyllm.worlds.fact_pool import V4_TABLE
        from retrieval_bench import r2_qfhrr as Q
        got, info = Q.arms(rows, P, list(qfhrr_taus), str(V4_TABLE))
        for r in rows:
            r["arms"].update({a: {"outcome": o["outcome"], "item": o.get("item"), "judge": judge(r, o)} for a, o in got[r["id"]].items()})
    return info


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['set']}/{r['kind']}" if r["set"] in ("fame", "shadowlink") else r["set"]].append(r)
        if r["set"] == "shadowlink" and r.get("gold_bears") is not None:     # read where the gold bears the mention
            groups[f"shadowlink/{r['kind']}/{'bears' if r['gold_bears'] else 'does not bear'}"].append(r)
        if r["set"] == "fame" and r["kind"] == "bare" and r.get("shared_today") is not None:   # rule (c), both ways
            groups[f"fame/bare/{'shared today' if r['shared_today'] else 'one bearer today'}"].append(r)
        groups[f"{r['set']}/{r['lang']}"].append(r)
    for g, rs in sorted(groups.items()):
        arms = sorted({a for r in rs for a in r["arms"]})
        out[g] = {"rows": len(rs), "gold_in_candidates": sum(r["gold_in_cands"] for r in rs),
                  **{a: dict(collections.Counter(r["arms"][a]["judge"] for r in rs if a in r["arms"])) for a in arms}}
    return out


def pick_tau(summary: dict, taus: list[float], arm: str = "bm25") -> float | None:
    """The smallest tau with 0 wrong picks on the entity-linking sets' dev rows."""
    for t in sorted(taus):
        if all(summary.get(s, {}).get(f"{arm}@{t}", {}).get("wrong", 0) == 0 for s in ("mintaka", "webqsp_el", "shadowlink/Top",
                                                                                         "shadowlink/Shadow", "shadowlink/Tail")):
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--tau", type=float, help="test: dev's tau for BM25")
    ap.add_argument("--tau-qfhrr", type=float, help="test: dev's tau for qFHRR")
    ap.add_argument("--no-vm", action="store_true")
    ap.add_argument("--no-qfhrr", action="store_true", help="skip the qFHRR arm (it runs on grilly2)")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--fame-n", type=int, default=0, help="a smoke: the first N fame rows (0: all)")
    ap.add_argument("--no-fame", action="store_true")
    ap.add_argument("--no-el", action="store_true", help="the fame set only")
    ap.add_argument("--tag", default="", help="appended to the log's name (a partial rerun keeps the full run's log)")
    ap.add_argument("--fame-set", default="fame_set", help="fame_set (seed 0, dev) or fame_set_seed1 (read once)")
    ap.add_argument("--el-fresh", action="store_true",
                    help="order 0069's fresh read: the frozen fresh EL test rows (r2_fresh), not seed 0's sample")
    ap.add_argument("--b1q-rules", nargs="*", default=[], choices=("namesake_cue", "namesake_cue_class", "article"),
                    help="order 0069 step D's rules 2 and 3 (each measured on dev before it is kept)")
    ap.add_argument("--b1q-walk", choices=("walk", "veto"), default="veto",
                    help="order 0069 B: a 'located in' chain makes a candidate hold a cue (walk) or only vetoes a pick")
    ap.add_argument("--b1-qfhrr", type=float, nargs="*", default=None,
                    help="the comparison arm (B1, then qFHRR among the branches that verify) at these margins")
    a = ap.parse_args()
    from retrieval_bench import r2_b1q
    r2_b1q.WALK, r2_b1q.RULES = a.b1q_walk, set(a.b1q_rules)
    t0 = time.time()
    test = a.split == "test"
    taus = [] if a.no_bm25 else [a.tau] if test and a.tau is not None else a.taus
    qtaus = [] if a.no_qfhrr else [a.tau_qfhrr] if test and a.tau_qfhrr is not None else a.taus
    fame = [] if a.no_fame else fame_rows(a.fame_set)
    fresh = None
    if a.el_fresh:
        from retrieval_bench.r2_fresh import ids
        fresh = ids()
    rows = (fame[:a.fame_n] if a.fame_n else fame) + ([] if a.no_el else el_rows(a.split, a.n, fresh))
    qinfo = run(rows, taus, use_vm=not a.no_vm, qfhrr_taus=qtaus, b1_qfhrr_taus=a.b1_qfhrr or ())
    summary = summarize(rows)
    rep = {"what": "order 0046 R2: choosing among a name's items", "date": _dt.date.today().isoformat(), "split": a.split,
           "n_per_set_lang": a.n, "taus": {"bm25": taus, "qfhrr": qtaus},
           "tau_from_dev": ({"bm25": pick_tau(summary, taus), "qfhrr": pick_tau(summary, qtaus, "qfhrr")} if not test
                            else {"bm25": a.tau, "qfhrr": a.tau_qfhrr}),
           "qfhrr_scorer": qinfo,
           "b1_qfhrr_margin_from_seed0": (__import__("retrieval_bench.r2_b1q_prep", fromlist=["x"]).pick_margin(
               summary, a.b1_qfhrr) if a.b1_qfhrr and a.fame_set == "fame_set" else None),
           "not_run": {"usage": "no anchor counts here (the local Wikipedia text kept no links)"},
           "summary": summary, "wall_s": round(time.time() - t0),
           "rows": [{k: r.get(k) for k in ("id", "set", "lang", "kind", "gold", "n_api", "n_local", "gold_in_cands", "arms",
                                           "gold_bears", "shared_today", "pids_given", "cues")} for r in rows]}
    from retrieval_bench.r2_gates import budget, gates
    rep["gates"] = gates(summary)
    rep["b1_budget"] = budget(rows)
    rep["args"] = {"el_fresh": a.el_fresh, "b1q_walk": a.b1q_walk, "b1q_rules": sorted(a.b1q_rules), "fame_set": a.fame_set, "no_fame": a.no_fame, "no_el": a.no_el, "no_vm": a.no_vm, "no_qfhrr": a.no_qfhrr, "no_bm25": a.no_bm25,
                   "fame_n": a.fame_n, "tag": a.tag}
    rep["vm"] = None if a.no_vm else __import__("cubbyllm.bridges.vm_identity", fromlist=["x"]).identity()   # the VM it ran on
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R2_{a.split}_{rep['date']}{a.tag}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": summary, "tau_from_dev": rep["tau_from_dev"]}, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
