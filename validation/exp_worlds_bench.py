"""exp_worlds_bench -- order 0015's acceptance over what `standin/worlds_ingest.py` wrote (CPU, no network, no model).

1. **Coverage:** every manifest row has a decision in the registry (ingested, excluded with its reason, or queued),
   and per ingested dataset its rows, entities, worlds, tier, as-of and time.
2. **Fiction isolation:** 50 in-fiction facts (kin a novel states of its own people, whose names no reality world
   holds), asked as reality questions through the reality stack (the wiki world, then hdc) -> none may be spoken with
   the in-fiction value; asked naming their work ("In <work>, who was ...?") -> answered in-fiction.
3. **Exclusion:** no file of an excluded dataset is in the registry; no evaluation question (the WebQuestions and NQ
   test and validation sets) occurs verbatim in a passage world (an FTS phrase query per question of 5+ words);
   the repo's own bench logs' questions checked the same way, reported apart.
4. **No regression:** 0012's and 0014's benches, run with the stores the registry names (`--regression`), against
   their logged results.

The log keeps counts and the fiction questions' relations; no work title, passage or name.

    python validation/exp_worlds_bench.py [--root <worlds dir>] [--fiction 50] [--regression]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import pathlib
import random
import re
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE, os.path.join(ROOT, "standin", "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(HERE, "logs", f"exp_worlds_bench_{_dt.date.today().isoformat()}.json")
# The evaluation sets checked for leaks (local parquet files, `;`-separated in CUBBY_EVAL_SETS).
EVAL = [p for p in os.environ.get("CUBBY_EVAL_SETS", "").split(";") if p]


def coverage(reg) -> dict:
    import worlds_manifest as wm
    have = {n: (s, r) for n, s, r in reg.execute("SELECT name, status, report FROM datasets")}
    rows = wm.plan()
    missing = [d.dataset for d in rows if d.dataset not in have]
    status = {}
    for d in rows:
        status[have.get(d.dataset, ("missing",))[0]] = status.get(have.get(d.dataset, ("missing",))[0], 0) + 1
    return {"rows": len(rows), "decided": len(rows) - len(missing), "by_status": status, "undecided": len(missing)}


def fiction(store, n, rng) -> dict:
    """In-fiction kin facts whose subject and object no reality world names, asked both ways."""
    from ask import AskLoop
    from cubbyllm.reasoning.planner import normalize
    from question_forms import Forms, TemplateEmitter
    from sources import WikidataSource
    fw = store.fiction
    real = store.reality()
    names = set(getattr(real, "primary", real).index._by_subj)
    hdc = (getattr(real, "others", None) or [None])[0]
    known = lambda x: normalize(x) in names or (hdc is not None and hdc.label(x) is not None)   # noqa: E731
    rows = fw.db.execute("SELECT work, subj, rel, obj FROM fiction_facts WHERE rel IN ('father','mother','spouse','sibling','child')").fetchall()
    rng.shuffle(rows)
    picks, seen = [], set()
    for w, s, r, o in rows:
        if len(picks) >= n or (w, s, r) in seen or known(s) or known(o) or fw.work_named(f"In {w},") != w:
            continue
        seen.add((w, s, r)); picks.append((w, s, r, o))
    forms, loops = Forms(), {}

    def loop_for(world):                                  # one loop a world: each starts the CubeLang VM
        if id(world) not in loops:
            loops[id(world)] = AskLoop(TemplateEmitter(forms), world=world, source=WikidataSource(offline=True), lexicon=False)
        return loops[id(world)]
    out = {"asked": len(picks), "reality_spoken": 0, "reality_spoke_the_fiction_value": 0, "named_spoken": 0,
           "named_correct": 0, "relations": {}}
    for w, s, r, o in picks:
        q = forms.render(r, s, "past")[0]
        a = fw.ask(q, loop_for(real), loop_for)
        out["reality_spoken"] += bool(a["verified"])
        out["reality_spoke_the_fiction_value"] += bool(a["verified"] and normalize(a.get("answer") or "") == normalize(o))
        b = fw.ask(f"In {w}, {q[0].lower()}{q[1:]}", loop_for(real), loop_for)
        out["named_spoken"] += bool(b["verified"] and b["world"] == f"fiction:{w}")
        out["named_correct"] += bool(b["verified"] and normalize(b.get("answer") or "") == normalize(o))
        out["relations"][r] = out["relations"].get(r, 0) + 1
    return out


def time_rule_counts(store, bulk_worlds=None) -> dict:
    """0015's Addendum 2: per mounted world, the facts the time rule touches -- an undated value of a relation in
    `time_rule.CHANGES` (said "at one point", or with the world's "as of")."""
    bulk_worlds = bulk_worlds or str(pathlib.Path(os.environ.get("CUBBY_BOOKS_INDEX", ".")).parent / "bulk" / "worlds")
    from cubbyllm.reasoning.planner import normalize
    from time_rule import CHANGES
    out = {}
    real = store.reality()
    wiki = getattr(real, "primary", real)
    times = getattr(wiki, "times", None) or {}
    facts = [(f, t) for v in wiki.index._by_subj.values() for f, t in v]
    out["wiki"] = {"facts": len(facts), "touched": sum(normalize(t.rel) in CHANGES and not times.get(f) for f, t in facts),
                   "said_as": "at one point"}
    hdc = store.store_of("reality")
    if hdc and os.path.exists(hdc):
        db = sqlite3.connect(f"file:{hdc}?mode=ro", uri=True)
        by = dict(db.execute("SELECT rel, COUNT(*) FROM facts GROUP BY rel").fetchall())
        ts = dict(db.execute("SELECT rel, COUNT(*) FROM facts WHERE ts = 1 GROUP BY rel").fetchall())
        out["hdc"] = {"facts": sum(by.values()), "touched": sum(n for r, n in by.items() if normalize(r) in CHANGES),
                      "of_which_time_sensitive": sum(ts.values()), "said_as": "as of 2019"}
    if os.path.isdir(bulk_worlds):
        n = t = 0
        for f in glob.glob(os.path.join(bulk_worlds, "*.jsonl")):
            if os.path.basename(f).startswith("_"):
                continue
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                n += 1; t += bool(r.get("rel") and normalize(r["rel"]) in CHANGES and not r.get("time"))
        out["books_bulk"] = {"facts": n, "touched": t, "said_as": "at one point (hedged when attributed)"}
    if store.fiction is not None:
        rows = store.fiction.db.execute("SELECT rel, COUNT(*) FROM fiction_facts GROUP BY rel").fetchall()
        out["fiction"] = {"facts": sum(n for _r, n in rows), "touched": sum(n for r, n in rows if normalize(r) in CHANGES),
                          "said_as": "at one point, inside its work"}
    return out


def eval_questions() -> list[str]:
    import pyarrow.parquet as pq
    qs = []
    for f in EVAL:
        if os.path.exists(f):
            qs += [q for q in pq.read_table(f, columns=["question"]).column("question").to_pylist() if q]
    return qs


def bench_questions() -> list[str]:
    qs = []
    for f in glob.glob(os.path.join(HERE, "logs", "*bench*_2026-*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stack = [d]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if isinstance(x.get("question"), str):
                    qs.append(x["question"])
                stack += list(x.values())
            elif isinstance(x, list):
                stack += x
    return list(dict.fromkeys(qs))


def verbatim(store, questions) -> dict:
    """Questions of 5+ words found as a phrase in a passage world (FTS)."""
    worlds = [p for w, (k, p, _t, _a) in store.worlds.items() if p and p.endswith(".sqlite") and os.path.exists(p)
              and sqlite3.connect(f"file:{p}?mode=ro", uri=True).execute(
                  "SELECT 1 FROM sqlite_master WHERE name = 'fts'").fetchone()]
    hits, checked = {}, 0
    for p in worlds:
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        n = 0
        for q in questions:
            words = re.findall(r"[\w']+", q)
            if len(words) < 5:
                continue
            checked += 1
            try:
                n += db.execute("SELECT 1 FROM fts WHERE fts MATCH ? LIMIT 1", ('"' + " ".join(words) + '"',)).fetchone() is not None
            except sqlite3.OperationalError:
                continue
        hits[os.path.basename(p)] = n
    return {"passage_worlds": len(worlds), "questions": len(questions), "phrase_queries": checked, "hits": hits}


def regression(store) -> dict:
    """0012's and 0014's benches with the registry's stores; their headline lines against the logged runs. The hdc
    bench reads its source folder from CUBBY_HDC_SRC (local)."""
    books, hdc = store.store_of("reality:history"), store.store_of("reality")
    env = {**os.environ, "CUBBY_BOOKS_INDEX": books, "CUBBY_HDC_WORLD": hdc, "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    out = {}
    for name, cmd, log in [("books", [sys.executable, os.path.join(HERE, "exp_books_bench.py"), "--hdc", hdc,
                                       "--out", os.path.join(HERE, "logs", "_worlds_regress_books.json")], "exp_books_bench_final_2026-10-22.json"),
                           ("hdc", [sys.executable, os.path.join(HERE, "exp_hdc_bench.py"), "--src", os.environ.get("CUBBY_HDC_SRC", ""), "--db", hdc,
                                    "--out", os.path.join(HERE, "logs", "_worlds_regress_hdc.json")], "exp_hdc_bench_2026-10-22.json")]:
        subprocess.run(cmd, env=env, capture_output=True, timeout=7200)
        now = json.load(open(cmd[-1], encoding="utf-8"))
        then = json.load(open(os.path.join(HERE, "logs", log), encoding="utf-8"))
        pick = (lambda d: {k: d["bench"][k] for k in ("spoken", "flat", "hedged", "wrong_flat", "wrong_hedged")}) if name == "books" else \
               (lambda d: {"one_hop": {k: d["one_hop"][k] for k in ("spoken", "correct", "wrong")}, "flat": d["time_sensitive"]["said_flat"]})
        out[name] = {"now": pick(now), "logged": pick(then), "same": pick(now) == pick(then)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.environ.get("CUBBY_WORLDS"), required=not os.environ.get("CUBBY_WORLDS"),
                    help="the worlds folder (CUBBY_WORLDS)")
    ap.add_argument("--fiction", type=int, default=50)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    from worlds_store import WorldsStore
    t0, store = time.time(), WorldsStore(a.root)
    rep = {"coverage": coverage(store.reg)}
    print("coverage:", rep["coverage"], flush=True)
    excluded = [n for (n,) in store.reg.execute("SELECT name FROM datasets WHERE status = 'excluded'")]
    rep["exclusion"] = {"files_of_excluded_datasets": store.reg.execute(
        f"SELECT COUNT(*) FROM files WHERE dataset IN ({','.join('?' * len(excluded))})", excluded).fetchone()[0] if excluded else 0}
    rep["exclusion"]["eval"] = verbatim(store, eval_questions())
    rep["exclusion"]["repo_benches"] = verbatim(store, bench_questions())
    print("exclusion:", json.dumps(rep["exclusion"]), flush=True)
    rep["time_rule"] = time_rule_counts(store)
    print("time rule:", rep["time_rule"], flush=True)
    rep["fiction"] = fiction(store, a.fiction, random.Random(a.seed)) if store.fiction else None
    print("fiction:", rep["fiction"], flush=True)
    if a.regression:
        rep["regression"] = regression(store)
        print("regression:", rep["regression"], flush=True)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
