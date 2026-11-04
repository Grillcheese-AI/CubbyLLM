"""r1b_vm -- order 0046, R1b: finding through the VM's names segment (``retrieval_sets.names_segment``: every EN, FR and
``mul`` label, EN and FR alias, enwiki / frwiki title and redirect of Wikidata's dump of 2026-09-28, as CubeLang
knowledge keys), scored on R1's own mentions against R1's own API ranks.

Mentions: ``r1_finding.mentions_of`` with R1's arguments (500 a set and language, Mintaka 1,000; the same typo variants,
seed 0); checked to be R1's by their keys. The API's ranks are read from R1's log (2026-10-28), not asked again.

Arms (the store's own lookups, ``cubelang knowledge batch``, the same code a program's QUERY and ``fuzzy`` run):
  vm_exact   the items under the mention's key (the VM's normalization: lowercase, punctuation to spaces), a set
  vm_edit1   + the items under keys one edit away, as one group
  vm_edit2   + the items under keys two edits away, as one group
A group enters a shortlist whole or not at all (``r1_finding.position``). When the typo lookup reaches its limit, its
farthest group may be cut short, so that group is left out.

Lines (Cowork, 2026-10-31): recall@20 at least 0.90 on Mintaka EN and 0.85 on Mintaka FR; above the API on every set
that marks its own mentions (Mintaka EN, WebQSP_EL, GraphQ_EL, ShadowLink, AmbER); at most 5 points below it on the
others. Kill: below the API on Mintaka EN or FR. The arm judged is the best of the three at @20, per set.

    python validation/retrieval_bench/r1b_vm.py --split dev
-> validation/logs/retrieval_R1b_<split>_<date>.json (ids, ranks and counts only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

ARMS = ("vm_exact", "vm_edit1", "vm_edit2")
KS = (5, 20, 100)
LIMIT = 100
OWN_MENTIONS = ("mintaka/en", "webqsp_el/en", "graphq_el/en", "shadowlink/en", "amber/en")


def groups(hit: dict, limit: int = LIMIT) -> dict[str, list[list[str]]]:
    """The arms' groups from one ``knowledge batch`` answer."""
    exact = sorted({f["text"] for f in hit["exact"]})
    near = hit.get("near") or []
    if len(near) >= limit:                                   # the farthest group may be cut short: left out
        far = max(h["edits"] for h in near)
        near = [h for h in near if h["edits"] < far]
    by = {e: sorted({f["text"] for h in near if h["edits"] == e for f in h["facts"]}) for e in (1, 2)}
    return {"vm_exact": [exact], "vm_edit1": [exact, by[1]], "vm_edit2": [exact, by[1], by[2]]}


def batch(segment: str, queries: list[tuple[str, str]], exe: str, edits: int = 2) -> dict[str, dict]:
    qp = os.path.join(tempfile.mkdtemp(), "queries.jsonl")
    with open(qp, "w", encoding="utf-8") as f:
        for k, text in queries:
            f.write(json.dumps({"id": k, "key": text}, ensure_ascii=False) + "\n")
    r = subprocess.run([exe, "knowledge", "batch", segment, qp, "--edits", str(edits), "--limit", str(LIMIT)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(r.stderr.strip())
    print(r.stderr.strip(), flush=True)
    return {d["id"]: d for d in map(json.loads, r.stdout.splitlines())}


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def summarize(ms: list[dict]) -> dict:
    rate = lambda sub, f, k: round(sum(1 for m in sub if (f(m) or 10 ** 9) <= k) / len(sub), 4) if sub else None  # noqa: E731
    out = {}
    for g in sorted({f"{m['set']}/{m['lang']}" for m in ms}):
        sub = [m for m in ms if f"{m['set']}/{m['lang']}" == g]
        ty = [m for m in sub if "typo_pos" in m]
        row = {"mentions": len(sub), "typo": len(ty), "api": {f"@{k}": rate(sub, lambda m: m["api_rank"], k) for k in KS},
               "api_typo": {f"@{k}": rate(ty, lambda m: m["typo_api_rank"], k) for k in KS},
               "r1_best_local": {f"@{k}": max(rate(sub, lambda m, a=a: m["r1_pos"][a], k) for a in m_arms(sub)) for k in KS}}
        for a in ARMS:
            row[a] = {f"@{k}": rate(sub, lambda m, a=a: m["pos"][a], k) for k in KS}
            row[a + "_typo"] = {f"@{k}": rate(ty, lambda m, a=a: m["typo_pos"][a], k) for k in KS}
        row["exact_set"] = {"p50": pct([m["n_exact"] for m in sub], 0.5), "over_20": sum(1 for m in sub if m["n_exact"] > 20)}
        out[g] = row
    return out


def m_arms(sub: list[dict]) -> list[str]:
    return sorted({a for m in sub for a in (m.get("r1_pos") or {})}) or ["exact"]


def gates(summary: dict) -> dict:
    best = lambda r: max(r[a]["@20"] or 0 for a in ARMS)                                                    # noqa: E731
    g = {"recall@20 >= 0.90 on mintaka/en": best(summary["mintaka/en"]) >= 0.90,
         "recall@20 >= 0.85 on mintaka/fr": best(summary["mintaka/fr"]) >= 0.85}
    for k, r in summary.items():
        api = r["api"]["@20"] or 0
        g[f"{k}: above the API" if k in OWN_MENTIONS else f"{k}: within 5 points of the API"] = \
            best(r) > api if k in OWN_MENTIONS else best(r) >= api - 0.05
    g["KILL: below the API on Mintaka EN or FR"] = any(best(summary[k]) < (summary[k]["api"]["@20"] or 0)
                                                       for k in ("mintaka/en", "mintaka/fr"))
    return g


def main() -> None:
    from retrieval_bench.r1_finding import SETS, mentions_of, position
    from retrieval_sets.common import data_root
    from retrieval_sets.names_segment import cubelang_exe
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--r1-log", default=None, help="R1's log for this split (its mentions' keys and API ranks)")
    a = ap.parse_args()
    t0 = time.time()
    r1 = json.load(open(a.r1_log or os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R1_{a.split}_2026-10-28.json"),
                        encoding="utf-8"))
    ms, located = mentions_of(a.split, r1["n"], r1["n_mintaka"], SETS)
    old = {m["key"]: m for m in r1["mentions"]}
    if {m["key"] for m in ms} != set(old) or any(old[m["key"]]["gold"] != m["gold"] for m in ms):
        raise SystemExit("these mentions are not R1's: refusing to compare against its API ranks")
    seg = os.path.join(data_root(), "r1b_segment")
    got = batch(seg, [(m["key"], m["text"]) for m in ms] + [(m["key"] + "~typo", m["typo"]) for m in ms if "typo" in m],
                cubelang_exe())
    us_exact, us_near = [], []
    for m in ms:
        h = got[m["key"]]
        gs = groups(h)
        m.update(pos={x: position(g, m["gold"]) for x, g in gs.items()}, n_exact=len(gs["vm_exact"][0]),
                 api_rank=old[m["key"]]["api_rank"], typo_api_rank=old[m["key"]].get("typo_api_rank"),
                 r1_pos=old[m["key"]]["pos"])
        us_exact.append(h["us_exact"])
        us_near.append(h["us_near"])
        if "typo" in m:
            m["typo_pos"] = {x: position(g, m["gold"]) for x, g in groups(got[m["key"] + "~typo"]).items()}
    summary = summarize(ms)
    man = json.load(open(os.path.join(seg, "manifest.json"), encoding="utf-8"))
    rep = {"what": "order 0046 R1b: finding through the VM's names segment", "split": a.split,
           "date": _dt.date.today().isoformat(), "segment": {k: man.get(k) for k in ("names_rev", "redirect_dumps", "shards",
                                                                                   "facts", "bytes", "counts", "rule")},
           "api_from": os.path.basename(a.r1_log or f"retrieval_R1_{a.split}_2026-10-28.json"), "located": located,
           "lookup_us": {"exact": {"p50": pct(us_exact, .5), "p95": pct(us_exact, .95)},
                         "edits2": {"p50": pct(us_near, .5), "p95": pct(us_near, .95)}},
           "summary": summary, "gates": gates(summary), "wall_s": round(time.time() - t0),
           "mentions": [{k: m.get(k) for k in ("key", "gold", "pos", "typo_pos", "api_rank", "typo_api_rank", "n_exact")} for m in ms]}
    out = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R1b_{a.split}_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": {k: {x: v[x]["@20"] for x in ("api", "r1_best_local") + ARMS} for k, v in summary.items()},
                      "gates": rep["gates"], "lookup_us": rep["lookup_us"]}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
