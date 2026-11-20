"""Order 0076, step 5: how often an answer or a refusal of arm C touches an item and property for which the store holds
two or more best-rank values that differ (Wikidata often holds two dates of birth, both normal rank), what the host
did then, and whether the record names every value. Counts only, by benchmark.

Read from a run made with ``CB_CERT_FACTS=1`` (`cert_facts`): the (item, property) pairs a question touched are its
trace facts' statements and, for a refusal on an ambiguous hop, its values' statements. For each pair the store is
asked once (read only, through its pipe, at the run's snapshot: another snapshot stops the count) for the item's
statements of that property that are best rank and not deprecated.

Per question touching such a pair:
- **what the host did:** answered with one value; refused, and with which reason (an ambiguous hop names the values
  it found);
- **whether the record names every value:** the certificate names the one fact version it rests on; an ambiguous
  hop's refusal names the values the walk found, against the store's best-rank values; the references of each value
  are in no record yet (counted).

  python validation/exp_disagreements.py --run <run dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

__wiring__ = "STANDALONE"


def pairs(rec: dict) -> dict:
    """{(item, pid): {"from": "trace" | "hop", "guids": the statements the record names}}."""
    cert = rec.get("cert") or {}
    out: dict = {}
    for src, facts in (("trace", cert.get("facts") or []), ("hop", (cert.get("hop") or {}).get("facts") or [])):
        for f in facts:
            for w in f["worlds"]:
                for s in w["statements"]:
                    p = out.setdefault((s["item"], s["pid"]), {"from": src, "guids": set()})
                    p["guids"].add(s["guid"])
    return out


def best_values(ask, item: str, pid: str, cache: dict) -> list:
    """[(guid, value, references)] of the item's statements of `pid`, best rank and not deprecated."""
    if item not in cache:
        cache[item] = ask({"op": "get", "id": item}) or {}
    e = cache[item]
    return [(s.get("guid"), json.dumps(s.get("value"), sort_keys=True), len(s.get("references") or []))
            for s in e.get("statements", []) if f"P{s.get('pid')}" == pid and s.get("best") and s.get("rank") != 0]


SAME = (460,)                                             # "said to be the same as": the store's own sameness link
PART = (361, 527)                                         # "part of" / "has part": reported apart, not sameness


def linked(ask, vals: list, cache: dict) -> dict:
    """Step 6 (Nick, 2026-11-19: "if its written 2 way I would return both"): whether the store links the differing
    values as one thing. {"same": every value joined by "said to be the same as" (either way), "part": joined only
    once "part of" / "has part" also count, "links": the statements that join them}. Values that are not items are
    never the same here (no string compared)."""
    ids = {json.loads(v).get("entity") if v.startswith("{") else None for _, v, _ in vals}
    if None in ids or len(ids) < 2:
        return {"same": False, "part": False, "links": []}
    links = []
    for q in sorted(ids):
        if q not in cache:
            cache[q] = ask({"op": "get", "id": q}) or {}
        for s in cache[q].get("statements", []):
            o = (s.get("value") or {}).get("entity") if isinstance(s.get("value"), dict) else None
            if o in ids and o != q and s.get("rank") != 0 and s.get("pid") in SAME + PART:
                links.append((q, s["pid"], o, s.get("guid")))

    def joined(pids):
        comp = {q: q for q in ids}

        def root(x):
            while comp[x] != x:
                x = comp[x]
            return x
        for a, p, b, _ in links:
            if p in pids:
                comp[root(a)] = root(b)
        return len({root(q) for q in ids}) == 1
    return {"same": joined(SAME), "part": joined(SAME + PART), "links": links}


SINGLE = ("Q19474404", "Q52060874")                     # Wikidata's single-value and single-best-value constraints


def single(ask, pid: str, cache: dict) -> bool:
    """The property carries a single-value or single-best-value constraint in the store (P2302): one value is meant."""
    if pid not in cache:
        cache[pid] = ask({"op": "get", "id": pid}) or {}
    return any(s.get("pid") == 2302 and isinstance(s.get("value"), dict) and s["value"].get("entity") in SINGLE
               for s in cache[pid].get("statements", []))


def step6(rec: dict, links: list, singles: list | None = None) -> str:
    """What step 6's design would do with this question (nothing of it is built). ``singles``: per pair, whether its
    property is meant to hold one value; a property that holds several is not a disagreement."""
    word = str(rec.get("reason") or "").split(":")[0]
    if not rec.get("spoken") and word != "ambiguous_hop":
        return f"keeps its reason ({word})"
    if singles is not None and not any(singles):
        return "answers with every value (a property meant to hold several)"
    if all(x["same"] for x in links):
        return "answers with every value (the store says they are the same)"
    if all(x["part"] for x in links):
        return "a conflict refusal (joined only by part of / has part)"
    return "a conflict refusal (cubby:Conflict, cubby:ValueConflict)"


def classify(rec: dict, touched: dict, disagree: dict) -> dict:
    """What the host did and what the record names, for a question touching a disagreeing pair."""
    hop = (rec.get("cert") or {}).get("hop")
    did = "answered with one" if rec.get("spoken") else f"refused: {str(rec.get('reason') or 'none').split(':')[0]}"
    named = []
    for key, vals in disagree.items():
        guids = touched[key]["guids"]
        values = {v for _, v, _ in vals}
        named.append("every value" if {g for g, _, _ in vals} <= guids else
                     "one value" if len(guids) == 1 else f"{len(guids)} of {len(values)} values")
    return {"did": did, "named": named, "hop listed": bool(hop)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cubby_serve.board.store_client import from_env
    client = from_env()
    if client is None:
        raise SystemExit("the store is not configured (CUBBY_WD_STORE, CUBBY_WDSTORE_EXE)")
    snap = (client.ask({"op": "snapshot"}) or {}).get("snapshot")
    cache, t = {}, defaultdict(Counter)
    for f in sorted(glob.glob(os.path.join(a.run, "c_gather_*.jsonl"))):
        meta = json.load(open(f[:-6] + ".meta.json", encoding="utf-8"))
        if meta.get("wd_store") != snap:
            raise SystemExit(f"the store is at {snap}, the run read {meta.get('wd_store')}: not counted")
        bench = meta["bench"]
        for ln in open(f, encoding="utf-8"):
            rec = json.loads(ln)
            c = t[bench]
            c["questions"] += 1
            touched = pairs(rec)
            if not touched:
                continue
            c["questions touching a stored statement"] += 1
            dis = {}
            for (item, pid) in touched:
                vals = best_values(client.ask, item, pid, cache)
                if len({v for _, v, _ in vals}) >= 2:
                    dis[(item, pid)] = vals
            if not dis:
                continue
            c["questions touching a property with two or more differing best-rank values"] += 1
            got = classify(rec, touched, dis)
            c[f"host: {got['did']}"] += 1
            singles = [single(client.ask, pid, cache) for (_, pid) in dis]
            c["questions where such a property is meant to hold one value"] += any(singles)
            c[f"step 6: {step6(rec, [linked(client.ask, v, cache) for v in dis.values()], singles)}"] += 1
            for n in got["named"]:
                c[f"record names: {n}"] += 1
            c["values with references the record does not carry"] += sum(1 for v in dis.values() for _, _, r in v if r)
    client.close()
    report = {"order": "0076", "what": "step 5: answers and refusals touching disagreeing best-rank values, by benchmark",
              "run": os.path.basename(a.run.rstrip("/\\")), "snapshot": snap,
              "by benchmark": {b: dict(sorted(c.items())) for b, c in sorted(t.items())},
              "all": dict(sorted(sum(t.values(), Counter()).items()))}
    open(a.out, "w", encoding="utf-8").write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report["all"], indent=1))


if __name__ == "__main__":
    main()
