"""r2_misses -- order 0069, A: B1q's misses sorted by cause. No rule is written here; nothing is scored anew.

Each set is rebuilt as R2's harness builds it (``r2_choosing``: rows, candidates, profiles; ``r2_b1q_prep.prepare``:
cues, the cue names' items, the qualifier facts and the "located in" facts of the branch segment). B1q's decision is
then repeated here in Python from those facts, as the VM reads them (``r2_b1q.b1q``: a key ``<item> ~ <cue>`` held,
else the ``<item> P131`` chain walked up to MAX_HOPS; ``r2_vm.decide`` for a relation), and for each candidate it
keeps WHY it stands: the cue it holds and how (its own statement, a statement pointing at it, or the walk). Every
recomputed outcome must equal the logged one (``check``): a row that differs is reported and left out of the causes.

  sets   dev      seed 0's fame set and R2's dev rows (``retrieval_R2_dev_2026-11-04_b1q_compared.json``)
         test     R2's test rows (``retrieval_R2_test_2026-11-05_test.json``; a diagnostic of a read already made)
         seed1    the seed-1 fame set (``retrieval_R2_dev_2026-11-05_seed1.json``; the same)

The prepared rows are kept off-repo (``$CUBBY_DATA_ROOT/retrieval/r2_misses_<set>.pkl``) so the causes can be re-read.

Wired: STANDALONE (order 0069's bench).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

LOGS = {"dev": "retrieval_R2_dev_2026-11-04_b1q_compared.json", "test": "retrieval_R2_test_2026-11-05_test.json",
        "seed1": "retrieval_R2_dev_2026-11-05_seed1.json"}
SPEC = {"dev": ("dev", "fame_set", True, True), "test": ("test", None, False, True), "seed1": ("dev", "fame_set_seed1", True, False)}


def build(name: str) -> dict:
    """The set's rows, prepared as R2's harness prepares them, with the segment's facts read back."""
    from retrieval_bench import r2_arms as A
    from retrieval_bench import r2_vm as V
    from retrieval_bench.name_index import NameIndex
    from retrieval_bench.r2_b1q_prep import _ITEMS, prepare
    from retrieval_bench.r2_choosing import el_rows, fame_rows
    from retrieval_bench.r2_items import Profiles, candidates
    from retrieval_sets.common import data_root, source
    split, fame, use_fame, use_el = SPEC[name]
    rows = (fame_rows(fame) if use_fame else []) + (el_rows(split, 300) if use_el else [])
    src, ix = source(), NameIndex()
    prof, seen = Profiles(src), {}
    for i, r in enumerate(rows):
        key = (r["mention"], r["lang"])
        if key not in seen:
            seen[key] = candidates(src, ix, r["mention"], r["lang"])
        r["cands"] = seen[key]["all"]
        prof.get(r["cands"] + ([r["gold"]] if r["gold"] else []))
        if i % 500 == 0:
            print(f"{name}: {i}/{len(rows)} rows", flush=True)
    P = prof.p
    kpath = os.path.join(tempfile.mkdtemp(), "r2_segment.jsonl")
    pids = sorted({p for r in rows for p in r["pids"]})
    V.write_segment(kpath, {q: P[q] for r in rows if r["pids"] for q in r["cands"] if q in P}, pids)
    prep = prepare(rows, P, kpath)
    facts: dict[str, list[str]] = {}
    for line in open(kpath, encoding="utf-8"):
        d = json.loads(line)
        facts.setdefault(d["key"], []).append(d["text"])
    for r in rows:
        r["named"] = A.bearing(r["mention"], r["cands"], P) if r["cands"] else []
    keep = {q for r in rows for q in (r["cands"] or []) + ([r["gold"]] if r["gold"] else [])}
    out = {"rows": rows, "facts": facts, "items": dict(_ITEMS), "prep": prep,
           "profiles": {q: {"value_ids": P[q].get("value_ids") or {}} for q in keep if q in P}}
    path = os.path.join(data_root(), f"r2_misses_{name}.pkl")
    pickle.dump(out, open(path, "wb"))
    return out


def load(name: str) -> dict:
    from retrieval_sets.common import data_root
    path = os.path.join(data_root(), f"r2_misses_{name}.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else build(name)


def why_holds(q: str, cue: str, items: list[str], profiles: dict) -> str:
    """How a held qualifier holds: "own" (the candidate's statement has the cue's item as its value) or "in" (the cue's
    item states the candidate)."""
    out = {v for ids in ((profiles.get(q) or {}).get("value_ids") or {}).values() for v in ids}
    return "own" if out & set(items) else "in"


def walk(q: str, want: set, facts: dict, max_hops: int) -> int | None:
    """The hop at which the ``P131`` chain from `q` reaches one of `want`, or None."""
    seen, front = {q}, [q]
    for hop in range(1, max_hops + 1):
        nxt = [v for x in front for v in facts.get(f"{x} P131", ()) if v not in seen]
        if want & set(nxt):
            return hop
        seen |= set(nxt)
        front = nxt
        if not front:
            return None
    return None


def repeat(r: dict, facts: dict, profiles: dict, pid_mode: str = "b1q") -> dict:
    """B1q on one row from the segment's facts, with each candidate's reason. ``pid_mode``: "b1q" (the row's relation,
    if any) -- as ``r2_b1q_prep.b_arms`` runs the arm logged as ``b1q``."""
    from retrieval_bench import r2_vm as V
    from retrieval_bench.r2_b1q import MAX_HOPS, longest
    named, items_of = r.get("named") or [], r.get("cue_items") or {}
    pid = r["pids"][0] if r["pids"] else None
    if not named:
        return {"outcome": "refuse", "why": "no candidate bears the name", "cands": {}}
    checked = longest([c for c in r.get("cues") or () if items_of.get(c)])
    cands = {}
    if not checked:
        if not pid:
            return {"outcome": "undecided", "why": "no relation and no qualifier", "cands": {}, "qualifier": "none"}
        bs = [{"item": q, "values": sorted(set(facts.get(f"{q} {pid}", ())))} for q in named]
        return {**V.decide(bs), "qualifier": "none", "cands": {b["item"]: {"rel": bool(b["values"])} for b in bs}}
    bs = []
    for q in named:
        rel = sorted(set(facts.get(f"{q} {pid}", ()))) if pid else ["(no relation)"]
        how = None
        for c in checked:
            if facts.get(f"{q} ~ {c}"):
                how = {"cue_items": list(items_of[c]), "by": why_holds(q, c, items_of[c], profiles), "cue_words": len(c.split())}
                break
        if how is None:
            want = {v for c in checked for v in items_of.get(c) or ()}
            hop = walk(q, want, facts, MAX_HOPS)
            if hop is not None:
                how = {"by": "walk", "hops": hop, "cue_items": sorted(want)}
        cands[q] = {"rel": bool(rel), "holds": how}
        bs.append({"item": q, "values": rel if how else []})
    if pid:
        out = V.decide(bs)
    else:
        stand = [b["item"] for b in bs if b["values"]]
        out = {"outcome": "pick", "item": stand[0]} if len(stand) == 1 else {"outcome": "undecided", "items": stand or named}
    return {**out, "qualifier": "checked", "cands": cands, "checked": len(checked)}


def check(name: str, data: dict) -> dict:
    """Every row's repeated B1q against the log: {row id: repeated} for the rows that agree; the ones that differ."""
    from retrieval_bench.r2_choosing import judge
    log = {r["id"]: r for r in json.load(open(os.path.join(os.path.dirname(HERE), "logs", LOGS[name]), encoding="utf-8"))["rows"]}
    agree, differ = {}, []
    for r in data["rows"]:
        lg = (log.get(r["id"]) or {}).get("arms", {}).get("b1q")
        if lg is None:
            continue
        got = repeat(r, data["facts"], data["profiles"])
        if (got["outcome"], got.get("item"), judge(r, got)) == (lg["outcome"], lg.get("item"), lg["judge"]):
            agree[r["id"]] = {**got, "judge": lg["judge"], "tiers": (log[r["id"]]["arms"].get("tiers") or {}).get("judge")}
        else:
            differ.append({"id": r["id"], "logged": lg, "repeated": {k: got.get(k) for k in ("outcome", "item")}})
    return {"agree": agree, "differ": differ}


__all__ = ["build", "check", "load", "repeat", "walk", "why_holds"]
