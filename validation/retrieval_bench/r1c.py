"""r1c -- order 0046, R1c (Cowork, 2026-11-04): finding is the candidate set, not a shortlist. Read once, on Mintaka's
test split (EN and FR), which no arm had read.

  the set      the items under the mention's name in R1b's segment (the VM's names: labels, aliases, titles,
               redirects), exact; only when that finds nothing, the items up to 2 edits away (the typo lookup, which
               runs only on an exact miss). Unordered, whole.
  line         the gold within the set at least as often as within the API's 20 (R1's logged ranks for the same
               mentions, ``retrieval_R1_test_2026-10-28.json``)
  reported     the set's size (p50, p95, max); on the rows where R3's gold rules give one property, the set after the
               relation filter (the items that carry it, from their statements by id), its size, and whether the gold
               is kept whenever it carries the relation
A typo lookup that reaches its limit (100 keys) may have cut its farthest group short; such rows are counted.

    python validation/retrieval_bench/r1c.py
-> validation/logs/retrieval_R1c_test_<date>.json (ids, sizes and counts only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

LIMIT = 100


def pct(xs: list[int], q: float) -> int | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def the_set(hit: dict) -> tuple[list[str], str, bool]:
    """(items, how, cut): exact; else the typo groups; `cut` when the typo lookup hit its limit."""
    exact = sorted({f["text"] for f in hit.get("exact") or ()})
    if exact:
        return exact, "exact", False
    near = hit.get("near") or []
    return sorted({f["text"] for h in near for f in h["facts"]}), "typo" if near else "none", len(near) >= LIMIT


def main() -> None:
    from retrieval_bench.r1_finding import SETS, mentions_of
    from retrieval_bench.r1b_vm import batch
    from retrieval_bench.r3_gold import ClaimIds, links
    from retrieval_sets.common import data_root, read_rows, source
    from retrieval_sets.names_segment import cubelang_exe
    t0 = time.time()
    r1 = json.load(open(os.path.join(os.path.dirname(HERE), "logs", "retrieval_R1_test_2026-10-28.json"), encoding="utf-8"))
    ms, _located = mentions_of("test", r1["n"], r1["n_mintaka"], SETS)
    old = {m["key"]: m for m in r1["mentions"]}
    ms = [m for m in ms if m["set"] == "mintaka"]
    if any(m["key"] not in old or old[m["key"]]["gold"] != m["gold"] for m in ms):
        raise SystemExit("these mentions are not R1's: refusing to compare against its API ranks")
    seg, exe = os.path.join(data_root(), "r1b_segment"), cubelang_exe()
    exact = batch(seg, [(m["key"], m["text"]) for m in ms], exe, edits=0)
    miss = [m for m in ms if not exact[m["key"]]["exact"]]
    near = batch(seg, [(m["key"], m["text"]) for m in miss], exe, edits=2) if miss else {}
    rows = {r["id"]: r for lang in ("en", "fr") for r in read_rows("mintaka", split="test", lang=lang)}
    for m in ms:
        hit = near.get(m["key"]) or exact[m["key"]]
        m["set_items"], m["how"], m["cut"] = the_set(hit)
        m["row"] = rows[m["key"].rsplit("#", 1)[0]]
    claims = ClaimIds(source()).get(sorted({q for m in ms for q in [x["qid"] for x in m["row"]["mentions"]] + m["row"]["answer_qids"]}))
    for m in ms:
        ls = links([x["qid"] for x in m["row"]["mentions"]], m["row"]["answer_qids"], claims)
        m["pid"] = ls[0] if len(ls) == 1 else None
    rel_items = sorted({q for m in ms if m["pid"] for q in m["set_items"]})
    claims_rel = ClaimIds(source()).get(rel_items)
    out = {}
    for lang in ("en", "fr"):
        sub = [m for m in ms if m["lang"] == lang]
        in_set = [m for m in sub if m["gold"] in m["set_items"]]
        api20 = [m for m in sub if (old[m["key"]]["api_rank"] or 10 ** 9) <= 20]
        sizes = [len(m["set_items"]) for m in sub]
        rel = [m for m in sub if m["pid"]]
        kept_sizes, carries, kept = [], 0, 0
        for m in rel:
            kept_items = [q for q in m["set_items"] if m["pid"] in (claims_rel.get(q) or {})]
            kept_sizes.append(len(kept_items))
            if m["pid"] in (claims_rel.get(m["gold"]) or claims.get(m["gold"]) or {}):
                carries += 1
                kept += m["gold"] in kept_items
        out[lang] = {"mentions": len(sub), "gold_in_set": round(len(in_set) / len(sub), 4),
                     "gold_in_api_20": round(len(api20) / len(sub), 4),
                     "line_holds": len(in_set) >= len(api20),
                     "how": {h: sum(1 for m in sub if m["how"] == h) for h in ("exact", "typo", "none")},
                     "typo_cut_at_limit": sum(1 for m in sub if m["cut"]),
                     "set_size": {"p50": pct(sizes, .5), "p95": pct(sizes, .95), "max": max(sizes) if sizes else None},
                     "relation_given": {"rows": len(rel), "kept_size": {"p50": pct(kept_sizes, .5), "p95": pct(kept_sizes, .95),
                                                                        "max": max(kept_sizes) if kept_sizes else None},
                                        "gold_carries": carries, "gold_kept_when_it_carries": kept}}
    rep = {"what": "order 0046 R1c: finding is the candidate set (Mintaka test, read once)", "date": _dt.date.today().isoformat(),
           "api_from": "retrieval_R1_test_2026-10-28.json", "summary": out, "wall_s": round(time.time() - t0),
           "mentions": [{"key": m["key"], "gold": m["gold"], "how": m["how"], "size": len(m["set_items"]),
                         "gold_in_set": m["gold"] in m["set_items"], "api_rank": old[m["key"]]["api_rank"], "pid": m["pid"]} for m in ms]}
    path = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R1c_test_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1), "->", path)


if __name__ == "__main__":
    main()
