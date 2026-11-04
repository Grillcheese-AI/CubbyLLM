"""names_check -- order 0046, R1b: the names extract (a third party's derivation of Wikidata's dump of 2026-09-28) checked
against live Wikidata before anything is scored (Cowork, 2026-10-31).

300 items drawn by seed 0 from every item the extract names (its ids, sorted, then ``random.Random(0).sample``). For
each, the extract's EN, FR and ``mul`` labels, EN and FR aliases, and enwiki / frwiki titles are compared with live
Wikidata's (by id, through the source's cache: ``wbgetentities``, with ``info`` for the last edit's time).

An item differs when any of those differs. A difference is excused only when the item was edited after the dump
(its live ``modified`` on or after 2026-09-28); every difference is reported, excused or not.
Gate: at most 1% of items (3 of 300) differ without such an edit. Otherwise the extract is rebuilt from Wikidata's own dump.

    python validation/retrieval_sets/names_check.py [--src $CUBBY_DRIVE/cubbyllm/r1b_names] [--n 300]
-> validation/logs/r1b_names_check_<date>.json

Wired: STANDALONE (a data check).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

DUMP = "2026-09-28"
FIELDS = ("label:en", "label:fr", "label:mul", "alias:en", "alias:fr", "title:en", "title:fr")


def draw(files: list[str], n: int, seed: int = 0) -> list[str]:
    """`n` item ids by seed, from every Q-id the extract names, sorted first."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    ids = pc.unique(pa.chunked_array([c for f in files for c in pq.read_table(f, columns=["id"]).column("id").chunks]))
    ids = pc.filter(ids, pc.starts_with(ids, "Q"))
    ids = pc.take(ids, pc.sort_indices(ids)).to_pylist()
    return sorted(random.Random(seed).sample(ids, n)), len(ids)


def extract_names(files: list[str], qids: list[str]) -> dict[str, dict[str, set]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out = {q: {f: set() for f in FIELDS} for q in qids}
    want = pa.array(qids)
    for f in files:
        t = pq.read_table(f)
        t = t.filter(pc.is_in(t.column("id"), value_set=want))
        for q, n, k, lg in zip(*(t.column(c).to_pylist() for c in ("id", "name", "kind", "lang"))):
            if f"{k}:{lg}" in out[q]:
                out[q][f"{k}:{lg}"].add(n)
    return out


def live_names(e: dict) -> dict[str, set]:
    lab = lambda lg: {((e.get("labels") or {}).get(lg) or {}).get("value")} - {None}                    # noqa: E731
    al = lambda lg: {a["value"] for a in (e.get("aliases") or {}).get(lg, [])}                            # noqa: E731
    ti = lambda s: {((e.get("sitelinks") or {}).get(s) or {}).get("title")} - {None}                     # noqa: E731
    return {"label:en": lab("en"), "label:fr": lab("fr"), "label:mul": lab("mul"), "alias:en": al("en"),
            "alias:fr": al("fr"), "title:en": ti("enwiki"), "title:fr": ti("frwiki")}


def compare(ext: dict[str, set], live: dict[str, set] | None) -> dict:
    if live is None:
        return {"missing_live": True}
    return {f: {"extract_only": sorted(ext[f] - live[f]), "live_only": sorted(live[f] - ext[f])}
            for f in FIELDS if ext[f] != live[f]}


def main() -> None:
    from cubbyllm.local_paths import expand
    from retrieval_sets.common import entities, source
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    src = a.src or expand("$CUBBY_DRIVE/cubbyllm/r1b_names")
    files = sorted(glob.glob(os.path.join(src, "names_*", "*.parquet")))
    qids, pool = draw(files, a.n)
    print(f"{a.n} of {pool:,} items drawn", flush=True)
    ext = extract_names(files, qids)
    live = entities(source(), qids, props="labels|aliases|sitelinks|info", langs="en|fr|mul")
    rows, c = [], {"same": 0, "differ_edited_after": 0, "differ_unexcused": 0, "missing_live": 0}
    for q in qids:
        e = live.get(q)
        diff = compare(ext[q], live_names(e) if e else None)
        modified = (e or {}).get("modified")
        after = bool(modified) and modified[:10] >= DUMP
        kind = "same" if not diff else "missing_live" if diff.get("missing_live") else \
            "differ_edited_after" if after else "differ_unexcused"
        c[kind] += 1
        rows.append({"id": q, "verdict": kind, "modified": modified, "diff": diff})
    unexcused = c["differ_unexcused"] + c["missing_live"]
    rep = {"what": "order 0046 R1b: the names extract against live Wikidata", "date": _dt.date.today().isoformat(),
           "dump": DUMP, "pool": pool, "n": a.n, "seed": 0, "counts": c,
           "gate": {"line": "at most 1% of items differ without an edit after the dump", "unexcused": unexcused,
                    "share": round(unexcused / a.n, 4), "passes": unexcused <= a.n // 100},
           "items": rows}
    path = os.path.join(os.path.dirname(HERE), "logs", f"r1b_names_check_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({"counts": c, "gate": rep["gate"]}, indent=1), "->", path)


if __name__ == "__main__":
    main()
