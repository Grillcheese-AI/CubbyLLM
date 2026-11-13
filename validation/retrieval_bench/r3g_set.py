"""r3g_set -- order 0070, A: the hidden-fact set, and its split into guard-dev and guard-test.

The rows are R3o's dev rows (``retrieval_R3o_dev_2026-11-04.json``: 908 scored, 870 where the row's items carry a gold
property today). Each of the 870 is copied with its gold properties taken out of the row's candidates (``hide``): the
property stays in the whole codebook, only this row's items no longer carry it. A gold whose inverse property
(Wikidata's ``inverse property``, P1696, read off the property's own statements through the source's cache) the items
also carry takes that inverse out with it, so the fact cannot come back the other way ("drops both").

The split: by row, stratified by set and language, each stratum cut in two by seed 0 (``common.halves``: sorted ids,
shuffled by the seed, the first half guard-dev). Written to ``r3g_split.json`` beside this file with the SHA-256 of
its id lists, and committed before any guard runs (``tests/validation/test_r3g.py`` pins the hash).

Caveat, carried with every number read on this set: a hidden fact is not distributed like a missing one.

    python validation/retrieval_bench/r3g_set.py      # writes r3g_split.json (refuses to overwrite a different one)

Wired: STANDALONE (order 0070's bench).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

SEED = 0
SPLIT = os.path.join(HERE, "r3g_split.json")
SOURCE_LOG = "retrieval_R3o_dev_2026-11-04.json"
PARTS = ("guard-dev", "guard-test")
CAVEAT = "a hidden fact is not distributed like a missing one"


def hide(cands: set, golds, inverse: dict) -> set:
    """The row's candidates with its golds out, and each gold's inverse the items carry."""
    drop = set(golds)
    for g in golds:
        drop |= set(inverse.get(g) or ())
    return set(cands) - drop


def truthy(claims: dict, prop: str) -> list[str]:
    """The item values of `prop` at their best rank (preferred if any, else normal; deprecated never)."""
    sts = [s for s in (claims or {}).get(prop) or [] if s.get("rank") != "deprecated"]
    best = [s for s in sts if s.get("rank") == "preferred"] or sts
    out = []
    for s in best:
        dv = (s.get("mainsnak") or {}).get("datavalue") or {}
        if dv.get("type") == "wikibase-entityid" and (dv.get("value") or {}).get("id"):
            out.append(dv["value"]["id"])
    return list(dict.fromkeys(out))


def inverses(src, pids) -> dict[str, list[str]]:
    """{property: its inverse properties (P1696)}; a symmetric property (its own inverse) lists itself."""
    from retrieval_sets.common import entities
    ents = entities(src, sorted(set(pids), key=lambda p: int(p[1:])), props="claims", langs="en")
    return {p: truthy((ents.get(p) or {}).get("claims"), "P1696") for p in pids}


def split(strata: dict[str, list[str]], seed: int = SEED) -> dict[str, dict[str, list[str]]]:
    """{stratum: {"guard-dev": ids, "guard-test": ids}}, each stratum halved by the seed."""
    from retrieval_sets.common import halves
    out = {}
    for k in sorted(strata):
        h = halves(list(strata[k]), seed)
        out[k] = {PARTS[0]: sorted(i for i, v in h.items() if v == "dev"),
                  PARTS[1]: sorted(i for i, v in h.items() if v == "test")}
    return out


def digest(parts: dict) -> str:
    """SHA-256 of the split's id lists, as canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_split(path: str = SPLIT) -> dict:
    """The committed split, its hash checked; {"part_of": {id: part}, ...the file}."""
    d = json.load(open(path, encoding="utf-8"))
    if digest(d["strata"]) != d["sha256"]:
        raise SystemExit(f"{path}: the id lists do not match their SHA-256")
    d["part_of"] = {i: part for s in d["strata"].values() for part, ids in s.items() for i in ids}
    return d


def held_rows(log_path: str | None = None) -> list[dict]:
    """R3o's rows whose items carry a gold today (the 870), from its dev log: id, set, lang, golds."""
    path = log_path or os.path.join(os.path.dirname(HERE), "logs", SOURCE_LOG)
    return [r for r in json.load(open(path, encoding="utf-8"))["rows"] if r["why"] == "gold" and r["held"]]


def main() -> None:
    rows = held_rows()
    strata: dict[str, list[str]] = {}
    for r in rows:
        strata.setdefault(f"{r['set']}/{r['lang']}", []).append(r["id"])
    parts = split(strata)
    rep = {"what": "order 0070: the hidden-fact set's split (R3o's dev rows whose items carry the gold)",
           "made": _dt.date.today().isoformat(), "source_log": SOURCE_LOG, "seed": SEED,
           "rule": "by row, stratified by set and language; common.halves per stratum (sorted ids shuffled by the seed, "
                   "the first half guard-dev)",
           "caveat": CAVEAT, "rows": len(rows),
           "counts": {k: {p: len(v[p]) for p in PARTS} for k, v in parts.items()},
           "sha256": digest(parts), "strata": parts}
    if os.path.exists(SPLIT):
        old = json.load(open(SPLIT, encoding="utf-8"))
        if old.get("sha256") != rep["sha256"]:
            raise SystemExit(f"{SPLIT} holds another split ({old.get('sha256')}): not overwritten")
        print("the committed split is unchanged:", rep["sha256"])
        return
    json.dump(rep, open(SPLIT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep["counts"], indent=1), rep["sha256"], "->", SPLIT)


if __name__ == "__main__":
    main()
