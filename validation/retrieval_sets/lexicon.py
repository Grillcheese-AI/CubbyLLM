"""lexicon -- order 0046's relation lexicon, frozen before any held-out row is read: every Wikidata property's label and
aliases in English and French, from the property table the repo already uses (``wikidata_properties_en_fr.json``,
fetched 2026-09-12), as {folded wording: [P-ids]} plus each property's datatype. Written off-repo
(``$CUBBY_DATA_ROOT/retrieval/relation_lexicon.json``); its SHA-256 goes into ``frozen.json`` beside this file, which
is committed. ``check()`` refuses a lexicon whose hash moved.

    python validation/retrieval_sets/lexicon.py            # build and freeze (refuses to overwrite a frozen one)
    python validation/retrieval_sets/lexicon.py --check

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import data_root, fold  # noqa: E402

__wiring__ = "STANDALONE"

FROZEN = os.path.join(HERE, "frozen.json")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen() -> dict:
    return json.load(open(FROZEN, encoding="utf-8")) if os.path.exists(FROZEN) else {}


def freeze(key: str, path: str, note: str) -> str:
    """Record `path`'s SHA-256 under `key` in frozen.json; refuses to move a key already frozen."""
    d, h = frozen(), sha256(path)
    if key in d and d[key]["sha256"] != h:
        raise SystemExit(f"{key} is frozen at {d[key]['sha256'][:12]}; this file is {h[:12]}: not overwritten")
    d[key] = {"sha256": h, "bytes": os.path.getsize(path), "file": os.path.basename(path), "frozen": d.get(key, {}).get("frozen") or time.strftime("%F %T"),
              "note": note}
    with open(FROZEN, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    return h


def build(table: str) -> dict:
    """{"by_wording": {lang: {folded: [P..]}}, "datatype": {P: type}, "label": {P: {lang: label}}, "source": ...}."""
    t = json.load(open(table, encoding="utf-8"))
    out = {"source": {"file": os.path.basename(table), "fetched": t.get("fetched"), "n_properties": t.get("n_properties")},
           "by_wording": {"en": {}, "fr": {}}, "datatype": {}, "label": {}}
    for pid, p in sorted(t["properties"].items(), key=lambda kv: int(kv[0][1:])):
        out["datatype"][pid] = p.get("datatype")
        out["label"][pid] = p.get("label") or {}
        for lang in ("en", "fr"):
            words = [(p.get("label") or {}).get(lang)] + list((p.get("aliases") or {}).get(lang, []))
            for w in words:
                k = fold(w or "")
                if k:
                    lst = out["by_wording"][lang].setdefault(k, [])
                    if pid not in lst:
                        lst.append(pid)
    return out


def path() -> str:
    return os.path.join(data_root(), "relation_lexicon.json")


def check() -> dict:
    d = frozen().get("relation_lexicon")
    if not d:
        raise SystemExit("the relation lexicon is not frozen yet")
    if sha256(path()) != d["sha256"]:
        raise SystemExit("the relation lexicon moved since it was frozen")
    return json.load(open(path(), encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "standin", "data", "out",
                                                    "wikidata_properties_en_fr.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        lx = check()
        print(f"frozen and unchanged: {sum(len(v) for v in lx['by_wording'].values())} wordings")
        return
    if "relation_lexicon" in frozen() and os.path.exists(path()):
        check()
        print("already frozen; unchanged")
        return
    lx = build(a.table)
    with open(path(), "w", encoding="utf-8") as f:
        json.dump(lx, f, ensure_ascii=False, sort_keys=True)
    h = freeze("relation_lexicon", path(), "property labels and aliases, EN and FR, folded; from the repo's property table "
               "(query.wikidata.org, 2026-09-12); frozen before any held-out row of order 0046 was read")
    n = {lang: len(v) for lang, v in lx["by_wording"].items()}
    print(f"froze the relation lexicon: {len(lx['datatype'])} properties, wordings {n}, sha256 {h}")


if __name__ == "__main__":
    main()
